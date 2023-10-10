"""Network definitions for all critic functions."""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from enum import Enum
from typing import List
from src.config.yamlize import (
    yamlize,
    ConfigurableDict,
    create_configurable_from_dict,
    NameToSourcePath,
)
from src.constants import DEVICE


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """Generate MLP from inputs

    Args:
        sizes (list[int]): List of sizes
        activation (nn.Module, optional): Activation function for hidden layers. Defaults to nn.ReLU.
        output_activation (nn.Module, optional): Activation function for output layer. Defaults to nn.Identity.

    Returns:
        nn.Module: MLP
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):
    """Squashed Gaussian MLP Actor."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        """Initialize Squashed Gaussian Actor

        Args:
            obs_dim (int): Observation dimension
            act_dim (int): Action dimension
            hidden_sizes (list[int]): List of hidden sizes
            activation (nn.Module): Activation function
            act_limit (int): Action limit
        """
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        """Get action from obs.

        Args:
            obs (int): Observation
            deterministic (bool, optional): Whether to use means instead of rsample. Defaults to False.
            with_logprob (bool, optional): Whether to return log probability. Defaults to True.

        Returns:
            tuple: Tuple of action, logprob
        """
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


@yamlize
class Qfunction(nn.Module):
    """ "Multimodal Architecture Fusing State, Action, and a Speed Embedding together to regress rewards."""

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 2,
        speed_encoder_hiddens: List[int] = [8, 8],
        fusion_hiddens: List[int] = [32, 64, 64, 32, 32],
        use_speed: bool = True,
    ):
        """Initialize Q (State, Action) -> Value Regressor

        Args:
            state_dim (int, optional): State dimension. Defaults to 32.
            action_dim (int, optional): Action dimension. Defaults to 2.
            speed_encoder_hiddens (List[int], optional): List of hidden layer dims for the speed encoder. Defaults to [1,8,8].
            fusion_hiddens (List[int], optional): List of hidden layer dims for the fusion section. Defaults to [32,64,64,32,32].
            use_speed (bool, optional): Whether to include a speed encoder or not. Defaults to True.
        """
        super().__init__()

        self.state_dim = state_dim
        self.use_speed = use_speed

        if use_speed:
            self.speed_encoder = mlp([1] + speed_encoder_hiddens)
            self.regressor = mlp(
                [state_dim + speed_encoder_hiddens[-1] + action_dim]
                + fusion_hiddens
                + [1]
            )
        else:
            self.regressor = mlp([state_dim + action_dim] + fusion_hiddens + [1])

    def forward(self, obs_feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get (s,a) value estimates

        Args:
            obs_feat (torch.Tensor): Input encoded and concatenated with speed (bs, dim)
            action (torch.Tensor): Action tensor (bs, action_dim)

        Returns:
            value: torch.Tensor of dim (bs,)
        """

        if self.use_speed:
            img_embed = obs_feat[..., : self.state_dim]  # n x latent_dims
            speed = obs_feat[..., self.state_dim :]  # n x 1
            spd_embed = self.speed_encoder(speed)  # n x 16
            out = self.regressor(
                torch.cat([img_embed, spd_embed, action], dim=-1)
            )  # n x 1
        else:
            out = self.regressor(
                torch.cat([obs_feat[..., : self.state_dim], action], dim=-1)
            )

        return out.view(-1)


@yamlize
class Vfunction(nn.Module):
    """ "Multimodal Architecture Fusing State, and a Speed Embedding together to regress rewards."""

    def __init__(
        self,
        state_dim: int = 32,
        speed_encoder_hiddens: List[int] = [8, 8],
        fusion_hiddens: List[int] = [32, 64, 64, 32, 32],
        use_speed: bool = True,
    ):
        """Initialize V (State,) -> Value Regressor

        Args:
            state_dim (int, optional): State dimension. Defaults to 32.
            speed_encoder_hiddens (List[int], optional): List of hidden layer dims for the speed encoder. Defaults to [1,8,8].
            fusion_hiddens (List[int], optional): List of hidden layer dims for the fusion section. Defaults to [32,64,64,32,32].
            use_speed (bool, optional): Whether to include a speed encoder or not. Defaults to True.
        """
        super().__init__()

        self.state_dim = state_dim
        self.use_speed = use_speed

        if use_speed:
            self.speed_encoder = mlp([1] + speed_encoder_hiddens)
            self.regressor = mlp(
                [state_dim + speed_encoder_hiddens[-1]] + fusion_hiddens + [1]
            )
        else:
            self.regressor = mlp([state_dim] + fusion_hiddens + [1])

    def forward(self, obs_feat):
        """Get state value estimates

        Args:
            obs_feat (torch.Tensor): Input encoded and concatenated with speed (bs, dim)

        Returns:
            value: torch.Tensor of dim (bs,)
        """

        if self.use_speed:
            img_embed = obs_feat[..., : self.state_dim]  # n x latent_dims
            speed = obs_feat[..., self.state_dim :]  # n x 1
            spd_embed = self.speed_encoder(speed)  # n x 16
            out = self.regressor(torch.cat([img_embed, spd_embed], dim=-1))  # n x 1
        else:
            out = self.regressor(obs_feat[..., : self.state_dim])

        return out.view(-1)


class ActivationType(Enum):
    """
    Enum class to indicate the type of activation
    """

    ReLU = torch.nn.ReLU
    Tanh = torch.nn.Tanh


@yamlize
class ActorCritic(nn.Module):
    """
    The actor-critic class that allows the basic A2C to be initialized and used in the agent files. This initializes the
    actor and critic networks and then defines a wrapper function for the policy and a function to get an action from the
    action network.
    """

    def __init__(
        self,
        activation: str = "ReLU",
        critic_cfg: ConfigurableDict = {
            "name": "Qfunction",
            "config": {"state_dim": 32},
        },  ## Flag to indicate architecture for Safety_actor_critic
        state_dim: int = 32,
        action_dim: int = 2,
        max_action_value: float = 1.0,
        speed_encoder_hiddens: List[int] = [8, 8],
        fusion_hiddens: List[int] = [32, 64, 64, 32, 32],
        use_speed: bool = True,
    ):
        """
        Initialize the observation dimension and action space dimensions, as well as the actor and critic networks.
        """

        super().__init__()
        self.state_dim = state_dim
        obs_dim = state_dim
        act_dim = action_dim
        act_limit = max_action_value
        self.use_speed = use_speed

        # build policy and value functions
        if self.use_speed:
            self.speed_encoder = mlp([1] + speed_encoder_hiddens)
            self.policy = SquashedGaussianMLPActor(
                obs_dim + speed_encoder_hiddens[-1],
                act_dim,
                fusion_hiddens,
                ActivationType.__getattr__(activation).value,
                act_limit,
            )

        else:
            self.policy = SquashedGaussianMLPActor(
                obs_dim,
                act_dim,
                fusion_hiddens,
                ActivationType.__getattr__(activation).value,
                act_limit,
            )

        if critic_cfg["name"] == "Qfunction":
            self.q1 = create_configurable_from_dict(
                critic_cfg, NameToSourcePath.network
            )
            self.q2 = create_configurable_from_dict(
                critic_cfg, NameToSourcePath.network
            )
        elif critic_cfg["name"] == "Vfunction":
            self.v = create_configurable_from_dict(critic_cfg, NameToSourcePath.network)

    def pi(self, obs_feat, deterministic=False):
        """
        Wrapper around the policy. Helps manage dimensions and add/remove features from the input space.
        """

        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        if self.use_speed:
            img_embed = obs_feat[..., : self.state_dim]
            speed = self.speed_encoder(obs_feat[..., self.state_dim :])
            feat = torch.cat([img_embed, speed], dim=-1)

        else:
            img_embed = obs_feat[..., : self.state_dim]  # n x latent_dims
            feat = torch.cat(
                [
                    img_embed,
                ],
                dim=-1,
            )
        return self.policy(feat, deterministic, True)

    def act(self, obs_feat, deterministic=False):
        """
        Uses the policy to get and return an action on the appropriate device in the right format.
        """
        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        with torch.no_grad():
            if self.use_speed:
                img_embed = obs_feat[..., : self.state_dim]
                speed = self.speed_encoder(obs_feat[..., self.state_dim :])
                feat = torch.cat([img_embed, speed], dim=-1)
            else:
                img_embed = obs_feat[..., : self.state_dim]  # n x latent_dims
                feat = torch.cat(
                    [
                        img_embed,
                    ],
                    dim=-1,
                )
            a, _ = self.policy(feat, deterministic, False)
            a = a.squeeze(0)
        return a.numpy() if a.device == "cpu" else a.cpu().numpy()


@yamlize
class IQN(nn.Module):
    """
    Define the Implicit Quantile Network. Taken from the really clear https://github.com/BY571/IQN-and-Extensions/blob/master/IQN-DQN.ipynb.
    Motivation - See if a bang-bang controller will work for our purpose.
    """

    def __init__(
        self,
        state_size: int = 32,
        action_size: int = 9,
        tau_num: int = 32,
        n_cos: int = 64,
        hidden_size: int = 64,
        speed_encoder_hiddens: List[int] = [8, 8],
        use_speed: bool = True,
    ):
        super(IQN, self).__init__()
        self.state_dim = state_size
        self.action_size = action_size
        self.K = tau_num
        self.n_cos = n_cos
        self.layer_size = hidden_size
        self.pis = (
            torch.FloatTensor([np.pi * i for i in range(self.n_cos)])
            .view(1, 1, self.n_cos)
            .to(DEVICE)
        )  # Starting from 0 as in the paper

        self.cos_embedding = nn.Linear(self.n_cos, hidden_size)
        self.ff_1 = nn.Linear(hidden_size, hidden_size)
        self.ff_2 = nn.Linear(hidden_size, action_size)

        self.use_speed = use_speed

        if self.use_speed:
            self.speed_encoder = mlp([1] + speed_encoder_hiddens)
            self.head = nn.Linear(
                self.state_dim + speed_encoder_hiddens[-1], hidden_size
            )
        else:
            self.head = nn.Linear(self.state_dim, hidden_size)

    def _calc_cos(self, batch_size):
        """
        Calculate cosine values based on tau, which is the number of samples from this distribution.
        """
        taus = (
            torch.rand(batch_size, self.K).to(DEVICE).unsqueeze(-1)
        )  # (batch_size, n_tau, 1)
        cos = torch.cos(taus * self.pis)

        assert cos.shape == (batch_size, self.K, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def forward(self, input):
        """
        Calculate quantiles using tau ( sample count ), and action space.

        Returns:
        quantiles, which have shape bs,tau,action_space
        tau, which has shape bs, tau, 1
        """
        batch_size = input.shape[0]

        if self.use_speed:
            img_embed = input[..., : self.state_dim]
            speed = self.speed_encoder(input[..., self.state_dim :])
            input = torch.cat([img_embed, speed], dim=-1)
        else:
            img_embed = input[..., : self.state_dim]

        x = torch.relu(self.head(input))
        cos, taus = self._calc_cos(batch_size)  # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * self.K, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(
            batch_size, self.K, self.layer_size
        )  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * self.K, self.layer_size)

        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)

        return out.view(batch_size, self.K, self.action_size), taus
