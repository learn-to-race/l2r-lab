import torch
import torch.nn as nn
from src.networks.network_baselines import (
    MLPCategoricalActor,
    MLPGaussianActor,
    SquashedGaussianMLPActor,
)
from enum import Enum
from typing import List
from src.config.yamlize import (
    yamlize,
    ConfigurableDict,
    create_configurable_from_dict,
    NameToSourcePath,
)


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


# Not currently working
class DuelingNetwork(nn.Module):
    """
    Further modify from Qfunction to
        - Add an action_encoder
        - Separate state-dependent value and advantage
            Q(s, a) = V(s) + A(s, a)
    """

    def __init__(self, cfg):
        """
        Initialize the layers for the Dueling Network
        """

        super().__init__()
        self.cfg = cfg

        self.speed_encoder = mlp([1] + [8, 8])
        self.action_encoder = mlp([2] + [64, 64, 32])

        n_obs = 32 + [8, 8][-1]
        # self.V_network = mlp([n_obs] + [32,64,64,32,32] + [1])
        self.A_network = mlp([n_obs + [64, 64, 32][-1]] + [32, 64, 64, 32, 32] + [1])
        # self.lr = cfg['resnet']['LR']

    # TODO: We're not currently using the advantage????
    def forward(self, obs_feat, action, advantage_only=False):
        """
        Get image, speed and action encoding and get the Value by passing through an MLP
        """

        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        img_embed = obs_feat[..., :32]  # n x latent_dims
        speed = obs_feat[..., 32:]  # n x 1
        spd_embed = self.speed_encoder(speed)  # n x 16
        action_embed = self.action_encoder(action)

        out = self.A_network(torch.cat([img_embed, spd_embed, action_embed], dim=-1))
        """
        if advantage_only == False:
            V = self.V_network(torch.cat([img_embed, spd_embed], dim = -1)) # n x 1
            out += V
        """
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
        self.action_dim = action_dim
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
            img_embed = obs_feat[..., :self.state_dim]
            speed = self.speed_encoder(obs_feat[..., self.state_dim:])
            feat = torch.cat([img_embed, speed], dim=-1)

        else:
            img_embed = obs_feat[..., :self.state_dim]  # n x latent_dims
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
                img_embed = obs_feat[..., :self.state_dim]
                speed = self.speed_encoder(obs_feat[..., self.state_dim:])
                feat = torch.cat([img_embed, speed], dim=-1)
            else:
                img_embed = obs_feat[..., :self.state_dim]  # n x latent_dims
                feat = torch.cat(
                    [
                        img_embed,
                    ],
                    dim=-1,
                )
            a, _ = self.policy(feat, deterministic, False)
            a = a.squeeze(0)
        return a.numpy() if a.device == "cpu" else a.cpu().numpy()


class PPOMLPActorCritic(nn.Module):
    """
    The Actor-Critic for PPO. Like class ActorCritic, it initializes action and observation space dimensions and the actor
    and critic networks. It also defines the wrapper for the policy and a function to get action.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        cfg,
        activation=nn.Tanh,
        latent_dims=None,
        device="cpu",
    ):
        """
        Initialize the observation and action space dimensions and the actor and critic networks.
        """

        super().__init__()

        obs_dim = observation_space.shape[0] if latent_dims is None else latent_dims
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.speed_encoder = mlp([1] + [8, 8])
        self.policy = SquashedGaussianMLPActor(
            obs_dim, act_dim, [64, 64, 32], activation, act_limit
        )

        # build value function
        self.v = Vfunction(cfg)

        self.to(device)
        self.device = device

    def pi(self, obs_feat, deterministic=False):
        """
        Wrapper around the policy. Helps manage dimensions and add/remove features from the input space.
        """

        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        img_embed = obs_feat[..., :32]  # n x latent_dims
        # speed = obs_feat[..., 32:]  # n x 1
        # spd_embed = self.speed_encoder(speed)  # n x 8
        feat = torch.cat(
            [
                img_embed,
            ],
            dim=-1,
        )
        return self.policy(feat, deterministic, True)

    def step(self, obs, deterministic=False):
        """
        Uses the policy to get and return an action on the appropriate device in the right format.
        """
        with torch.no_grad():
            img_embed = obs[..., :32]  # n x latent_dims
            # speed = obs_feat[..., 32:] # n x 1
            # raise ValueError(obs_feat.shape, img_embed.shape, speed.shape)
            # pdb.set_trace()
            # spd_embed = self.speed_encoder(speed) # n x 8
            feat = img_embed
            a, logp_a = self.policy(feat, deterministic, True)
            a = a.squeeze(0)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()
