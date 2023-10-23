"""This is OpenAI' Spinning Up PyTorch implementation of Soft-Actor-Critic with
minor adjustments.
For the official documentation, see below:
https://spinningup.openai.com/en/latest/algorithms/sac.html#documentation-pytorch-version
Source:
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
"""
import itertools
from copy import deepcopy

import torch
import numpy as np
from gym.spaces import Box, Tuple
from torch.optim import Adam
from functools import reduce

from src.agents.base import BaseAgent
from src.config.yamlize import yamlize, create_configurable, NameToSourcePath
from src.utils.utils import ActionSample
from src.networks.rtacnet import ConvSeparate, PopArt

from src.constants import DEVICE


@yamlize
class RTACAgent(BaseAgent):
    """Adopted from https://github.com/rmst/rtrl/blob/master/rtrl/rtac.py"""

    def __init__(
        self,
        steps_to_sample_randomly: int,
        gamma: float,
        alpha: float,
        polyak: float,
        lr: float,
        model_cfg_path: str,
        load_checkpoint_from: str = "",
        loss_alpha: float = 0.2,
        entropy_scale: float = 0.05,
        reward_scale: float = 1.0,
    ):

        super(RTACAgent, self).__init__()

        self.steps_to_sample_randomly = steps_to_sample_randomly
        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak
        self.load_checkpoint_from = load_checkpoint_from
        self.lr = lr
        self.loss_alpha = loss_alpha
        self.entropy_scale = entropy_scale
        self.reward_scale = reward_scale

        self.t = 0
        self.deterministic = False

        self.record = {"transition_actor": ""}  # rename

        self.action_space = Box(-1, 1, (2,))

        self.model = ConvSeparate(Tuple((Tuple((Box(0, 255, (3, 384, 512)), Box(low=0, high=100, shape=(1,))), None), Box(-1, 1, shape=(2,))), None), Box(-1, 1, (2,)))
        self.model.to(DEVICE)
        self.model_target = deepcopy(self.model)
        self.model_target.to(DEVICE)

        self.outputnorm = PopArt(self.model.critic_output_layers)
        self.outputnorm_target = PopArt(self.model_target.critic_output_layers)

        self.outputnorm.to(DEVICE)
        self.outputnorm_target.to(DEVICE)

        if self.load_checkpoint_from != "":
            self.load_model(self.load_checkpoint_from)

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.model_target.parameters():
            p.requires_grad = False
        


    def select_action(self, obs):
        """Select action from obs.

        Args:
            obs (np.array): Observation to act on.

        Returns:
            ActionObj: Action object.
        """
        action_obj = ActionSample()
        (oa,ob),oc = obs
        obs = ((torch.from_numpy(oa).float().to(DEVICE), ob.float().to(DEVICE)), torch.from_numpy(oc).float().to(DEVICE))
        action_obj.action = self.model.act(obs, train=(not self.deterministic))
        self.t = self.t + 1
        return action_obj

    def register_reset(self, obs):
        """
        Same input/output as select_action, except this method is called at episodal reset.
        """
        pass

    def load_model(self, path):
        """Load model from path.

        Args:
            path (str): Load model from path.
        """
        self.actor_critic.load_state_dict(torch.load(path))

    def save_model(self, path):
        """Save model to path

        Args:
            path (str): Save model to path
        """
        torch.save(self.actor_critic.state_dict(), path)


    def update(self, data):
        """Update SAC Agent given data

        Args:
            data (dict): Data from ReplayBuffer object.
        """
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        action_dist, _, hidden = self.model(o)
        actions = action_dist.rsample()
        actions_log_prob = action_dist.log_prob(actions)[:, None]

        _, target, _ = self.model_target((o2[0], actions.detach()))
        next_value_target = reduce(torch.min,target)

        value_target = (1. - d) * self.gamma * self.outputnorm_target.unnormalize(next_value_target)
        value_target += self.reward_scale * r
        value_target -= self.entropy_scale * actions_log_prob.detach()
        value_target = self.outputnorm.update(value_target)        

        values = tuple(c(h) for c, h in zip(self.model.critic_output_layers, hidden))

        assert values[0].shape == value_target.shape and not value_target.requires_grad
        loss_critic = sum(torch.nn.functional.mse_los(v, value_target) for v in values)

        # actor loss
        with torch.no_grad():
            _, next_value, _ = self.model((o2[0], actions))
        next_value = torch.min(next_value)
        loss_actor = - (1. - d) * self.gamma * self.outputnorm.unnormalize(next_value)
        loss_actor += self.entropy_scale * actions_log_prob
        assert loss_actor.shape == (a.shape[0], 1)
        loss_actor = self.outputnorm.normalize(loss_actor).mean()

        # update model
        self.optimizer.zero_grad()
        loss_total = self.loss_alpha * loss_actor + (1 - self.loss_alpha) * loss_critic
        loss_total.backward()
        self.optimizer.step()


        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.model.parameters(), self.model_target.parameters()
            ):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            
            for p, p_targ in zip(
                self.outputnorm.parameters(), self.outputnorm_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            
        
        return {
            "loss_actor": loss_actor.item(),
            "loss_critic": loss_critic.item(),
            "loss_total": loss_total.item(),   
            "outputnorm_mean": self.outputnorm.mean,
            "outputnorm_std": self.outputnorm.std         
        } 
