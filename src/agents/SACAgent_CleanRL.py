# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import Box
from src.config.yamlize import yamlize
from src.utils.utils import ActionSample, RecordExperience
from torch.utils.tensorboard import SummaryWriter
from src.agents.base import BaseAgent
from src.constants import DEVICE

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((2.0 ) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((0.0) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1,-1))
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        if x.shape[0] == 1:
            action = action.squeeze()
        return action, log_prob, mean


@yamlize
class CleanSACAgent(BaseAgent, nn.Module):
    """Adopted from https://github.com/learn-to-race/l2r/blob/main/l2r/baselines/rl/sac.py"""

    def __init__(
        self,
        q_lr:float = 1e-3,
        policy_lr:float = 3e-4,
        autotune:bool = True,
        state_dim: int = 24,
        action_dim: int = 4,
        alpha: float = 0.2,
        gamma: float = 0.99,
        tau: float = 0.005,
        tnf: int = 1,
        pf: int = 2,
        sample_randomly: int = 5000
    ):
        super(CleanSACAgent, self).__init__()
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.qf1 = SoftQNetwork(state_dim, action_dim).to(DEVICE)
        self.qf2 = SoftQNetwork(state_dim, action_dim).to(DEVICE)
        self.qf1_target = SoftQNetwork(state_dim, action_dim).to(DEVICE)
        self.qf2_target = SoftQNetwork(state_dim, action_dim).to(DEVICE)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=policy_lr)
        self.action_space = Box(-1, 1, (4,))
        self.policy_frequency = pf
        self.target_network_frequency = tnf
        self.steps_to_sample_randomly = sample_randomly
        self.autotune = autotune
        self.load_checkpoint = False
        self.deterministic = False
        self.model_save_path = './'
        self.gamma = gamma
        self.tau = tau
        self.t = 0

        self.record = {}

        if autotune:
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(DEVICE)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr)
        
        else:
            self.alpha = alpha

    def select_action(self, obs) -> np.array:
        action_obj = ActionSample()
        if self.t > self.steps_to_sample_randomly:
            a = self.actor.get_action(obs.to(DEVICE))[0 if not self.deterministic else 2]
            action_obj.action = a.detach().cpu().numpy()
            #print(action_obj.action)
            self.record["transition_actor"] = "learner"
        else:
            a = self.action_space.sample()
            action_obj.action = a
            #print(action_obj.action)
            self.record["transition_actor"] = "random"
        self.t = self.t + 1
        return action_obj

    def update(self, data):

        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(o2)
            qf1_next_target = self.qf1_target(o2, next_state_actions)
            qf2_next_target = self.qf2_target(o2, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = r.flatten() + (1 - d.flatten()) * self.gamma * (min_qf_next_target).view(-1)

        qf1_a_values = self.qf1(o, a).view(-1)
        qf2_a_values = self.qf2(o, a).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if self.t % self.policy_frequency == 0:  # TD 3 Delayed update support
            for _ in range(
                self.policy_frequency
            ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi, _ = self.actor.get_action(o)
                qf1_pi = self.qf1(o,  pi)
                qf2_pi = self.qf2(o, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(o)
                    alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    alpha = self.log_alpha.exp().item()

        # update the target networks
        if self.t % self.target_network_frequency == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


