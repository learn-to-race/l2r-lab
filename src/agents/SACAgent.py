"""This is OpenAI' Spinning Up PyTorch implementation of Soft-Actor-Critic with
minor adjustments.
For the official documentation, see below:
https://spinningup.openai.com/en/latest/algorithms/sac.html#documentation-pytorch-version
Source:
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
"""
import itertools
from multiprocessing.sharedctypes import Value
import queue, threading
from copy import deepcopy

import torch
import numpy as np
from gym.spaces import Box
from torch.optim import Adam

from src.agents.base import BaseAgent
from src.config.yamlize import yamlize, create_configurable, NameToSourcePath
from src.encoders.vae import VAE
from src.utils.utils import ActionSample, RecordExperience

from src.constants import DEVICE

from src.config.parser import read_config
from src.config.schema import agent_schema

from src.utils.envwrapper import EnvContainer


@yamlize
class SACAgent(BaseAgent):
    """Adopted from https://github.com/learn-to-race/l2r/blob/main/l2r/baselines/rl/sac.py"""

    def __init__(
        self,
        steps_to_sample_randomly: int,
        record_dir: str,
        track_name: str,
        experiment_name: str,
        gamma: float,
        alpha: float,
        polyak: float,
        make_random_actions: bool,
        checkpoint: str,
        load_checkpoint: bool,
        model_save_path: str,
        actor_critic_cfg: str,
        lr: float,
    ):
        super(SACAgent, self).__init__()

        self.steps_to_sample_randomly = steps_to_sample_randomly
        self.record_dir = record_dir
        self.track_name = track_name
        self.experiment_name = experiment_name
        self.gamma = gamma
        self.alpha = 1.0
        self.polyak = polyak
        self.make_random_actions = make_random_actions
        self.checkpoint = checkpoint
        self.load_checkpoint = load_checkpoint
        self.model_save_path = model_save_path
        self.lr = lr
        self.target_entropy = -4.0
        self.log_ent_coef = torch.log(torch.ones(1, device=DEVICE) * self.alpha).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.lr)

        self.save_episodes = True
        self.episode_num = 0
        self.best_ret = 0
        self.t = 0
        self.deterministic = False
        self.atol = 1e-3
        self.store_from_safe = False
        self.pi_scheduler = None
        self.t_start = 0
        self.best_pct = 0

        self.record = {"transition_actor": ""}

        self.actor_critic = create_configurable(
            actor_critic_cfg, NameToSourcePath.network
        )
        self.actor_critic.to(DEVICE)

        self.action_space = Box(-1, 1, (self.actor_critic.action_dim,))
        self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)

        if self.checkpoint and self.load_checkpoint:
            self.load_model(self.checkpoint)

        self.actor_critic_target = deepcopy(self.actor_critic)
        self.q_params = itertools.chain(
            self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters()
        )

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.actor_critic.policy.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.pi_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pi_optimizer, 1, gamma=0.5
        )

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_critic_target.parameters():
            p.requires_grad = False

    def select_action(self, obs):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        action_obj = ActionSample()
        if self.t > self.steps_to_sample_randomly:
            a = self.actor_critic.act(obs.to(DEVICE), self.deterministic)
            if a.shape == ():
                # In case a is a scalar
                a = np.array([a])
            action_obj.action = a
            self.record["transition_actor"] = "learner"
        else:
            a = self.action_space.sample()
            action_obj.action = a
            self.record["transition_actor"] = "random"
        self.t = self.t + 1
        return action_obj

    def register_reset(self, obs) -> np.array:
        """
        Same input/output as select_action, except this method is called at episodal reset.
        """
        # camera, features, state = obs
        self.deterministic = True  # TODO: Confirm that this makes sense.
        self.t = 1e6

    def load_model(self, path_or_checkpoint):
        if isinstance(path_or_checkpoint, str):
            self.actor_critic.load_state_dict(torch.load(path_or_checkpoint))
        else:
            self.actor_critic.load_state_dict(path_or_checkpoint)

    def state_dict(self):
        """Emulate torch behavior; note to self to remove once better refactor comes in."""
        return self.actor_critic.state_dict()

    def save_model(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    # def compute_loss_q(self, data):

    #     """Set up function for computing SAC Q-losses."""
    #     o, a, r, o2, d = (
    #         data["obs"],
    #         data["act"],
    #         data["rew"],
    #         data["obs2"],
    #         data["done"],
    #     )

    #     q1 = self.actor_critic.q1(o, a)
    #     q2 = self.actor_critic.q2(o, a)

    #     # Bellman backup for Q functions
    #     with torch.no_grad():
    #         # Target actions come from *current* policy
    #         a2, logp_a2 = self.actor_critic.pi(o2)

    #         # Target Q-values
    #         q1_pi_targ = self.actor_critic_target.q1(o2, a2)
    #         q2_pi_targ = self.actor_critic_target.q2(o2, a2)
    #         q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
    #         backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

    #     # MSE loss against Bellman backup
    #     loss_q1 = ((q1 - backup) ** 2).mean()
    #     loss_q2 = ((q2 - backup) ** 2).mean()
    #     loss_q = loss_q1 + loss_q2

    #     # Useful info for logging
    #     q_info = dict(
    #         Q1Vals=q1.detach().cpu().numpy(), Q2Vals=q2.detach().cpu().numpy()
    #     )

    #     return loss_q, q_info

    
    def compute_loss_pi(self, data):
        """Set up function for computing SAC pi loss."""
        o = data["obs"]
        pi, logp_pi = self.actor_critic.pi(o)
        q1_pi = self.actor_critic.q1(o, pi)
        q2_pi = self.actor_critic.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
        
        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    #def compute_loss_ent(self, data):
    #    """Set up function for computing temperature loss."""
    #    o = data["obs"]
    #    pi, logp_pi = self.actor_critic.pi(o)
    #    self.alpha = torch.exp(self.log_ent_coef.detach())
    #    ent_coef_loss = -(self.log_ent_coef * (logp_pi + self.target_entropy).detach()).mean()
    #    return ent_coef_loss
    
    def update(self, data):
        
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        pi, logp_pi = self.actor_critic.pi(o)
        self.alpha = torch.exp(self.log_ent_coef.detach())
        ent_coef_loss = -(self.log_ent_coef * (logp_pi + self.target_entropy).detach()).mean()

        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        with torch.no_grad():
            q1_pi_targ = self.actor_critic_target.q1(o2, pi)
            q2_pi_targ = self.actor_critic_target.q2(o2, pi)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_pi)

        q1 = self.actor_critic.q1(o, a)
        q2 = self.actor_critic.q2(o, a)
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        q1_pi = self.actor_critic.q1(o, pi)
        q2_pi = self.actor_critic.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.actor_critic.parameters(), self.actor_critic_target.parameters()
            ):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        
        return [loss_q.item(), loss_pi.item(), ent_coef_loss.item()]

    def update_best_pct_complete(self, info):
        if self.best_pct < info["metrics"]["pct_complete"]:
            for cutoff in [93, 100]:
                if (self.best_pct < cutoff) & (
                    info["metrics"]["pct_complete"] >= cutoff
                ):
                    self.pi_scheduler.step()
            self.best_pct = info["metrics"]["pct_complete"]

    def add_experience(
        self,
        action,
        camera,
        next_camera,
        done,
        env,
        feature,
        next_feature,
        info,
        reward,
        state,
        next_state,
        step,
    ):
        self.recording = {
            "step": step,
            "nearest_idx": env.nearest_idx,
            "camera": camera,
            "feature": feature.detach().cpu().numpy(),
            "state": state,
            "action_taken": action,
            "next_camera": next_camera,
            "next_feature": next_feature.detach().cpu().numpy(),
            "next_state": next_state,
            "reward": reward,
            "episode": self.episode_num,
            "stage": "training",
            "done": done,
            "transition_actor": self.record["transition_actor"],
            "metadata": info,
        }
        return self.recording
