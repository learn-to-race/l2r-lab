import itertools
from multiprocessing.sharedctypes import Value
import queue, threading
from copy import deepcopy
from src.loggers.FileLogger import FileLogger

import torch
import numpy as np
from gym.spaces import Box
from torch.optim import Adam

from src.agents.base import BaseAgent
from src.config.yamlize import yamlize
from src.networks.critic import PPOMLPActorCritic
from src.utils.utils import ActionSample

from src.constants import DEVICE

from src.config.parser import read_config
from src.config.schema import agent_schema

from src.utils.envwrapper import EnvContainer


@yamlize
class PPOAgent(BaseAgent):
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
        lr: float,
        clip_ratio: float,
    ):
        super(PPOAgent, self).__init__()
        self.steps_to_sample_randomly = steps_to_sample_randomly
        self.record_dir = record_dir
        self.track_name = track_name
        self.experiment_name = experiment_name
        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak
        self.make_random_actions = make_random_actions
        self.checkpoint = checkpoint
        self.load_checkpoint = load_checkpoint
        self.model_save_path = model_save_path
        self.lr = lr
        self.clip_ratio = clip_ratio

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
        self.train_pi_iters = 80
        self.train_v_iters = 80

        self.metadata = {}
        self.record = {"transition_actor": ""}

        self.action_space = Box(-1, 1, (2,))
        self.act_dim = self.action_space.shape[0]
        self.obs_dim = 32

        self.actor_critic = PPOMLPActorCritic(
            self.obs_dim,
            self.action_space,
            None,
            latent_dims=self.obs_dim,
            device=DEVICE,
        )

        self.target_kl = 0.01

        if self.checkpoint and self.load_checkpoint:
            self.load_model(self.checkpoint)

        self.actor_critic_target = deepcopy(self.actor_critic)

        self.v_params = itertools.chain(self.actor_critic.v.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.actor_critic.policy.parameters(), lr=self.lr)
        self.v_optimizer = Adam(self.v_params, lr=self.lr)
        self.pi_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pi_optimizer, 1, gamma=0.5
        )

    def select_action(self, obs, encode=False) -> np.array:
        action_obj = ActionSample()
        if self.t > self.steps_to_sample_randomly:
            a, v, logp = self.actor_critic.step(obs.to(DEVICE))
            a = a  # numpy array...
            action_obj.action = a
            action_obj.value = v
            action_obj.logp = logp
            self.record["transition_actor"] = "learner"
        else:
            a = self.action_space.sample()
            # logp = np.ones((self.action_space.shape[0], ))/self.action_space.shape[0]
            logp = np.ones((1,))
            # TODO: add default value after getting value shape
            v = np.ones((1,))
            action_obj.action = a
            action_obj.logp = logp
            action_obj.value = v
            self.record["transition_actor"] = "random"
        self.t = self.t + 1
        return action_obj

    def register_reset(self, obs) -> np.array:
        self.deterministic = True
        self.t = 1e6

    def compute_loss_pi(self, data):

        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        # Policy loss
        pi, logp = self.actor_critic.pi(obs.to(DEVICE))
        # logp = logp.cpu().numpy()
        # pi = pi.cpu()
        logp_old = logp_old.to(DEVICE)
        adv = adv.to(DEVICE)
        ratio = torch.exp(logp - logp_old)
        if ratio.isnan().any().item():
            print("ratio is nan ---")
            print(ratio)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        # print("entropy", pi.entropy)
        # ent = pi.entropy.mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        # pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        pi_info = dict(kl=approx_kl, cf=clipfrac)
        return loss_pi, pi_info

    def compute_loss_v(self, data):
        ## Check this.
        obs, ret = data["obs"], data["ret"]
        ret = ret.to(DEVICE)
        return ((self.actor_critic.v(obs.to(DEVICE)) - ret) ** 2).mean()

    def update(self, data):

        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info["kl"]
            if kl > 1.5 * self.target_kl:
                # print(next(self.actor_critic.pi.mu_net.parameters()))
                # self.file_logger('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            self.pi_optimizer.step()
        # print(next(self.actor_critic.pi.mu_net.parameters()))
        # logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_v_iters):
            self.v_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.v_optimizer.step()

    def load_model(self, path):
        self.actor_critic.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.actor_critic.state_dict(), path)
