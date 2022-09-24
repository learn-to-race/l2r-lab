import torch
import numpy as np
from typing import Tuple

from src.config.yamlize import yamlize
from src.constants import DEVICE


@yamlize
class SimpleReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int, batch_size: int):

        self.obs_buf = np.zeros(
            (size, obs_dim), dtype=np.float32
        )  # +1:spd #core.combined_shape(size, obs_dim)
        self.obs2_buf = np.zeros(
            (size, obs_dim), dtype=np.float32
        )  # +1:spd #core.combined_shape(size, obs_dim)
        self.act_buf = np.zeros(
            (size, act_dim), dtype=np.float32
        )  # core.combined_shape(size, act_dim)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.batch_size = batch_size
        self.weights = None

    def store(self, obs, act, rew, next_obs, done):
        # pdb.set_trace()

        def convert(arraylike):
            obs = arraylike
            if isinstance(obs, torch.Tensor):
                if obs.requires_grad:
                    obs = obs.detach()
                obs = obs.cpu().numpy()
            return obs

        self.obs_buf[self.ptr] = convert(obs)
        self.obs2_buf[self.ptr] = convert(next_obs)
        self.act_buf[self.ptr] = act  # .detach().cpu().numpy()
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):

        idxs = np.random.choice(
            self.size, size=min(self.batch_size, self.size), replace=False
        )
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        self.weights = torch.tensor(
            np.zeros_like(idxs), dtype=torch.float32, device=DEVICE
        )
        return {
            k: torch.tensor(v, dtype=torch.float32, device=DEVICE)
            for k, v in batch.items()
        }
