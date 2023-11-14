"""Default Replay Buffer."""
import collections
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
        """Initialize simple replay buffer

        Args:
            obs_dim (int): Observation dimension
            act_dim (int): Action dimension
            size (int): Buffer size
            batch_size (int): Batch size
        """
        self.max_size = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=self.max_size)

    def __len__(self):
        return len(self.buffer)

    def store(self, values):
        # pdb.set_trace()

        def convert(arraylike):
            """Convert from tensor to nparray

            Args:
                arraylike (Torch.Tensor): Tensor to convert

            Returns:
                np.array: Converted numpyarray
            """
            obs = arraylike
            if isinstance(obs, torch.Tensor):
                if obs.requires_grad:
                    obs = obs.detach()
                obs = obs.cpu()
            return obs

        if type(values) is dict:
            # convert to deque
            obs = convert(values["obs"]).squeeze()
            next_obs = convert(values["next_obs"]).squeeze()
            action = torch.Tensor(values["act"].action) # .detach().cpu().numpy()
            reward = values["rew"]
            done = values["done"]
            currdict = {
                "obs": obs,
                "obs2": next_obs,
                "act": action,
                "rew": reward,
                "done": done,
            }
            self.buffer.append(currdict)

        elif type(values) == self.__class__:
            self.buffer.extend(values.buffer)
        else:
            print(type(values), self.__class__)
            raise Exception(
                "Sorry, invalid input type. Please input dict or buffer of same type"
            )

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self):
        """Sample batch from self.

        Returns:
            dict: Dictionary of batched information.
        """

        idxs = np.random.choice(
            len(self.buffer), size=min(self.batch_size, len(self.buffer)), replace=False
        )

        batch = dict()
        for idx in idxs:
            currdict = self.buffer[idx]
            for k, v in currdict.items():
                if isinstance(v, float):
                    v  = torch.Tensor([v])
                if isinstance(v, bool):
                    v  = torch.Tensor([v])
                if isinstance(v, int):
                    v = torch.Tensor([v])
                if k in batch:
                    batch[k].append(v)
                else:
                    batch[k] = [v]

        return  {
            k: torch.stack(v).to(DEVICE)
            for k, v in batch.items()
        }

    def finish_path(self, action_obj=None):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        pass
