import numpy as np
import torch
import itertools
from src.constants import DEVICE
import gym


class EnvContainer(gym.Env):
    """Container for L2R Environment."""

    def __init__(self, encoder=None, env=None):
        self.encoder = encoder
        self.env = env

    def _process_obs(self, obs: dict):
        """Process observation using encoder

        Args:
            obs (dict): Observation as a dict.

        Returns:
            torch.Tensor: encoded image.
        """
        obs_camera = obs["images"]["CameraFrontRGB"]
        obs_encoded = self.encoder.encode(obs_camera).to(DEVICE)
        speed = (
            torch.tensor(np.linalg.norm(obs["pose"][3:6], ord=2))
            .to(DEVICE)
            .reshape((-1, 1))
            .float()
        )
        return torch.cat((obs_encoded, speed), 1).to(DEVICE)

    def step(self, action, env=None):
        """Step env.

        Args:
            action (np.array): Action to apply
            env (gym.env, optional): Environment to step upon. Defaults to None.

        Returns:
            tuple: Tuple of next_obs, reward, done, info
        """
        action = action.reshape((2,))
        if env:
            self.env = env
        obs, reward, terminated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, info

    def reset(self, random_pos=False, env=None):
        """Reset env.

        Args:
            random_pos (bool, optional): Whether to reset to a random position ( might not exist in current iteration ). Defaults to False.
            env (gym.env, optional): Environment to step upon. Defaults to None.

        Returns:
            next_obs: Encoded next observation.
        """
        if env:
            self.env = env
        obs = self.env.reset(random_pos=random_pos)
        return self._process_obs(obs)

    def __getattr__(self, name):
        try:
            import logging

            return getattr(self.env, name)
        except Exception as e:
            raise e
