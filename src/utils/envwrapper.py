"""Container for the pip-installable L2R environment. As L2R has some slight differences compared to what we expect, this allows us to fit the pieces together."""
import numpy as np
import torch
import itertools
from src.constants import DEVICE
import wandb


class EnvContainer:
    """Container for the pip-installed L2R Environment."""

    def __init__(self, encoder=None):
        """Initialize container around encoder object

        Args:
            encoder (nn.Module, optional): Encoder object to encoder inputs. Defaults to None.
        """
        self.encoder = encoder
        self.image_list = []

    def _process_obs(self, obs: dict):
        """Process observation using encoder

        Args:
            obs (dict): Observation as a dict.

        Returns:
            torch.Tensor: encoded image.
        """
        obs_camera = obs["images"]["CameraFrontRGB"]
        obs2 = np.transpose(obs_camera,(2,0,1))
        self.image_list.append(obs2)
        obs_encoded = self.encoder.encode(obs_camera).to(DEVICE)
        speed = (
            torch.tensor(np.linalg.norm(obs["pose"][3:6], ord=2))
            .to(DEVICE)
            .reshape((-1, 1))
            .float()
        )/100.0
        return torch.cat((obs_encoded, speed), 1).to(DEVICE)

    def step(self, action, env=None):
        """Step env.

        Args:
            action (np.array): Action to apply
            env (gym.env, optional): Environment to step upon. Defaults to None.

        Returns:
            tuple: Tuple of next_obs, reward, done, info
        """
        if env:
            self.env = env
        obs, reward, done, info = self.env.step(action)
        reward = min(reward / 150.0, 1.0)
        return self._process_obs(obs), reward, done, info

    def reset(self, random_pos=False, env=None):
        """Reset env.

        Args:
            random_pos (bool, optional): Whether to reset to a random position ( might not exist in current iteration ). Defaults to False.
            env (gym.env, optional): Environment to step upon. Defaults to None.

        Returns:
            next_obs: Encoded next observation.
        """
        if len(self.image_list) > 0:
            wandb.log({"Episode Video": wandb.Video(np.stack(self.image_list), fps=8, format='gif')})
            self.image_list = []
        if env:
            self.env = env
        obs = self.env.reset(random_pos=random_pos)
        return self._process_obs(obs)
