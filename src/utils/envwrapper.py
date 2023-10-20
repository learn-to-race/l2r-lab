"""Container for the pip-installable L2R environment. As L2R has some slight differences compared to what we expect, this allows us to fit the pieces together."""
import numpy as np
import torch
import itertools
from src.constants import DEVICE
import os
import json


class EnvContainer:
    """Container for the pip-installed L2R Environment."""

    def __init__(self, encoder=None,collect_data=False):
        """Initialize container around encoder object

        Args:
            encoder (nn.Module, optional): Encoder object to encoder inputs. Defaults to None.
        """
        self.encoder = encoder
        if collect_data:
            self.counter = 1
            self.arrcounter = 1
            self.k = 5
            self.buffer = []
            self.infobuffer = []
        self.collect_data = collect_data

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
        if env:
            self.env = env
        obs, reward, done, info = self.env.step(action)
        if self.collect_data:
            self.counter += 1
            image = obs["images"]["CameraFrontRGB"]
            self.buffer.append(image)
            self.infobuffer.append(info)
            if done:                
                nparr = np.stack(self.buffer,axis=0)
                os.makedirs(f"/mnt/data/collected_data_{self.encoder.__class__.__name__}/",exist_ok=True)
                np.save(f"/mnt/data/collected_data_{self.encoder.__class__.__name__}/episode_{self.arrcounter}.npy",nparr)
                with open(f"/mnt/data/collected_data_{self.encoder.__class__.__name__}/episode_{self.arrcounter}_info.json", "w") as json_file:
                    json.dump(self.infobuffer, json_file)
                print("---------------------------------")
                print("Saved episode",self.arrcounter)
                print("---------------------------------")
                self.arrcounter += 1
                del nparr
                self.buffer = []
                self.infobuffer = []
                self.counter = 1
        return self._process_obs(obs), reward, done, info

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
