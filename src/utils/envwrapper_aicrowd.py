import numpy as np
import torch
import itertools
from src.constants import DEVICE


class EnvContainer:
    """Container for L2R Environment."""

    def __init__(self, encoder=None, env=None):
        self.encoder = encoder
        self.env = env

    def _process_obs(self, obs_camera, obs_state):
        if type(obs_camera) == list and len(obs_camera) == 1:
            obs_camera = np.array(obs_camera[0])
            obs_camera = np.reshape(obs_camera, (384, 512, 3))
        obs_encoded = self.encoder.encode(obs_camera).to(DEVICE)
        speed = (
            torch.tensor(np.linalg.norm(obs_state[3:6], ord=2))
            .to(DEVICE)
            .reshape((-1, 1))
            .float()
        )
        return torch.cat((obs_encoded, speed), 1).to(DEVICE)

    def step(self, action, env=None):
        if env:
            self.env = env
        (state, camera), reward, done, info = self.env.step(action)
        return self._process_obs(camera, state), reward, done, info

    def reset(self, random_pos=False, env=None):
        if env:
            self.env = env
        obs = self.env.reset(random_pos=random_pos)
        (state, camera), _ = obs
        return self._process_obs(camera, state)

    def reset_episode(self, t):
        camera, feat, state, _, _ = self.reset(random_pos=True)
        ep_ret, ep_len, experience = 0, 0, []
        t_start = t + 1
        camera, feat, _, _, _, _ = self.step([0, 1])
        return camera, ep_len, ep_ret, experience, feat, state, t_start
