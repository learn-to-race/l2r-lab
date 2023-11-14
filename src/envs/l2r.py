"""Container for the pip-installable L2R environment. As L2R has some slight differences compared to what we expect, this allows us to fit the pieces together."""
import numpy as np
import torch
import itertools
from src.constants import DEVICE
from src.config.yamlize import create_configurable, yamlize, NameToSourcePath
from l2r import build_env


@yamlize
class L2RSingleCamera:
    """Container for the pip-installed L2R Environment."""

    def __init__(self, encoder_config_path: str):
        """Initialize env around encoder [TODO: Make configurations as params to this class.]

        Args:
            encoder (nn.Module, optional): Encoder object to encoder inputs. Defaults to None.
        """
        self.encoder = create_configurable(
            encoder_config_path, NameToSourcePath.encoder)
        self.encoder.to(DEVICE)

        self.env = build_env(
            controller_kwargs={"quiet": True},
            camera_cfg=[
                {
                    "name": "CameraFrontRGB",
                            "Addr": "tcp://0.0.0.0:8008",
                            "Width": 512,
                            "Height": 384,
                            "sim_addr": "tcp://0.0.0.0:8008",
                }
            ],
            env_kwargs={
                "multimodal": True,
                "eval_mode": True,
                "n_eval_laps": 5,
                "max_timesteps": 5000,
                "obs_delay": 0.1,
                "not_moving_timeout": 50000,
                "reward_pol": "custom",
                "provide_waypoints": False,
                "active_sensors": ["CameraFrontRGB"],
                "vehicle_params": False,
            },
            action_cfg={
                "ip": "0.0.0.0",
                "port": 7077,
                "max_steer": 0.3,
                "min_steer": -0.3,
                "max_accel": 6,
                "min_accel": -1,
            },
        )

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

    def step(self, action):
        """Step env.

        Args:
            action (np.array): Action to apply

        Returns:
            tuple: Tuple of next_obs, reward, done, info
        """
        obs, reward, done, info = self.env.step(action)
        return self._process_obs(obs), reward, done, info

    def reset(self, options=None):
        """Reset env.

        Args:
            random_pos (bool, optional): Whether to reset to a random position ( might not exist in current iteration ). Defaults to False.

        Returns:
            next_obs: Encoded next observation.
        """
        obs = self.env.reset(
            random_pos=False if options is None else options.get("random_pos", False))
        return self._process_obs(obs)
