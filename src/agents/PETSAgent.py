import itertools
from multiprocessing.sharedctypes import Value
import queue, threading
from copy import deepcopy

import torch
import numpy as np
from gym.spaces import Box
from torch.optim import Adam

from src.agents.base import BaseAgent
from src.config.yamlize import (
    create_configurable,
    yamlize,
    create_configurable_from_dict,
    ConfigurableDict,
    NameToSourcePath,
)
from src.utils.utils import ActionSample

from src.constants import DEVICE

import torch.nn as nn
import torch.nn.functional as F

from src.constants import DEVICE


@yamlize
class PETSAgent(BaseAgent):
    """Adopted from https://github.com/BY571/PETS-MPC. Currently not using CEM, but random AS."""

    def __init__(
        self,
        network_config_path: str,
        planner_config: ConfigurableDict,
        n_ensembles: int = 7,
        lr: float = 1e-2,
        model_save_path: str = "/mnt/blah",
        load_checkpoint: bool = False,
        deterministic: bool = False,
    ):
        """Initialize PETS Agent

        Args:
            network_config_path (str): Path to network config.
            planner_config (ConfigurableDict): Configuration of planner
            n_ensembles (int, optional): Number of networks in ensemble. Defaults to 7.
            lr (float, optional): Learning rate. Defaults to 1e-2.
            model_save_path (str, optional): Model save path (unused). Defaults to '/mnt/blah'.
            load_checkpoint (bool, optional): Whether to load checkpoint or not (unused). Defaults to False.
            deterministic (bool, optional): Whether to act deterministically (unused). Defaults to False.
        """
        super().__init__()
        self.model = create_configurable(network_config_path, NameToSourcePath.network)
        self.model.to(DEVICE)
        self.model_save_path = model_save_path
        self.load_checkpoint = load_checkpoint
        self.deterministic = deterministic
        self.n_ensembles = n_ensembles
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr
        )
        self.planner = create_configurable_from_dict(
            planner_config, NameToSourcePath.planner
        )

    def select_action(self, obs) -> np.array:
        """Select action given obs

        Args:
            obs (np.array): Observation

        Returns:
            np.array: Action
        """
        action_obj = ActionSample()
        state = obs.detach().float()
        action_obj.action = self.planner.get_action(state, self.model)
        return action_obj

    def register_reset(self, obs) -> np.array:
        """Handle episode reset

        Args:
            obs (np.array): Observation

        Returns:
            np.array: Action
        """
        pass

    def update(self, data):
        """Update given data

        Args:
            data (dict): Dict of data from SimpleReplayBuffer
        """
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        o2 = torch.tensor(torch.concatenate((o2 - o, r.reshape((-1, 1))), axis=-1))
        o2 = o2.reshape((1, *o2.shape))
        o = torch.tensor(torch.concatenate((o, a), axis=-1))
        o = o.reshape((1, *o.shape)).repeat((self.n_ensembles, 1, 1))
        mu, log_var = self.model(o)
        assert mu.shape[1:] == o2.shape[1:]
        # Remove validate atm.
        inv_var = (-log_var).exp()
        loss = ((mu - o2) ** 2 * inv_var).mean(-1).mean(-1).sum() + log_var.mean(
            -1
        ).mean(-1).sum()
        loss += 0.01 * torch.sum(self.model.max_logvar) - 0.01 * torch.sum(
            self.model.min_logvar
        )
        loss.backward()
        self.optimizer.step()

    def load_model(self, path):
        """Unused but load model from data

        Args:
            path (str): Path to data

        Returns:
            model: self
        """
        return self

    def save_model(self, path):
        """Unusued but save model to path

        Args:
            path (str): _description_
        """
        pass


# TODO: Make ensemble_size and num_ensembles agree.
