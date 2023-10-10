"""PPOAgent Definition. """
import itertools
from copy import deepcopy

import torch
import numpy as np
from gym.spaces import Box, Discrete
from torch.optim import Adam

from src.agents.base import BaseAgent
from src.config.yamlize import yamlize, create_configurable, NameToSourcePath
from src.utils.utils import ActionSample

from src.constants import DEVICE


@yamlize
class IQNAgent(BaseAgent):
    """Implicit Quantile Network Agent"""

    def __init__(
        self,
        steps_to_sample_randomly: int,
        lr: float,
        network_cfg_path: str = "",
        eps: float = 0.1
    ):
        """
        
        """
        super(IQNAgent, self).__init__()
        self.steps_to_sample_randomly = steps_to_sample_randomly
        self.lr = lr

        self.t = 0
        self.deterministic = False 

        self.record = {"transition_actor": ""}

        self.action_space = Discrete(9)

        self.mapper = { # TODO: PARAMETRIZE
            0: np.array([0.0, 0.0]),
            1: np.array([0.0, 1.0]),
            2: np.array([0.0, -1.0]),
            3: np.array([1.0, 0.0]),
            4: np.array([1.0, 1.0]),
            5: np.array([1.0, -1.0]),
            6: np.array([-1.0, 0.0]),
            7: np.array([-1.0, 1.0]),
            8: np.array([-1.0, -1.0]),
        }

        self.act_dim = self.action_space.n

        self.iqn_local = create_configurable( network_cfg_path, NameToSourcePath.network).to(DEVICE)
        self.iqn_target = create_configurable( network_cfg_path, NameToSourcePath.network).to(DEVICE)
        self.optimizer = Adam(self.iqn_local.parameters(), lr=self.lr)

        self.eps = eps

    def select_action(self, obs) -> np.array:
        """Select action given observation array.

        Args:
            obs (np.array): Observation array

        Returns:
            np.array: Action array
        """
        action_obj = ActionSample()
        if self.t > self.steps_to_sample_randomly:
            self.iqn_local.eval()
            with torch.no_grad():
                print(type(obs),obs)
                obs = torch.Tensor(obs).to(DEVICE)
                action, _ = self.iqn_local(obs)
                action = action.mean(dim=1)
                action = action.cpu().numpy()
                if self.deterministic or np.random.uniform(0,1) >= self.eps:
                    action_obj.action = np.argmax(action)
                else:
                    action_obj.action = self.action_space.sample()

        else:
            action_obj.action = self.action_space.sample()
        
        action_obj.action = self.mapper[int(action_obj.action)]

        self.t = self.t + 1
        return action_obj

    def register_reset(self, obs):
        """Handle reset of episode."""
        pass


    def update(self, data):
        """Update parameters given batch of data.

        Args:
            data (dict): Dict of batched data to update params from.
        """
        self.optimizer.zero_grad()
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        Q_targets_next, _ = self.qnetwork_target(o2)
        Q_targets_next = Q_targets_next.detach().max(2)[0].unsqueeze(1) # (batch_size, 1, N)
        
        Q_targets = r.unsqueeze(-1) + (self.GAMMA**self.n_step * Q_targets_next * (1. - d.unsqueeze(-1)))
        # Get expected Q values from local model
        Q_expected, taus = self.qnetwork_local(o)
        Q_expected = Q_expected.gather(2, a.unsqueeze(-1).expand(self.BATCH_SIZE, 8, 1))

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, 8, 8), "wrong td error shape"

        k = 1.0
        huber_l = torch.where(td_error.abs() <= k, 0.5 * td_error.pow(2), k * (td_error.abs() - 0.5 * k))
        assert huber_l.shape == (td_error.shape[0], 8, 8), "huber loss has wrong shape"
        
        quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
        
        loss = quantil_l.sum(dim=1).mean(dim=1) # , keepdim=True if per weights get multipl
        loss = loss.mean()


        # Minimize the loss
        loss.backward()
        #clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        print("IQN LOSS:",loss.item())
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)


    def _soft_update(self):
        """Exponential average for double q network."""
        for target_param, local_param in zip(self.iqn_target.parameters(), self.iqn_local.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)


    def load_model(self, path):
        """Load model from path

        Args:
            path (str): Load path using str
        """
        iqn_local_dict, iqn_target_dict = torch.load(path)
        self.iqn_local.load_state_dict(iqn_local_dict)
        self.iqn_target.load_state_dict(iqn_target_dict)

    def save_model(self, path):
        """Save model to path. Untested.

        Args:
            path (str): Save path using str
        """
        torch.save((self.iqn_local, self.iqn_target), path)
