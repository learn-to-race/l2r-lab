"""Simple Random Agent."""
import json
import time
import numpy as np
from src.agents.base import BaseAgent
from src.utils.utils import ActionSample
from src.config.yamlize import yamlize, create_configurable, NameToSourcePath


@yamlize
class randomAgent(BaseAgent):
    """Randomly pick actions in the space."""
    
    def __init__(
        self,
        steps_to_sample_randomly: int,
        gamma: float,
        alpha: float,
        polyak: float,
        lr: float,
        actor_critic_cfg_path: str,
        load_checkpoint_from: str = "",
    ):
        super(randomAgent, self).__init__()

    def select_action(self, obs) -> np.array:
        ac = ActionSample()
        myaction = self.action_space.sample()
        ac.action = myaction

        return ac

    def register_reset(self, obs) -> np.array:
        pass

    def update(self, data):
        pass

    def load_model(self, path):
        pass
    
    def save_model(self, path):
        pass
