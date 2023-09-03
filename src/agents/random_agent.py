import time
import numpy as np
from src.agents.base import BaseAgent
from src.config.yamlize import yamlize, create_configurable, NameToSourcePath
from src.utils.utils import ActionSample

@yamlize
class RandomAgent(BaseAgent):
    """Randomly pick actions in the space."""
    def __init__(
        self,
        steps_to_sample_randomly: int,
    ):
        super(RandomAgent, self).__init__()

    def select_action(self, obs) -> np.array:
        """Selection action through random sampling.
	@@ -17,4 +24,50 @@ def select_action(self, obs) -> np.array:
        Returns:
            np.array: Action
        """
        obj = ActionSample()
        a = self.action_space.sample()
        obj.action = a
        return obj
    def register_reset(self, obs) -> np.array:  # pragma: no cover
        """Handle reset of episode.
        Args:
            obs (np.array): Observation
        Returns:
            np.array: Action
        """
        return self.select_action(obs)

    def update(self, data):  # pragma: no cover
        """Model update given data
        Args:
            data (dict): Data.
        Raises:
            NotImplementedError: Need to overload
        """
        pass

    def load_model(self, path):  # pragma: no cover
        """Load model checkpoint from path
        Args:
            path (str): Path to checkpoint
        Raises:
            NotImplementedError: Need to overload
        """
        pass

    def save_model(self, path):  # pragma: no cover
        """Save model checkpoint to path
        Args:
            path (str): Path to checkpoint
        Raises:
            NotImplementedError: Need to overload.
        """
        pass