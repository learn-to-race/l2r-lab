"""Logger Base Class."""
from abc import ABC, abstractmethod
import os


class BaseLogger(ABC):
    """Base Logger."""

    def __init__(self, log_dir, exp_name) -> None:
        """Initialize logger

        Args:
            log_dir (str): Log directory
            exp_name (str): Experiment Name
        """
        super().__init__()
        self.exp_name = exp_name
        self.log_dir = log_dir
        if not os.path.exists(f"{self.log_dir}/{exp_name}/runlogs"):
            os.umask(0)
            os.makedirs(self.log_dir, mode=0o777, exist_ok=True)
            os.makedirs(f"{self.log_dir}/{exp_name}/runlogs", mode=0o777, exist_ok=True)
            os.makedirs(f"{self.log_dir}/{exp_name}/tblogs", mode=0o777, exist_ok=True)
        pass

    def log(self, logging_data):
        pass

    def log_env_train(self, metric_data):
        pass

    def log_env_val(self, metric_data):
        pass
