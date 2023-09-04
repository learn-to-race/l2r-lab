"""Weights and Biases Logging."""
from src.loggers.base import BaseLogger
from src.config.yamlize import yamlize
import logging, re, sys
import wandb
from datetime import datetime


class WanDBLogger(BaseLogger):
    """Wandb Logger Wrapper."""

    def __init__(self, api_key: str, project_name: str) -> None:
        """Create Weights and Biases Logger

        Args:
            api_key (str): api key (DO NOT STORE IN REPO)
            project_name (str): project name
        """
        # super().__init__(log_dir, experiment_name)
        wandb.login(key=api_key)
        # wandb.init(project=project_name)
        wandb.init(project = "canny")

    def log(self, data):
        """Log metrics to WandB, using names present in dict.

        Args:
            data (dict): Dict to log
        """
        wandb.log(data)
