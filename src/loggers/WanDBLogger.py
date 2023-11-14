"""Weights and Biases Logging."""
from src.loggers.base import BaseLogger
from src.config.yamlize import yamlize
import logging, re, sys
import wandb
from datetime import datetime


class WanDBLogger(BaseLogger):
    """Wandb Logger Wrapper."""

    def __init__(self, api_key: str, project_name: str, exp_name: str) -> None:
        """Create Weights and Biases Logger

        Args:
            api_key (str): api key (DO NOT STORE IN REPO)
            project_name (str): project name
            exp_name (str): the experiment name (name of the run)
        """
        
        wandb.login(key=api_key)
        wandb.init(project=project_name, name=exp_name)

    def log(self, data):
        print(data)
        wandb.log({"reward": data})
        #wandb.log({"Distance": data[1]})
        #wandb.log({"Time": data[2]})
        #wandb.log({"num_infractions": data[3]})
        #wandb.log({"average_speed_kph": data[4]})
        #wandb.log({"average_displacement_error": data[5]})
        #wandb.log({"trajectory_efficiency": data[6]})
        #wandb.log({"trajectory_admissibility": data[7]})
        #wandb.log({"movement_smoothness": data[8]})
        #wandb.log({"timestep/sec": data[9]})
        #wandb.log({"laps_completed": data[10]})

    def eval_log(self, data):
        wandb.log({"Eval reward": data})
        #wandb.log({"Eval Distance": data[1]})
        #wandb.log({"Eval Time": data[2]})
        #wandb.log({"Eval Num_infractions": data[3]})
        #wandb.log({"Eval Average_speed_kph": data[4]})
        #wandb.log({"Eval Average_displacement_error": data[5]})
        #wandb.log({"Eval Trajectory_efficiency": data[6]})
        #wandb.log({"Eval Trajectory_admissibility": data[7]})
        #wandb.log({"Eval Movement_smoothness": data[8]})
        #wandb.log({"Eval Timestep/sec": data[9]})
        #wandb.log({"Eval Laps_completed": data[10]})

    def log_metric(self, data, name):
        wandb.log({name: data})
