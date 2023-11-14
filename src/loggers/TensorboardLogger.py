"""Wrapper around TensorBoard for logging. Unused for WandB, but should work."""
from datetime import datetime
import os
from src.loggers.base import BaseLogger
from src.config.yamlize import yamlize
from tensorboardX import SummaryWriter


class TensorboardLogger(BaseLogger):
    """TBLogger Instance"""

    def __init__(self, log_dir: str, experiment_name: str) -> None:
        """TensorBoard Logger

        Args:
            log_dir (str): Log directory
            experiment_name (str): Experiment Name
        """

        super().__init__(log_dir, experiment_name)
        current_time = datetime.now().strftime("%m%d%H%M%S")
        self.exp_name = experiment_name
        self.tb_log_dir = (
            f"{log_dir}/{experiment_name}/tblogs/{experiment_name}_{current_time}"
        )
        self.tb_logger = SummaryWriter(log_dir=self.tb_log_dir)

    def log_train_metrics(self, ep_ret, t, t_start, episode_num, metadata):
        self.tb_logger.add_scalar("train/episodic_return", ep_ret, episode_num)
        self.tb_logger.add_scalar(
            "train/ep_total_time",
            metadata["info"]["metrics"]["total_time"],
            episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_total_distance",
            metadata["info"]["metrics"]["total_distance"],
            episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_avg_speed",
            metadata["info"]["metrics"]["average_speed_kph"],
            episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_avg_disp_err",
            metadata["info"]["metrics"]["average_displacement_error"],
            episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_traj_efficiency",
            metadata["info"]["metrics"]["trajectory_efficiency"],
            episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_traj_admissibility",
            metadata["info"]["metrics"]["trajectory_admissibility"],
            episode_num,
        )
        self.tb_logger.add_scalar(
            "train/movement_smoothness",
            metadata["info"]["metrics"]["movement_smoothness"],
            episode_num,
        )
        self.tb_logger.add_scalar("train/ep_n_steps", t - t_start, episode_num)

        ## Sid needs to change a bunch of stuff here to fix the move

    def log_val_metrics(self, info, ep_ret, n_eps, n_val_steps, metadata):
        self.tb_logger.add_scalar("val/episodic_return", ep_ret, n_eps)
        self.tb_logger.add_scalar("val/ep_n_steps", n_val_steps, n_eps)

        try:
            self.tb_logger.add_scalar(
                "val/ep_pct_complete", info["metrics"]["pct_complete"], n_eps
            )
            self.tb_logger.add_scalar(
                "val/ep_total_time", info["metrics"]["total_time"], n_eps
            )
            self.tb_logger.add_scalar(
                "val/ep_total_distance", info["metrics"]["total_distance"], n_eps
            )
            self.tb_logger.add_scalar(
                "val/ep_avg_speed", info["metrics"]["average_speed_kph"], n_eps
            )
            self.tb_logger.add_scalar(
                "val/ep_avg_disp_err",
                info["metrics"]["average_displacement_error"],
                n_eps,
            )
            self.tb_logger.add_scalar(
                "val/ep_traj_efficiency",
                info["metrics"]["trajectory_efficiency"],
                n_eps,
            )
            self.tb_logger.add_scalar(
                "val/ep_traj_admissibility",
                info["metrics"]["trajectory_admissibility"],
                n_eps,
            )
            self.tb_logger.add_scalar(
                "val/movement_smoothness",
                info["metrics"]["movement_smoothness"],
                n_eps,
            )
        except:
            pass

        # TODO: Find a better way: requires knowledge of child class API :(
        if "safety_info" in metadata:
            self.tb_logger.add_scalar(
                "val/ep_interventions",
                metadata["safety_info"]["ep_interventions"],
                n_eps,
            )
