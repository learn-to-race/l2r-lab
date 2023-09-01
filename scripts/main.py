from l2r import build_env

# from l2r import RacingEnv
from src.config.yamlize import NameToSourcePath, create_configurable
import sys
import logging


if __name__ == "__main__":
    print(sys.argv)
    # Build environment
    env = build_env(
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
    runner = create_configurable(
        "config_files/canny/runner.yaml", NameToSourcePath.runner
    )

    with open(
        f"{runner.model_save_dir}/{runner.experiment_name}/git_config",
        "w+",
    ) as f:
        f.write(" ".join(sys.argv[1:3]))
    # Race!
    try:
        import torch

        torch.autograd.set_detect_anomaly(True)
        runner.run(env, sys.argv[3])
    except IndexError as e:
        logging.warning(e)
        runner.run(env, "")
