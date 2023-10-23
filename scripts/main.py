from l2r import build_env

# from l2r import RacingEnv
from src.config.yamlize import NameToSourcePath, create_configurable
import sys
import logging


if __name__ == "__main__":
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
        "config_files/example_iqn/runner.yaml", NameToSourcePath.runner
    )

    with open(
        f"{runner.model_save_dir}/{runner.experiment_name}/git_config",
        "w+",
    ) as f:
        f.write(" ".join(sys.argv[1:3]))
    # Race!
    import warnings
    import dreamerv3
    from dreamerv3 import embodied
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['medium'])
    config = config.update({
        'logdir': '~/logdir/run1',
        'run.train_ratio': 64,
        'run.log_every': 30,  # Seconds
        'batch_size': 16,
        'jax.prealloc': False,
        'encoder.mlp_keys': '$^',
        'decoder.mlp_keys': '$^',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
        # 'jax.platform': 'cpu',
    })
    config = embodied.Flags(config).parse()

    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        #embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        #embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandBOutput(logdir.name, config),
        # embodied.logger.MLFlowOutput(logdir.name),
    ])

    import crafter
    from embodied.envs import from_gym
    env = crafter.Env()  # Replace this with your Gym env.
    env = from_gym.FromGym(env, obs_key='image')  # Or obs_key='vector'.
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / 'replay')
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)
    embodied.run.train(agent, env, replay, logger, args)
    import torch

    torch.autograd.set_detect_anomaly(True)
    runner.run(env, sys.argv[3])