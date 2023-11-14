"""Store schema from older config system. DO NOT USE."""
from strictyaml import (
    load,
    Map,
    Str,
    Int,
    Seq,
    Enum,
    Float,
    Optional,
    Bool,
    CommaSeparated,
)

# Config YAML Schema
experiment_schema = Map(
    {
        "experiment_config": Map(
            {
                "experiment_name": Str(),
                "inference_only": Bool(),
                "load_checkpoint": Bool(),
                "record_experience": Bool(),
                "update_after": Int(),
                "update_every": Int(),
                "experiment_state_path": Str(),
            }
        )
    }
)

encoder_schema = Map(
    {
        "encoder_config": Map(
            {
                "encoder_type": Enum(["vae"]),
                Optional(
                    "load_checkpoint_from", default=False, drop_if_none=True
                ): Str()
                | Bool(),
                "latent_dims": Int(),
                "hiddens": Seq(Int()),
                "speed_hiddens": Seq(Int()),
                "actor_hiddens": Seq(Int()),
                "image_channels": Int(),
                "image_width": Int(),
                "image_height": Int(),
                "ac_input_dims": Int(),
                "include_speed": Bool(),
            }
        )
    }
)

agent_schema = Map(
    {
        "agent_config": Map(
            {
                "make_random_actions": Bool(),
                "steps_to_sample_randomly": Int(),
                "gamma": Float(),
                "polyak": Float(),
                "lr": Float(),
                "alpha": Float(),
                "checkpoint": Str(),
                "load_checkpoint": Bool(),
                "model_save_path": Str(),
                "experiment_name": Str(),
                "record_dir": Str(),
                "track_name": Str(),
            }
        )
    }
)

runner_schema = Map(
    {
        "runner_config": Map(
            {
                "model_save_dir": Str(),
                "experience_save_dir": Str(),
                "num_test_episodes": Int(),
                "num_run_episodes": Int(),
                "save_every_nth_episode": Int(),
                "total_environment_steps": Int(),
                "update_model_after": Int(),
                "update_model_every": Int(),
                "eval_every": Int(),
                "max_episode_length": Int(),
            }
        )
    }
)

env_schema = Map(
    {
        "env_config": Map(
            {"track_name": Str(), "runtime": Enum(["local"]), "dirhash": Str()}
        )
    }
)

replay_buffer_schema = Map(
    {"replay_buffer_config": Map({"replay_size": Int(), "batch_size": Int()})}
)


cv_trainer_schema = Map(
    {
        "cv_trainer_config": Map(
            {
                "batch_size": Int(),
                "num_epochs": Int(),
                "lr": Float(),
                "model_save_path": Str(),
                "train_data_path": Str(),
                "val_data_path": Str(),
            }
        )
    }
)
