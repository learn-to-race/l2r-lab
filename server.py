from src.config.yamlize import NameToSourcePath, create_configurable
import threading
import sys
import argparse

from distrib_l2r.async_learner import AsyncLearningNode

if __name__ == "__main__":
    # Argparse for environment + training paradigm selection and wandb config
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        choices=["l2r", "mcar", "walker"],
        help="Select the environment ('l2r', 'mcar', or 'walker')."
    )

    parser.add_argument(
        "--paradigm",
        choices=["dCollect", "dUpdate"],
        help="Select the distributed training paradigm ('dCollect', 'dUpdate')."
    )

    parser.add_argument(
        "--wandb_apikey",
        type=str,
        help="Enter your Weights-And-Bias API Key."
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        help="Enter your experiment name, to be recorded by Weights-And-Bias."
    )

    args = parser.parse_args()
    print(f"Server Configured - '{args.env}'")
    print(f"Training Paradigm Configured - '{args.paradigm}'")

    # NOTE: walker -> https://www.gymlibrary.dev/environments/box2d/bipedal_walker/
    # NOTE: mcar -> https://mgoulao.github.io/gym-docs/environments/classic_control/mountain_car_continuous/
                
    learner = AsyncLearningNode(
        agent=create_configurable(
            f"config_files/async_sac_{args.env}/agent.yaml", NameToSourcePath.agent
        ),
        api_key=args.wandb_apikey,
        exp_name=args.exp_name,
        env_name=args.env,
        paradigm=args.paradigm
    )

    server_thread = threading.Thread(target=learner.serve_forever)
    server_thread.start()
    learner.learn()
