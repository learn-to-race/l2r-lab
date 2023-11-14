from src.config.yamlize import NameToSourcePath, create_configurable
import torch
import argparse

if __name__ == "__main__":

    # Argparse for environment selection and wandb config
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        choices=["l2r", "mcar", "walker"],
        help="Select the environment ('l2r', 'mcar', or 'walker')."
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

    # Initialize the runner and start run
    print(f"Environment: {args.env} | Experiment: {args.exp_name}")

    if args.env == "l2r":
        runner = create_configurable(
            "config_files/l2r_sac/runner.yaml", NameToSourcePath.runner)
    elif args.env == "mcar":
        runner = create_configurable(
            "config_files/mcar_sac/runner.yaml", NameToSourcePath.runner)
    elif args.env == "walker":
        runner = create_configurable(
            "config_files/walker_sac/runner.yaml", NameToSourcePath.runner)
    else:
        raise NotImplementedError

    torch.autograd.set_detect_anomaly(True)
    runner.run(args.wandb_apikey, args.exp_name)
