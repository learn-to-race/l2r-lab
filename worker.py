import socket
import os
import argparse

from distrib_l2r.async_worker import AsnycWorker

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
    
    args = parser.parse_args()
    print(f"Worker Configured - '{args.env}'")
    print(f"Training Paradigm Configured - '{args.paradigm}'")

    # Configure learner IP (by environment)
    learner_ip = socket.gethostbyname(f"{args.env}-{args.paradigm.lower()}-learner")
    learner_address = (learner_ip, 4444)
    
    # Configure worker (by training paradigm)
    worker = AsnycWorker(learner_address=learner_address, env_name=args.env, paradigm=args.paradigm)

    worker.work()
