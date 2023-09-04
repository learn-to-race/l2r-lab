import numpy as np
import tqdm
import torch
import cv2

from src.config.parser import read_config
from src.config.schema import cv_trainer_schema
import os
from src.config.yamlize import create_configurable, NameToSourcePath
import argparse
from src.constants import DEVICE
from src.loggers.WanDBLogger import WanDBLogger

if __name__ == "__main__":
    # TODO: data augmentation
    parser = argparse.ArgumentParser(description='CLSTM Trainer')
    # parser.add_argument('git_repository', type=str,
    #                     help='repository for git')
    # parser.add_argument('git_commit', type=str,
    #                     help='branch/commit for git')
    parser.add_argument('yaml_dir', type=str,
                        help='ex. ../config_files/train_clstm')
    parser.add_argument('--wandb_key', type=str, dest='wandb_key',
                        help='api key for weights and biases')                        
    args = parser.parse_args()
    
    training_config = read_config(
        f"{args.yaml_dir}/training.yaml", cv_trainer_schema
    )

    if not os.path.exists(training_config['model_save_path']):
        os.mkdir(training_config['model_save_path'])
    
    with open(
        f"{training_config['model_save_path']}/git_config",
        "w+",
    ) as f:
        f.write(str(args))

    if not os.path.exists(f"{training_config['model_save_path']}"):
        os.umask(0)
        os.makedirs(training_config["model_save_path"], mode=0o777, exist_ok=True)

    bsz = training_config["batch_size"]
    lr = training_config["lr"]
    encoder = create_configurable(
        f"{args.yaml_dir}/encoder.yaml", NameToSourcePath.encoder
    ).to(DEVICE)

    data_fetcher = create_configurable(
        f"{args.yaml_dir}/data_fetcher.yaml", NameToSourcePath.encoder_dataloader
    )
    optim = torch.optim.Adam(encoder.parameters(), lr=lr)
    num_epochs = training_config["num_epochs"]
    best_loss = 1e10

    train_ds, val_ds, train_dl, val_dl = data_fetcher.get_dataloaders(
        bsz,
        DEVICE,
    )
    if args.wandb_key is not None:
        logger = WanDBLogger(args.wandb_key, "test-project")
    # this is a stopgap - get_expert_demo_dataloaders has 1 value (autoencoding objective)
    # get_expert_demo_dataloaders has 1 value (autoencoding objective)
    # multiple_inputs = type(val_ds[0]) == tuple

    for epoch in range(num_epochs):
        train_loss = []
        encoder.train()
        for batch in tqdm.tqdm(train_dl, desc=f"Epoch #{epoch + 1} train"):
            # if multiple_inputs:
            #     # todo expand to more than 1, or 2 things passed by dataloader?
            #     x = batch[:-1]
            #     y = batch[-1]
            #     loss = encoder.loss(y, encoder(x))
            # else:
            #     loss = encoder.loss(batch, encoder(batch))
            x = batch[:,:-1]
            if x.shape[0] != bsz:
                continue
            y = batch[:,-1:]
            loss = encoder.loss(y, encoder(x))
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            if args.wandb_key is not None:
                logger.log({"train_loss": loss.item()})
        train_loss = np.mean(train_loss)
        test_loss = []
        encoder.eval()
        for batch in tqdm.tqdm(val_dl, desc=f"Epoch #{epoch + 1} test"):
            # vae had a kld_weight=0.0 here... but yeah not sure how to parameterize that
            x = batch[:,:-1]
            if x.shape[0] != bsz:
                continue
            y = batch[:,-1:]
            loss = encoder.loss(y, encoder(x))
            test_loss.append(loss.item())
            if args.wandb_key is not None:
                logger.log({"test_loss": loss.item()})
        test_loss = np.mean(test_loss)
        print(f"#{epoch + 1} train_loss: {train_loss:.6f}, test_loss: {test_loss:.6f}")
        # TODO: ADD WANDB LOGGING
        if test_loss < best_loss:
            best_loss = test_loss
            print(f"save model at epoch #{epoch + 1}")
            torch.save(
                encoder.state_dict(), f"{training_config['model_save_path']}/best.pth"
            )