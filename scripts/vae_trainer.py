import numpy as np
import tqdm
import torch
import cv2

from src.encoders.vae import VAE
from src.encoders.dataloaders.expert_demo_dataloader import get_expert_demo_dataloaders
from src.config.parser import read_config
from src.config.schema import cv_trainer_schema
import os
from src.config.yamlize import create_configurable, NameToSourcePath
import sys

if __name__ == "__main__":
    # TODO: data augmentation

    training_config = read_config(
        "config_files/train_vae/training.yaml", cv_trainer_schema
    )

    with open(
        f"{training_config['model_save_path']}/git_config",
        "w+",
    ) as f:
        f.write(" ".join(sys.argv[1:]))

    if not os.path.exists(f"{training_config['model_save_path']}"):
        os.umask(0)
        os.makedirs(training_config["model_save_path"], mode=0o777, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bsz = training_config["batch_size"]
    lr = training_config["lr"]
    vae = create_configurable(
        "config_files/train_vae/encoder.yaml", NameToSourcePath.encoder
    ).to(device)
    optim = torch.optim.Adam(vae.parameters(), lr=lr)
    num_epochs = training_config["num_epochs"]
    best_loss = 1e10

    train_ds, val_ds, train_dl, val_dl = get_expert_demo_dataloaders(
        training_config["train_data_path"],
        training_config["val_data_path"],
        bsz,
        device,
    )

    for epoch in range(num_epochs):
        train_loss = []
        vae.train()
        for batch in tqdm.tqdm(train_dl, desc=f"Epoch #{epoch + 1} train"):
            loss = vae.loss(batch, *vae(batch))
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        test_loss = []
        vae.eval()
        for batch in tqdm.tqdm(val_dl, desc=f"Epoch #{epoch + 1} test"):
            loss = vae.loss(batch, *vae(batch), kld_weight=0.0)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)
        print(f"#{epoch + 1} train_loss: {train_loss:.6f}, test_loss: {test_loss:.6f}")
        if test_loss < best_loss:
            best_loss = test_loss
            print(f"save model at epoch #{epoch + 1}")
            torch.save(
                vae.state_dict(), f"{training_config['model_save_path']}/best_vae.pth"
            )
        # print imgs for visualization
        orig_img = torch.as_tensor(val_ds[10], device=device, dtype=torch.float)
        vae_img = vae(orig_img[None])[0][0]
        # (C, H, W)/RGB -> (H, W, C)/BGR
        cv2.imwrite(
            f"{training_config['model_save_path']}/orig.png",
            orig_img.detach().cpu().numpy()[::-1].transpose(1, 2, 0) * 255,
        )
        cv2.imwrite(
            f"{training_config['model_save_path']}/vae.png",
            vae_img.detach().cpu().numpy()[::-1].transpose(1, 2, 0) * 255,
        )
