import cv2
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys
import os

sys.path.append(os.path.abspath('./') + '/src/encoders')
sys.path.append(os.path.abspath('./') + '/src/encoders/segmentation_model')

print(sys.path)

from src.config.yamlize import yamlize
from src.constants import DEVICE
from src.encoders.base import BaseEncoder
from src.encoders.transforms.preprocessing import crop_resize_center
from segmentation_model.inference import Inference

import yaml
from datetime import datetime
from train import setup_seeds, expand_cfg_vars
from configs.machine_config import MachineConfig
from PIL import Image

@yamlize
class SEGVAE(BaseEncoder, torch.nn.Module):
    """Input should be (bsz, C, H, W) where C=3, H=42, W=144"""

    def __init__(
        self,
        image_channels: int = 3,
        image_height: int = 42,
        image_width: int = 144,
        z_dim: int = 32,
        load_checkpoint_from: str = "",
    ):
        super().__init__()

        self.im_c = image_channels
        self.im_h = image_height
        self.im_w = image_width
        encoder_list = [
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        ]
        self.encoder = nn.Sequential(*encoder_list)
        sample_img = torch.zeros([1, image_channels, image_height, image_width])
        em_shape = nn.Sequential(*encoder_list[:-1])(sample_img).shape[1:]
        h_dim = np.prod(em_shape)

        # Creating the config object
        # Need to replace args.model with path
        model_files_path = "/mnt/l2r-lab/model_files/sem_seg_models/cityscapes_model"

        checkpoint_file = os.path.join(model_files_path, "best_model_without_opt.pkl")
        cfg_file        = os.path.join(model_files_path, "cfg.yml")
        with open(cfg_file) as fp:
            cfg = yaml.safe_load(fp)

        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        cfg['machine'] = 'ws'
        cfg['data']['dataset'] = 'inference'
        cfg['model']['disable_pose'] = True
        cfg['name'] = 'inference' + run_id
        cfg['training']['log_path'] = os.path.join(cfg['training']['log_path'], cfg['name']) + '/'
        cfg['training']['resume'] = checkpoint_file
        name = cfg['name']

        MachineConfig(cfg["machine"])
        expand_cfg_vars(cfg)

        logdir = cfg['training']['log_path']

        # Creating the inference object
        self.infer = Inference(cfg, logdir, os.path.join(name, str(run_id)))


        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, em_shape),
            nn.ConvTranspose2d(
                em_shape[0],
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=(1, 0),
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1, output_padding=(1, 0)
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        if load_checkpoint_from == "":
            logging.info("Not loading any visual encoder checkpoint")
        else:
            self.load_state_dict(torch.load(load_checkpoint_from))
        # TODO: Figure out where speed encoder should go.

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=mu.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        # raise ValueError(h.shape)
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def image_resize(self,
                     x,
                     depth = False,
                     nrow = 8,
                     padding = 2,
                     normalize = False,
                     img_range = None,
                     scale_each = False,
                     pad_value = 0):

        # grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, range= img_range, scale_each=scale_each)
        grid = x

        if depth:
            ndarr = grid.to('cpu', torch.uint8).numpy()
            ndarr = np.squeeze(ndarr, axis = 0)
            im = Image.fromarray(ndarr, 'L')

        else:
            # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)

        im = im.resize((512, 384))
        im2arr = np.array(im)

        if depth:
            d_arr = im2arr
            im2arr = np.zeros((384, 512, 3))
            im2arr[:, :, 0] = d_arr
            im2arr[:, :, 1] = d_arr
            im2arr[:, :, 2] = d_arr

        return im2arr

    def encode(self, x: np.ndarray, device=DEVICE) -> torch.Tensor:
        # assume x is RGB image with shape (H, W, 3)
        print("X shape: ", x.shape, "X type: ", type(x))
        #if x.shape != (384, 512, 3):
        #    print("wrong X shape: ", x.shape)
        #    x = self.prev_frame
        try:
            seg_op, depth_op = self.infer.run_for_img(x)
        except:
            print("wrong X shape: ", x.shape)
            return self.prev_v
        # print("SRI DEBUG: depth_op shape: ", depth_op.shape)
        # seg_np_op = self.image_resize(seg_op)
        depth_np_op = self.image_resize(depth_op, depth = True)
        # print("transform done, seg_op shape", seg_op.shape)
        # print("transform done, depth_np_op shape: ", depth_np_op.shape)
        # print("input shape: ", x.shape)
        h = crop_resize_center(depth_np_op).unsqueeze(0)
        v = self.representation(h)

        self.prev_v = v

        #print("representation done")
        return v

    def distribution(self, x, device=DEVICE):
        # expects (N, H, W, C)
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        return self.decoder(z)

    def forward(self, x):
        # expects (N, H, W, C)
        z, mu, logvar = self.distribution(x)
        z = self.decode(z)
        return z, mu, logvar

    def loss(self, actual, pred, kld_weight=1.0):
        recon, mu, logvar = pred
        bce = F.binary_cross_entropy(recon, actual, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return bce + kld * kld_weight

    def update(self, batch_of_images):
        # TODO: Add train method here that makes sense
        pass
