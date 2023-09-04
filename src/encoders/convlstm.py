import cv2
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from src.config.yamlize import yamlize
from src.constants import DEVICE
from src.encoders.base import BaseEncoder
from src.encoders.transforms.preprocessing import crop_resize_center


@yamlize
class ConvLSTM(BaseEncoder, torch.nn.Module):
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
        self.fc1 = nn.Linear(h_dim, z_dim)

        # self.fc1 = nn.Linear(h_dim, z_dim)
        # self.fc2 = nn.Linear(h_dim, z_dim)
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
            # nn.Sigmoid(),
        )
        num_layers = 3
        self.lstm = nn.LSTM(z_dim, z_dim, batch_first = True, num_layers = num_layers)
        self.lstm_hidden_init = torch.nn.Parameter(torch.normal(0, 1, size=(num_layers,1,z_dim)), requires_grad=True)
        self.lstm_cell_init = torch.nn.Parameter(torch.normal(0, 1, size=(num_layers,1,z_dim)), requires_grad=True)
        # if load_checkpoint_from == "":
        #     logging.info("Not loading any visual encoder checkpoint")
        # else:
        #     self.load_state_dict(torch.load(load_checkpoint_from,map_location=DEVICE))
        # TODO: Figure out where speed encoder should go.

    # def reparameterize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()
    #     esp = torch.randn(*mu.size(), device=mu.device)
    #     z = mu + std * esp
    #     return z

    # def bottleneck(self, h):
    #     # raise ValueError(h.shape)
    #     mu, logvar = self.fc1(h), self.fc2(h)
    #     z = self.reparameterize(mu, logvar)
    #     return z, mu, logvar

    # def representation(self, x):
    #     return self.bottleneck(self.encoder(x))[0]

    def encode(self, x: np.ndarray, device=DEVICE) -> torch.Tensor:
        # assume x is RGB image with shape (H, W, 3)
        cropped = torch.zeros((x.shape[0], 3, 42, 144)).to(device)
        for i in range(x.shape[0]):
            cropped[i] = crop_resize_center(x[i]).unsqueeze(0)
        # v = self.representation(h)
        v = self.fc1(self.encoder(cropped))
        return v

    # def distribution(self, x, device=DEVICE):
    #     # expects (N, H, W, C)
    #     h = self.encoder(x)
    #     z, mu, logvar = self.bottleneck(h)
    #     return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        return self.decoder(z)

    def forward(self, x):
        # OPERATE ON A SEQUENCE OF IMAGES
        # TODO: Modify encoding to work on a sequence of images
        seqlen = x.shape[1]
        lstm_hidden = self.lstm_hidden_init.repeat(1,x.shape[0], 1)
        lstm_cell = self.lstm_cell_init.repeat(1,x.shape[0],1)
        latents = []
        for i in range(seqlen):
            latentint = self.encode(x[:, i, :, :, :])
            latents.append(latentint)
        latent = torch.stack(latents, dim=1)
        lstmout,_ = self.lstm(latent, (lstm_hidden, lstm_cell))
        lasttoken = lstmout[:, -1, :]
        lstm_hidden = lstmout[1][0]
        lstm_cell = lstmout[1][1]
        h = lstm_hidden
        # LSTM STEP HERE
        # TODO: Add LSTM step here

        o = self.decode(lasttoken)
        return o
        # return z, mu, logvar

    def loss(self, actual, pred):

        # TIME CONSISTENCY LOSS as well as L2 loss or maybe KL loss for comparison.
        # recon, mu, logvar = pred
        # bce = F.binary_cross_entropy(recon, actual, reduction="sum")
        # kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        # return bce + kld * kld_weight
        actual = actual[:,0]
        cropped = torch.zeros((actual.shape[0], 3, 42, 144)).to(DEVICE)
        for i in range(actual.shape[0]):
            cropped[i] = crop_resize_center(actual[i]).unsqueeze(0)
        actual = cropped
        return F.mse_loss(pred, actual, reduction="sum")

    def update(self, batch_of_images):
        # TODO: Add train method here that makes sense
        pass