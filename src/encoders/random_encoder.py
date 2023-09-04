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

@yamlize
class RandomEncoder(BaseEncoder, torch.nn.Module):
    def __init__(
        self,
        load_checkpoint_from: str = "",
    ):
        super().__init__()
    def encode(self, image):
        # image: np.array (H, W, C)
        # returns torch.Tensor(*) where * depends on the encoder

        linear = torch.nn.Linear(image.shape[0]*image.shape[1]*image.shape[2],32)
        return linear(torch.Tensor((image*np.random.randn(*(image.shape))).flatten())).unsqueeze(0)