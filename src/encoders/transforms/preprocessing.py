import cv2
from PIL import Image
import numpy as np
import torch
from src.constants import DEVICE


def crop_resize_center(img):
    assert img.shape == (
        384,
        512,
        3,
    ), f"img shape should be (384, 512, 3), found {img.shape}"
    p = cv2.resize(img.numpy(), (144, 144))[68:110] / 255
    x = p.transpose(2, 0, 1)  # (H, W, C) --> (C, H, W)
    x = torch.as_tensor(x, device=DEVICE, dtype=torch.float)
    return x
