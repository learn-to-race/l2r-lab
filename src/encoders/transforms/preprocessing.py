import cv2
from PIL import Image
import numpy as np
import torch
from src.constants import DEVICE

def crop_resize_center(img):
    assert img.shape == (384, 512, 3)
    img = cv2.convertScaleAbs(img)
    img = cv2.Canny(img, 100, 200) # Canny Edge Detector
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    p = cv2.resize(img, (144, 144))[68:110] / 255
    x = p.transpose(2, 0, 1)  # (H, W, C) --> (C, H, W)
    x = torch.as_tensor(x, device=DEVICE, dtype=torch.float)
    return x
