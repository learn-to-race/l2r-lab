import torch
from torch.utils.data import Dataset
import numpy as np
import os

import torchvision
from torch import nn
import torch.nn.functional as F
import cv2
from collections import OrderedDict
from src.encoders.dataloaders.base import BaseDataFetcher
from src.config.yamlize import yamlize


class SeqDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.files = os.listdir(data_path)
        self.data = []
        for file_name in self.files:
            file_path = os.path.join(data_path, file_name)
            data = np.load(file_path)
            self.data.append(data)

        self.lengths = [len(data) - 9 for data in self.data]
        self.cumulative_lengths = np.cumsum(self.lengths)

    # def load_folder(self, folder):
    #     rgb_img_path = os.path.join(folder, "rgb_imgs")
    #     segm_img_path = os.path.join(folder, "segm_imgs")
    #     out = []
    #     for i in os.listdir(rgb_img_path):
    #         r = os.path.join(rgb_img_path, i)
    #         s = os.path.join(segm_img_path, i)
    #         if not os.path.isfile(s):
    #             continue
    #         out.append((r, s))
    #     return out

    def __len__(self):
        return self.cumulative_lengths[-1]

    # def prepare_rgb(self, img_path):
    #     img = cv2.imread(img_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     x = np.array(cv2.resize(img, (512, 384)))
    #     x = torch.Tensor(x.transpose(2, 0, 1)) / 255
    #     return x

    # def prepare_segm(self, img_path):
    #     img = cv2.imread(img_path)
    #     mask = np.where(img == (109, 80, 204), 1, 0).astype(np.uint8)
    #     mask = cv2.resize(mask, (512, 384))[:, :, 1]
    #     mask = torch.Tensor(mask)
    #     return mask

    def __getitem__(self, idx):
        file_index = np.searchsorted(self.cumulative_lengths, idx + 1)
        if file_index > 0:
            idx -= self.cumulative_lengths[file_index - 1]

        data = self.data[file_index]
        start = idx
        end = idx + 10
        return torch.from_numpy(data[start:end]).float()


@yamlize
class SeqDataFetcher(BaseDataFetcher):
    def __init__(self, train_path: str, val_path: str):
        self.train_path = train_path
        self.val_path = val_path

    def get_dataloaders(self, batch_size, device):
        train_ds = SeqDataset(self.train_path)
        val_ds = SeqDataset(self.val_path)
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True
        )
        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False
        )
        return train_ds, val_ds, train_dl, val_dl