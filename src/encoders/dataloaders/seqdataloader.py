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
from tqdm import tqdm

class SeqDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.files = os.listdir(data_path)
        self.cumulative_lengths = [0]
        for file_name in tqdm(self.files,desc='Loading data'):
            data = np.load(os.path.join(data_path, file_name))
            self.cumulative_lengths.append(len(data) + self.cumulative_lengths[-1] - 9)

        self.total_length = self.cumulative_lengths[-1]

    def find_file_and_index(self, idx):
        file_index = np.searchsorted(self.cumulative_lengths, idx + 1) - 1
        if file_index < 0:
            file_index = 0
        return file_index, idx - self.cumulative_lengths[file_index]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        file_index, idx = self.find_file_and_index(idx)
        file_name = self.files[file_index]
        data = np.load(os.path.join(self.data_path, file_name))
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