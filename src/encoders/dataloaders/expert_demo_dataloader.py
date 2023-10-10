import torch
from tqdm import tqdm
import numpy as np
import os
from src.encoders.dataloaders.base import BaseDataFetcher

from src.encoders.transforms.preprocessing import crop_resize_center
from src.config.yamlize import yamlize


class ExpertDemoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.imgs = []
        # files = os.listdir(input_path)
        # files = sorted(files, key=lambda x: int(re.search("_(\d+).npz", x ).group(1))) # TODO get all train/episode_n

        files = [
            os.path.join(dir, fname)
            for dir, _, flist in os.walk(data_dir)
            for fname in flist
        ]
        for filename in tqdm(files, desc=f"Loading files..."):
            if filename.endswith(".npz"):
                data = np.load(filename)["img"]
                self.imgs.append(crop_resize_center(data))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]


@yamlize
class ExpertDemoDataFetcher(BaseDataFetcher):
    def __init__(self, train_path: str, val_path: str):
        super(ExpertDemoDataFetcher, self).__init__()
        self.train_path = train_path
        self.val_path = val_path

    def get_dataloaders(self, batch_size, device):
        def collate(batch):
            return torch.stack(batch).to(device)

        train_ds = ExpertDemoDataset(self.train_path)
        val_ds = ExpertDemoDataset(self.val_path)
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, collate_fn=collate, shuffle=True
        )
        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, collate_fn=collate, shuffle=False
        )
        return train_ds, val_ds, train_dl, val_dl
