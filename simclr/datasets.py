import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import random


class ContrastiveDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform

        with h5py.File(h5_path, 'r') as f:
            self.length = f['data'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            # Get original sample (raw view)
            orig_img = f['data'][idx]  # [13,288,288]
            orig_img = np.delete(orig_img, 12, axis=0)  # [12,288,288]
            orig_img = torch.from_numpy(orig_img).float()

            # Get temporal neighbor
            shift = random.choice([-2, -1,  1, 2])

            neighbor_idx = np.clip(idx + shift, 0, len(self) - 1)
            neighbor = f['data'][neighbor_idx]
            neighbor = np.delete(neighbor, 12, axis=0)
            neighbor = torch.from_numpy(neighbor).float()

        if self.transform:
            # Apply augmentation only to neighbor
            neighbor = self.transform(neighbor)

        return orig_img, neighbor