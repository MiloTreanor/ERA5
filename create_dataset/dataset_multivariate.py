from torch.utils.data import Dataset
import h5py
import numpy as np


class SingleStepPrecipDataset(Dataset):
    def __init__(self, h5_file, transform=None):
        self.h5_path = h5_file
        self.transform = transform

        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['labels'].shape[0]  # Directly use the full dataset

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            x = np.array(f['data'][idx], dtype=np.float16)  # [C, H, W]
            x = np.delete(x, 12, axis=0)  # Adjust if your channel count differs
            y = np.array(f['labels'][idx], dtype=np.float16)  # [H, W]

        if self.transform:
            x = self.transform(x)
        return x, y