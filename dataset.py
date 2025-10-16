# dataset.py
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class KeypointSequenceDataset(Dataset):
    def __init__(self, seq_dir="data/sequences"):
        self.files = sorted(glob.glob(seq_dir + "/*.npz"))
        if len(self.files) == 0:
            raise RuntimeError("No sequence files found in data/sequences. Run preprocess.py first.")
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        kp = data['keypoints'].astype(np.float32)  # T x K x 2
        label = int(data['label'])
        T, K, C = kp.shape
        flat = kp.reshape(T, K*C)  # T x (K*2)
        # normalize per-sequence (zero mean, unit std)
        mean = flat.mean()
        std = flat.std() if flat.std() > 0 else 1.0
        flat = (flat - mean) / std
        return torch.from_numpy(flat), torch.tensor(label, dtype=torch.long)
