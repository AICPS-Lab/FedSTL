"""
PyTorch custom sequential dataset.
"""
import torch
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, x, y, device='cuda'):
        self.x = torch.tensor(x).float().to(device)
        self.y = torch.tensor(y).float().to(device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i): 
        return self.x[i], self.y[i]