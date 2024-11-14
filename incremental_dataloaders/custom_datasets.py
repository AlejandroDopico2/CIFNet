import torch
from torch.utils.data import Dataset, Subset


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.targets = y

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.X[idx], self.targets[idx]


class TensorSubset(Subset):
    def __getitem__(self, index):
        x, y = self.dataset[self.indices[index]]
        return x, torch.tensor(y) if not isinstance(y, torch.Tensor) else y
