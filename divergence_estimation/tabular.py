import torch

import numpy as np

from typing import Union
from torch.utils.data import DataLoader, Dataset

_MAX_DIMS = 2


class _NumpyDataset(Dataset):
    def __init__(self, X, y=None) -> None:
        self.X = X
        self.y = None
        if y is not None:
            self.y = y

    def __getitem__(self, index):
        if self.y is not None:
            return self.X[index], self.y[index]
        return self.X[index]

    def __len__(self):
        return len(self.X)

def tensor_to_dataloader(
    X: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray, None] = None, batch_size=1000000, shuffle=False
):
    ds = _NumpyDataset(X, y)
    dl = DataLoader(ds, batch_size, shuffle=shuffle)
    return dl