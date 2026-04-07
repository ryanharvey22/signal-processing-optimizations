"""MNIST via torchvision (backup / prototype for 2D FFT + wavelet pipelines)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    """Grayscale MNIST images as float tensor (1, 28, 28) in [0, 1]."""

    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: str = "train",
        download: bool = True,
    ) -> None:
        super().__init__()
        split = split.lower()
        if split not in ("train", "val", "test"):
            raise ValueError("split must be one of: train, val, test")

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        self._split = split
        if split == "test":
            self._base = datasets.MNIST(
                root=str(root),
                train=False,
                download=download,
                transform=transforms.ToTensor(),
            )
            self._indices: Optional[List[int]] = None
        else:
            self._base = datasets.MNIST(
                root=str(root),
                train=True,
                download=download,
                transform=transforms.ToTensor(),
            )
            n = len(self._base)
            g = torch.Generator().manual_seed(42)
            perm = torch.randperm(n, generator=g).tolist()
            n_train = int(0.9 * n)
            if split == "train":
                self._indices = perm[:n_train]
            else:
                self._indices = perm[n_train:]

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(self._base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self._indices is not None:
            idx = self._indices[idx]
        img, target = self._base[idx]
        return img, int(target)
