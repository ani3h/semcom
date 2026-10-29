from pathlib import Path

import torch
from torchvision import datasets, transforms

from .base import SemComDataset
from .splits import train_val_indices


class CIFAR100Dataset(SemComDataset):
    modality = "image"
    task_type = "classification"

    def __init__(self, root: str, split: str = "train", val_fraction: float = 0.1, seed: int = 42, download: bool = True):
        super().__init__(split)
        root = Path(root)
        is_test = split == "test"
        base = datasets.CIFAR100(root=str(root), train=not is_test, download=download, transform=transforms.ToTensor())
        self.class_names = [c.replace("_", " ") for c in base.classes]
        self._base = base
        if is_test:
            self._indices = list(range(len(base)))
        else:
            train_idx, val_idx = train_val_indices(len(base), val_fraction, seed, cache_path=root / "splits_cifar100.json")
            self._indices = train_idx if split == "train" else val_idx

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        img, label = self._base[int(self._indices[idx])]
        return {"input": img, "label": torch.tensor(label), "raw": None}
