from pathlib import Path

import torch
from torchvision import datasets, transforms

from .base import SemComDataset
from .splits import train_val_indices


class MNISTDataset(SemComDataset):
    modality = "image"
    task_type = "classification"

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
        download: bool = True,
    ):
        super().__init__(split)
        root = Path(root)
        transform = transforms.ToTensor()
        is_test = split == "test"
        base = datasets.MNIST(root=str(root), train=not is_test, download=download, transform=transform)

        if is_test:
            self._base = base
            self._indices = list(range(len(base)))
        else:
            train_idx, val_idx = train_val_indices(
                len(base), val_fraction, seed, cache_path=root / "splits_mnist.json"
            )
            self._base = base
            self._indices = train_idx if split == "train" else val_idx

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        img, label = self._base[int(self._indices[idx])]
        return {"input": img, "label": torch.tensor(label), "raw": None}
