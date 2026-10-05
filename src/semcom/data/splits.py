import json
from pathlib import Path
from typing import Optional

import numpy as np


def _permutation(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n)


def train_val_indices(
    n: int, val_fraction: float, seed: int, cache_path: Optional[Path] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Split n items into (train, val) indices. Used when the source dataset
    already provides its own held-out test set (CIFAR-10, MNIST, Edinburgh audio)."""
    if cache_path is not None and cache_path.exists():
        obj = json.loads(cache_path.read_text())
        return np.array(obj["train"]), np.array(obj["val"])

    idx = _permutation(n, seed)
    n_val = int(round(n * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"train": train_idx.tolist(), "val": val_idx.tolist()}))
    return train_idx, val_idx


def train_val_test_indices(
    n: int, val_fraction: float, test_fraction: float, seed: int, cache_path: Optional[Path] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split n items into (train, val, test) indices. Used when the source only
    provides a single pool (Europarl)."""
    if cache_path is not None and cache_path.exists():
        obj = json.loads(cache_path.read_text())
        return np.array(obj["train"]), np.array(obj["val"]), np.array(obj["test"])

    idx = _permutation(n, seed)
    n_val = int(round(n * val_fraction))
    n_test = int(round(n * test_fraction))
    val_idx = idx[:n_val]
    test_idx = idx[n_val : n_val + n_test]
    train_idx = idx[n_val + n_test :]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()})
        )
    return train_idx, val_idx, test_idx
