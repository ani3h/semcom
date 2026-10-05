import pytest
import torch

from semcom.data.image_cifar10 import CIFAR10Dataset


@pytest.mark.slow
def test_shapes_types_and_split_sizes(tmp_path):
    train = CIFAR10Dataset(root=str(tmp_path), split="train", val_fraction=0.1, seed=42)
    val = CIFAR10Dataset(root=str(tmp_path), split="val", val_fraction=0.1, seed=42)
    test = CIFAR10Dataset(root=str(tmp_path), split="test")

    assert len(train) + len(val) == 50000
    assert len(test) == 10000

    item = train[0]
    assert item["input"].shape == (3, 32, 32)
    assert item["input"].dtype == torch.float32
    assert 0 <= item["label"].item() < 10
