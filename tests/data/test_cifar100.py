import pytest
import torch

from semcom.data.image_cifar100 import CIFAR100Dataset


@pytest.mark.slow
def test_shapes_types_and_split_sizes(tmp_path):
    train = CIFAR100Dataset(root=str(tmp_path), split="train", val_fraction=0.1, seed=42)
    val = CIFAR100Dataset(root=str(tmp_path), split="val", val_fraction=0.1, seed=42)
    test = CIFAR100Dataset(root=str(tmp_path), split="test")
    assert len(train) + len(val) == 50000 and len(test) == 10000
    item = train[0]
    assert item["input"].shape == (3, 32, 32) and item["input"].dtype == torch.float32
    assert 0 <= item["label"].item() < 100 and len(train.class_names) == 100
