import pytest
import torch

from semcom.data.image_mnist import MNISTDataset


@pytest.mark.slow
def test_shapes_types_and_split_sizes(tmp_path):
    train = MNISTDataset(root=str(tmp_path), split="train", val_fraction=0.1, seed=42)
    val = MNISTDataset(root=str(tmp_path), split="val", val_fraction=0.1, seed=42)
    test = MNISTDataset(root=str(tmp_path), split="test")

    assert len(train) + len(val) == 60000
    assert len(test) == 10000

    item = train[0]
    assert item["input"].shape == (1, 28, 28)
    assert item["input"].dtype == torch.float32
    assert 0 <= item["label"].item() < 10
