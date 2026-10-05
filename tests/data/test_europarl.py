import pytest

from semcom.data.text_europarl import EuroparlDataset


@pytest.mark.slow
def test_shapes_types_and_split_sizes(tmp_path):
    kwargs = dict(root=str(tmp_path), lang="en", pair_lang="de", max_samples=1000, val_fraction=0.1, test_fraction=0.1, seed=42)
    train = EuroparlDataset(split="train", **kwargs)
    val = EuroparlDataset(split="val", **kwargs)
    test = EuroparlDataset(split="test", **kwargs)

    assert len(train) + len(val) + len(test) == 1000
    item = train[0]
    assert isinstance(item["input"], str)
    assert len(item["input"]) > 0
