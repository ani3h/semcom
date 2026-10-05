import pytest
from omegaconf import OmegaConf

from semcom.data.audio_edinburgh import EdinburghAudioDataset

URLS = {
    "clean_train": "https://datashare.ed.ac.uk/bitstreams/245452b6-6235-44b6-a6f9-e7eb19797769/download",
    "noisy_train": "https://datashare.ed.ac.uk/bitstreams/ecb5a102-bb00-46d3-8af5-40c79823b837/download",
    "clean_test": "https://datashare.ed.ac.uk/bitstreams/dec213d3-bf57-4777-9663-c24bdce92d5e/download",
    "noisy_test": "https://datashare.ed.ac.uk/bitstreams/13c1bfbf-14a6-41db-9b41-8f7310f01ad5/download",
}


@pytest.mark.slow
def test_shapes_types_and_split_sizes(tmp_path):
    kwargs = dict(
        root=str(tmp_path),
        urls=OmegaConf.create(URLS),
        target_sample_rate=16000,
        segment_seconds=2.0,
        val_fraction=0.1,
        seed=42,
    )
    train = EdinburghAudioDataset(split="train", **kwargs)
    val = EdinburghAudioDataset(split="val", **kwargs)
    test = EdinburghAudioDataset(split="test", **kwargs)

    total = len(train) + len(val)
    assert total > 0
    assert abs(len(val) / total - 0.1) < 0.02  # requested val_fraction=0.1
    assert len(test) > 0

    item = train[0]
    assert item["input"].shape == (32000,)  # 2s @ 16kHz
    assert item["input"].dtype.is_floating_point
