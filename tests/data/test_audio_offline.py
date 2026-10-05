import numpy as np
import soundfile as sf
import torch

from semcom.data.audio_edinburgh import EdinburghAudioDataset

SR_SRC = 48000


def _make_corpus(root, n_train=6, n_test=3, seconds=3.0):
    base = root / "edinburgh_datashare"
    rng = np.random.default_rng(0)
    for dirname, n in (("clean_trainset_28spk_wav", n_train), ("clean_testset_wav", n_test)):
        d = base / dirname
        d.mkdir(parents=True)
        for i in range(n):
            sf.write(d / f"p{i:03d}.wav", rng.standard_normal(int(SR_SRC * seconds)).astype("float32") * 0.1, SR_SRC)


def _ds(root, split, **kw):
    kwargs = dict(root=str(root), urls={}, split=split, target_sample_rate=16000, segment_seconds=1.0,
                  val_fraction=0.34, seed=0, download=False)
    kwargs.update(kw)
    return EdinburghAudioDataset(**kwargs)


def test_resampled_cropped_shape_and_dtype(tmp_path):
    _make_corpus(tmp_path)
    item = _ds(tmp_path, "train")[0]
    assert item["input"].shape == (16000,)
    assert item["input"].dtype == torch.float32


def test_val_and_test_items_are_deterministic(tmp_path):
    _make_corpus(tmp_path)
    for split in ("val", "test"):
        ds = _ds(tmp_path, split)
        assert torch.equal(ds[0]["input"], ds[0]["input"])


def test_train_items_use_random_crops(tmp_path):
    _make_corpus(tmp_path)
    ds = _ds(tmp_path, "train")
    crops = [ds[0]["input"] for _ in range(5)]
    assert any(not torch.equal(crops[0], c) for c in crops[1:])


def test_short_clips_are_zero_padded(tmp_path):
    _make_corpus(tmp_path, seconds=0.5)
    item = _ds(tmp_path, "test")[0]
    assert item["input"].shape == (16000,)
    assert torch.count_nonzero(item["input"][8000:]) == 0


def test_split_sizes_and_cache_reuse(tmp_path):
    _make_corpus(tmp_path)
    train, val, test = (_ds(tmp_path, s) for s in ("train", "val", "test"))
    assert len(train) + len(val) == 6 and len(val) == 2 and len(test) == 3
    assert list((tmp_path / "edinburgh_datashare").glob("cache_*_16000hz.pt"))  # cache written once, reused
