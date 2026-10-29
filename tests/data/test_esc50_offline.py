import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from semcom.data.audio_esc50 import ESC50Dataset


def _fake_esc50(root, per_fold=2, classes=("dog", "chirping_birds", "sea_waves")):
    top = root / "esc50" / "ESC-50-master"
    (top / "audio").mkdir(parents=True)
    (top / "meta").mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(1, 6):
        for k in range(per_fold):
            t = (fold + k) % len(classes)
            name = f"{fold}-{100 + k}-A-{t}.wav"
            sf.write(top / "audio" / name, (rng.standard_normal(44100 * 5) * 0.1).astype("float32"), 44100)
            rows.append({"filename": name, "fold": fold, "target": t, "category": classes[t], "esc10": False, "src_file": 1, "take": "A"})
    pd.DataFrame(rows).to_csv(top / "meta" / "esc50.csv", index=False)


def test_esc50_splits_shapes_and_names(tmp_path):
    _fake_esc50(tmp_path)
    ds = {s: ESC50Dataset(str(tmp_path), split=s, download=False) for s in ("train", "val", "test", "all")}
    assert [len(ds[s]) for s in ("train", "val", "test", "all")] == [6, 2, 2, 10]  # folds 1-3 / 4 / 5 / everything
    item = ds["train"][0]
    assert item["input"].shape == (80000,) and item["input"].dtype == torch.float32  # 5 s at 16 kHz
    assert item["label"].dtype == torch.long and isinstance(item["raw"], str)
    assert ds["all"].class_names == ["dog", "chirping birds", "sea waves"]  # underscores replaced for prompts
    assert ds["all"].categories[1] == "chirping_birds"  # original ids kept for the CIFAR-10 mapping


def test_esc50_is_deterministic_cached_and_exposes_fullband(tmp_path):
    _fake_esc50(tmp_path)
    ds = ESC50Dataset(str(tmp_path), split="test", download=False)
    assert torch.equal(ds[0]["input"], ds[0]["input"])
    assert (tmp_path / "esc50" / "cache_16000hz.pt").exists()
    wave, sr = ds.fullband(0)
    assert sr == 44100 and wave.shape == (44100 * 5,)


def test_esc50_missing_data_without_download_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ESC50Dataset(str(tmp_path), split="train", download=False)


def test_train_and_val_splits_never_contain_fold_5_files(tmp_path):
    _fake_esc50(tmp_path)
    ds = {s: ESC50Dataset(str(tmp_path), split=s, download=False) for s in ("train", "val", "test")}
    folds = {s: set(ds[s].meta["fold"]) for s in ds}
    assert folds["train"] == {1, 2, 3} and folds["val"] == {4} and folds["test"] == {5}
    assert not (set(ds["train"].meta["filename"]) | set(ds["val"].meta["filename"])) & set(ds["test"].meta["filename"])
