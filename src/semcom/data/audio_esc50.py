import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio

from .base import SemComDataset

URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
SPLIT_FOLDS = {"train": (1, 2, 3), "val": (4,), "test": (5,), "all": (1, 2, 3, 4, 5)}


class ESC50Dataset(SemComDataset):
    """ESC-50 environmental sounds (2000 five-second clips, 50 classes, CC BY-NC).

    Splits follow the official folds: train = 1-3, val = 4, test = 5 ("all" = every fold, for zero-shot
    evaluation that fits no model). The "input" is the clip resampled to target_sample_rate (16 kHz, the
    rate the audio transceivers run at) as float32 mono; the original 44.1 kHz audio is available through
    `fullband(idx)` for reference measurements. Decoded 16 kHz audio is cached once as fp16.
    """

    modality = "audio"
    task_type = "classification"

    def __init__(self, root: str, split: str = "train", target_sample_rate: int = 16000, download: bool = True):
        super().__init__(split)
        self.target_sample_rate = target_sample_rate
        base = Path(root) / "esc50"
        base.mkdir(parents=True, exist_ok=True)
        top = base / "ESC-50-master"
        if not top.exists():
            if not download:
                raise FileNotFoundError(f"{top} missing and download=False")
            zip_path = base / "ESC-50-master.zip"
            if not zip_path.exists():
                urllib.request.urlretrieve(URL, zip_path.with_suffix(".part"))
                zip_path.with_suffix(".part").rename(zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(base)
        self.audio_dir = top / "audio"
        meta = pd.read_csv(top / "meta" / "esc50.csv")
        names = meta.drop_duplicates("target").sort_values("target")["category"].tolist()
        self.class_names = [c.replace("_", " ") for c in names]
        self.categories = names  # original underscored ids, e.g. "chirping_birds"
        meta = meta[meta["fold"].isin(SPLIT_FOLDS[split])].sort_values("filename").reset_index(drop=True)
        self.meta = meta

        cache = base / f"cache_{target_sample_rate}hz.pt"
        if cache.exists():
            blob = torch.load(cache)
        else:
            all_meta = pd.read_csv(top / "meta" / "esc50.csv").sort_values("filename")
            waves = {}
            for fn in all_meta["filename"]:
                data, sr = sf.read(str(self.audio_dir / fn), dtype="float32", always_2d=True)
                wav = torch.from_numpy(data).mean(dim=1)
                if sr != target_sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
                waves[fn] = wav.half()
            blob = waves
            torch.save(blob, cache)
        self._waves = blob

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        row = self.meta.iloc[idx]
        return {"input": self._waves[row["filename"]].float(), "label": torch.tensor(int(row["target"])),
                "raw": row["category"]}

    def fullband(self, idx: int) -> tuple[torch.Tensor, int]:
        """Original-rate mono waveform and its sample rate (no resampling)."""
        data, sr = sf.read(str(self.audio_dir / self.meta.iloc[idx]["filename"]), dtype="float32", always_2d=True)
        return torch.from_numpy(data).mean(dim=1), sr
