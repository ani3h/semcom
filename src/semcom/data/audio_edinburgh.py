import urllib.request
import zipfile
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from .base import SemComDataset
from .splits import train_val_indices


def _download_and_extract(url: str, dest_dir: Path, zip_name: str) -> None:
    zip_path = dest_dir / zip_name
    if not zip_path.exists():
        part_path = zip_path.with_suffix(".zip.part")
        urllib.request.urlretrieve(url, part_path)
        part_path.rename(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def _build_cache(wav_paths: list[Path], target_sample_rate: int, cache_path: Path) -> list[torch.Tensor]:
    """Decode + mono + resample every wav once and store as fp16, so training
    doesn't re-decode 48 kHz audio on every __getitem__."""
    waveforms = []
    for path in wav_paths:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data).mean(dim=1)
        if sr != target_sample_rate:
            wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
        waveforms.append(wav.half())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"names": [p.name for p in wav_paths], "waveforms": waveforms}, cache_path)
    return waveforms


class EdinburghAudioDataset(SemComDataset):
    """Valentini-Botinhao noisy speech database (Edinburgh DataShare, handle
    10283/2791). Clean waveforms are the "input" to be pushed through our own
    simulated channel; the dataset's noisy counterpart is only located when
    include_noisy=True and exposed under "raw" for later comparison.

    Train items get a random segment_seconds crop; val/test items get a fixed
    center crop so evaluation metrics are reproducible."""

    modality = "audio"
    task_type = "transmission"

    def __init__(
        self,
        root: str,
        urls: dict,
        split: str = "train",
        speakers: str = "28spk",
        target_sample_rate: int = 16000,
        segment_seconds: float = 2.0,
        val_fraction: float = 0.1,
        seed: int = 42,
        download: bool = True,
        include_noisy: bool = False,
    ):
        super().__init__(split)
        self.target_sample_rate = target_sample_rate
        self.segment_length = int(segment_seconds * target_sample_rate)

        root = Path(root) / "edinburgh_datashare"
        root.mkdir(parents=True, exist_ok=True)
        is_test = split == "test"

        clean_key = "clean_test" if is_test else "clean_train"
        clean_dir_name = "clean_testset_wav" if is_test else f"clean_trainset_{speakers}_wav"
        clean_dir = root / clean_dir_name
        if download and not clean_dir.exists():
            _download_and_extract(urls[clean_key], root, f"{clean_dir_name}.zip")

        self.noisy_paths: dict[str, Path] = {}
        if include_noisy:
            noisy_key = "noisy_test" if is_test else "noisy_train"
            noisy_dir_name = "noisy_testset_wav" if is_test else f"noisy_trainset_{speakers}_wav"
            noisy_dir = root / noisy_dir_name
            if download and not noisy_dir.exists():
                _download_and_extract(urls[noisy_key], root, f"{noisy_dir_name}.zip")
            self.noisy_paths = {p.stem: p for p in noisy_dir.glob("*.wav")}

        clean_paths = sorted(clean_dir.glob("*.wav"))
        if not clean_paths:
            raise RuntimeError(f"No .wav files found under {clean_dir}; check download/extraction.")

        cache_path = root / f"cache_{clean_dir_name}_{target_sample_rate}hz.pt"
        if cache_path.exists():
            self._waveforms = torch.load(cache_path)["waveforms"]
        else:
            self._waveforms = _build_cache(clean_paths, target_sample_rate, cache_path)
        self._names = [p.stem for p in clean_paths]

        if is_test:
            self._indices = list(range(len(clean_paths)))
        else:
            train_idx, val_idx = train_val_indices(
                len(clean_paths), val_fraction, seed, cache_path=root / f"splits_{speakers}.json"
            )
            self._indices = [int(i) for i in (train_idx if split == "train" else val_idx)]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        i = self._indices[idx]
        waveform = self._waveforms[i].float()

        excess = waveform.shape[0] - self.segment_length
        if excess >= 0:
            start = int(torch.randint(0, excess + 1, (1,))) if self.split == "train" else excess // 2
            waveform = waveform[start : start + self.segment_length]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, -excess))

        noisy_path = self.noisy_paths.get(self._names[i])
        return {"input": waveform, "label": None, "raw": str(noisy_path) if noisy_path else None}
