from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from .base import l2_normalize

CLIP_NAME = "openai/clip-vit-base-patch32"
CLAP_NAME = "laion/clap-htsat-unfused"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLAP_SAMPLE_RATE = 48000


def _features(out) -> torch.Tensor:
    """transformers returns a tensor or a ModelOutput depending on version."""
    return out if torch.is_tensor(out) else out.pooler_output


def clip_preprocess(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) in [0, 1] -> CLIP input: grey -> RGB, bicubic to 224, normalise."""
    if x.size(1) == 1:
        x = x.expand(-1, 3, -1, -1)
    x = F.interpolate(x.float(), size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
    mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


_FE = None


def _fe_init(name: str) -> None:
    global _FE
    from transformers import AutoFeatureExtractor

    _FE = AutoFeatureExtractor.from_pretrained(name)


def _fe_run(waves: list) -> dict:
    return {k: v.numpy() for k, v in _FE(waves, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt").items()}


def _chunks(n: int, size: int):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


class FrozenCLIP:
    def __init__(self, name: str = CLIP_NAME, device="cpu"):
        from transformers import AutoTokenizer, CLIPModel

        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(name).eval().to(self.device).requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(name)

    @torch.no_grad()
    def encode_images(self, x: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        outs = [_features(self.model.get_image_features(pixel_values=clip_preprocess(x[s].to(self.device))))
                for s in _chunks(x.size(0), batch_size)]
        return l2_normalize(torch.cat(outs))

    @torch.no_grad()
    def encode_text(self, texts: list[str], batch_size: int = 256) -> torch.Tensor:
        outs = []
        for s in _chunks(len(texts), batch_size):
            tok = self.tokenizer(texts[s], padding=True, truncation=True, max_length=77, return_tensors="pt")
            outs.append(_features(self.model.get_text_features(
                input_ids=tok["input_ids"].to(self.device), attention_mask=tok["attention_mask"].to(self.device))))
        return l2_normalize(torch.cat(outs))


class FrozenCLAP:
    def __init__(self, name: str = CLAP_NAME, device="cpu"):
        from transformers import AutoFeatureExtractor, AutoTokenizer, ClapModel

        self.name = name
        self.device = torch.device(device)
        self.model = ClapModel.from_pretrained(name).eval().to(self.device).requires_grad_(False)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)

    @torch.no_grad()
    def encode_text(self, texts: list[str], batch_size: int = 256) -> torch.Tensor:
        """CLAP text-tower embeddings (unit norm): the same space as CLAP audio embeddings."""
        outs = []
        for s in _chunks(len(texts), batch_size):
            tok = self.tokenizer(texts[s], padding=True, truncation=True, max_length=77, return_tensors="pt")
            tok = {k: v.to(self.device) for k, v in tok.items()}
            outs.append(_features(self.model.get_text_features(**tok)))
        return l2_normalize(torch.cat(outs))

    def _pool(self, workers: int):
        if getattr(self, "_executor", None) is None or self._executor_workers != workers:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            self._executor = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_fe_init, initargs=(self.name,))
            self._executor_workers = workers
        return self._executor

    @torch.no_grad()
    def encode_audio(self, wave: torch.Tensor, sample_rate: int = 16000, batch_size: int = 64, num_workers: int = 0) -> torch.Tensor:
        """(B, N) waveforms -> unit CLAP audio embeddings. CLAP wants 48 kHz, padded/repeated to 10 s.
        num_workers > 0 runs the (CPU-bound) feature extraction in a process pool; the result is identical."""
        wave = wave.detach().float().cpu()
        if sample_rate != CLAP_SAMPLE_RATE:
            wave = torchaudio.functional.resample(wave, sample_rate, CLAP_SAMPLE_RATE)
        outs = []
        for s in _chunks(wave.size(0), batch_size):
            waves = [w.numpy() for w in wave[s]]
            if num_workers > 0:
                parts = [waves[i::num_workers] for i in range(num_workers)]
                futs = [self._pool(num_workers).submit(_fe_run, p) for p in parts if p]
                res = [f.result() for f in futs]
                order = [j for i in range(num_workers) for j in range(i, len(waves), num_workers)]  # undo the round-robin split
                feats = {k: torch.from_numpy(np.concatenate([r[k] for r in res]))[np.argsort(order)] for k in res[0]}
            else:
                feats = self.feature_extractor(waves, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt")
            feats = {k: v.to(self.device) for k, v in feats.items()}
            outs.append(_features(self.model.get_audio_features(**feats)))
        return l2_normalize(torch.cat(outs))


class PretrainedAnchors:
    """Lazy holder for the frozen CLIP / CLAP encoders (only what a run touches gets loaded).
    Tests substitute stubs exposing the same encode_images / encode_text / encode_audio."""

    def __init__(self, clip_name: str = CLIP_NAME, clap_name: str = CLAP_NAME, device="cpu"):
        self.clip_name, self.clap_name, self.device = clip_name, clap_name, device
        self._clip = self._clap = None

    @property
    def clip(self) -> FrozenCLIP:
        if self._clip is None:
            self._clip = FrozenCLIP(self.clip_name, self.device)
        return self._clip

    @property
    def clap(self) -> FrozenCLAP:
        if self._clap is None:
            self._clap = FrozenCLAP(self.clap_name, self.device)
        return self._clap

    def encode_images(self, x):
        return self.clip.encode_images(x)

    def encode_text(self, texts):
        return self.clip.encode_text(texts)

    def encode_audio(self, wave, sample_rate: int = 16000, num_workers: int = 0):
        return self.clap.encode_audio(wave, sample_rate, num_workers=num_workers)

    def encode_text_clap(self, texts):
        return self.clap.encode_text(texts)


# ---------------------------------------------------------------- anchor cache (keyed by dataset index)
def anchor_path(root, task: str, split: str, suffix: str = "") -> Path:
    return Path(root) / f"{task}_{split}{suffix}.pt"


def save_anchors(path, anchors: torch.Tensor, **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"anchors": anchors.detach().cpu().half(), **extra}, path)


def load_anchors(path) -> dict:
    blob = torch.load(path)
    blob["anchors"] = l2_normalize(blob["anchors"].float())  # undo fp16 rounding of the norm
    return blob
