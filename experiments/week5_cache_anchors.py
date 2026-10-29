"""Compute and cache the frozen CLIP / CLAP anchors for one task (all splits), keyed by dataset index.

    python experiments/week5_cache_anchors.py common=cifar10

Anchors depend only on the data (and, for the text ceiling, the tokenizer), not on the channel, so this
runs once per task. Audio training anchors use a fixed set of crops (cfg.anchors.audio_train_crops per
utterance) because the loader's random crops cannot be cached.
"""
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from semcom.common_space.adapters import detokenize, make_adapter
from semcom.common_space.anchors import PretrainedAnchors, anchor_path, load_anchors, save_anchors
from semcom.common_space.loader import load_week4, make_dataset
from semcom.common_space.retrieval import random_pair_cosine


def pick_device(name: str) -> torch.device:
    return torch.device(name) if name != "auto" else torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def chunked(n: int, size: int):
    for s in range(0, n, size):
        yield range(s, min(s + size, n))


def encode_dataset(task: str, ds, adapter, chunk: int = 512, crops=None):
    """Anchors for every item (or every fixed crop) of `ds`, on CPU."""
    out = []
    n = len(ds) if crops is None else len(crops[0])
    for idxs in chunked(n, chunk):
        if crops is not None:
            x = torch.stack([ds.get_crop(int(crops[0][i]), int(crops[1][i])) for i in idxs])
            out.append(adapter.anchors.encode_audio(x, adapter.sample_rate).cpu())
        elif task == "text":
            out.append(adapter.anchors.encode_text([ds[i]["input"] for i in idxs]).cpu())
        else:
            x = torch.stack([ds[i]["input"] for i in idxs])
            out.append(adapter.anchors.encode_audio(x).cpu() if task == "audio" else adapter.anchors.encode_images(x).cpu())
    return torch.cat(out)


@hydra.main(config_path="../configs", config_name="week5", version_base=None)
def main(cfg: DictConfig) -> None:
    task, device = cfg.task, pick_device(cfg.device)
    print(f"[{task}] anchors on {device}; clip={cfg.anchors.clip} clap={cfg.anchors.clap}", flush=True)
    w4 = load_week4(task, "awgn", cfg.w4_root, "cpu")
    adapter = make_adapter(task, PretrainedAnchors(cfg.anchors.clip, cfg.anchors.clap, device), w4.spec)

    if task == "text":  # the tokenizer (hence the ceiling's round trip) must not depend on the channel
        other = torch.load(f"{cfg.w4_root}/text/rayleigh/best.pt", weights_only=False)["spec"]["vocab"]
        assert other == w4.spec.tokenizer.vocab, "AWGN and Rayleigh text tokenizers differ"

    for split in ("train", "val", "test"):
        ds = make_dataset(w4, split, cfg.data_root)
        t0, extra, crops = time.time(), {}, None
        if task == "audio" and split == "train":
            rng = np.random.default_rng(cfg.seed)
            k = cfg.anchors.audio_train_crops
            utt, starts = [], []
            for u in range(len(ds)):
                excess = ds.utterance_length(u) - ds.segment_length
                utt += [u] * k
                starts += rng.integers(0, max(excess, 0) + 1, size=k).tolist()
            crops = (torch.tensor(utt), torch.tensor(starts))
            extra = {"utt_idx": crops[0], "starts": crops[1]}
        anchors = encode_dataset(task, ds, adapter, crops=crops)
        if task == "text" and split == "test":
            texts = [ds[i]["input"] for i in range(len(ds))]
            extra["anchors_roundtrip"] = adapter.anchors.encode_text(adapter.roundtrip(texts)).cpu().half()
        save_anchors(anchor_path(cfg.anchor_root, task, split), anchors, model=cfg.anchors.clap if task == "audio" else cfg.anchors.clip, **extra)
        print(f"[{task}/{split}] {tuple(anchors.shape)} in {time.time() - t0:.0f}s | random-pair cosine "
              f"{random_pair_cosine(anchors):.3f}", flush=True)

    # sanity: freshly recomputed anchors for random test items match the cache (determinism on this device)
    ds = make_dataset(w4, "test", cfg.data_root)
    blob = load_anchors(anchor_path(cfg.anchor_root, task, "test"))
    pick = np.random.default_rng(0).choice(len(ds), size=min(64, len(ds)), replace=False)
    sub = [ds[int(i)] for i in pick]
    batch = w4.spec.to_device(w4.spec.collate(sub), device)
    fresh = adapter.encode_source(batch).cpu()
    cos = (fresh * blob["anchors"][pick]).sum(-1)
    print(f"[{task}] cache check: fresh vs cached anchor cosine min={cos.min():.5f} mean={cos.mean():.5f}", flush=True)


if __name__ == "__main__":
    main()
