"""Stage 0, step 1: build the caption corpus and cache every embedding the bridge and the gate need.

    python experiments/week5b_prepare.py

Caches (under data_cache/): captions/corpus.json, bridge/{split}_{clap_text,clip_text}.pt,
anchors/esc50_clap_{16k,fullband}.pt, anchors/cifar100_{train,test}.pt. Re-running skips what exists.
"""
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from semcom.common_space.anchors import PretrainedAnchors, anchor_path, save_anchors
from semcom.common_space.captions import build_corpus, load_corpus, save_corpus
from semcom.data.audio_esc50 import ESC50Dataset
from semcom.data.image_cifar100 import CIFAR100Dataset


def pick_device(name: str) -> torch.device:
    return torch.device(name) if name != "auto" else torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def chunked_encode(fn, items, chunk: int, label: str):
    out, t0 = [], time.time()
    for s in range(0, len(items), chunk):
        out.append(fn(items[s:s + chunk]).cpu())
        print(f"    {label}: {min(s + chunk, len(items))}/{len(items)} [{time.time() - t0:.0f}s]", flush=True)
    return torch.cat(out)


@hydra.main(config_path="../configs", config_name="week5b", version_base=None)
def main(cfg: DictConfig) -> None:
    device = pick_device(cfg.device)
    anchors = PretrainedAnchors(device=device)
    bridge_root, root = Path(cfg.bridge_root), Path(cfg.captions.root)

    corpus_path = root / "corpus.json"
    if corpus_path.exists():
        corpus = load_corpus(corpus_path)
    else:
        corpus = build_corpus(root, cfg.seed, cfg.captions.n_coco_train, cfg.captions.n_coco_eval, cfg.captions.min_words)
        save_corpus(corpus, corpus_path)
    for split, rows in corpus.items():
        src = {s: sum(r["source"] == s for r in rows) for s in ("coco", "audiocaps")}
        print(f"[captions] {split}: {len(rows)} captions {src}", flush=True)

    for split, rows in corpus.items():
        texts = [r["text"] for r in rows]
        for name, fn in (("clap_text", anchors.encode_text_clap), ("clip_text", anchors.encode_text)):
            path = bridge_root / f"{split}_{name}.pt"
            if not path.exists():
                emb = chunked_encode(fn, texts, 5000, f"{split}/{name}")
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(emb.half(), path)

    for tag, kw in (("16k", {}), ("fullband", {"fullband": True})):
        path = anchor_path(cfg.anchor_root, "esc50", "all", f"_clap_{tag}")
        if path.exists():
            continue
        ds = ESC50Dataset(cfg.data_root, split="all")
        embs = []
        for s in range(0, len(ds), 50):
            idx = range(s, min(s + 50, len(ds)))
            if kw:  # original 44.1 kHz audio straight to CLAP's 48 kHz (reference for the 16 kHz band-limit cost)
                waves = [ds.fullband(i) for i in idx]
                embs.append(torch.cat([anchors.clap.encode_audio(w.unsqueeze(0), sr) for w, sr in waves]).cpu())
            else:
                embs.append(anchors.encode_audio(torch.stack([ds[i]["input"] for i in idx]), 16000).cpu())
        save_anchors(path, torch.cat(embs), filenames=list(ds.meta["filename"]), folds=torch.tensor(ds.meta["fold"].values),
                     labels=torch.tensor(ds.meta["target"].values), class_names=ds.class_names, categories=ds.categories)
        print(f"[esc50] {tag}: {len(ds)} clips -> {path}", flush=True)

    if not (Path(cfg.data_root) / "cifar-100-python" / "meta").exists():
        print("[cifar100] not downloaded yet; re-run prepare once the download has finished", flush=True)
        return
    for split in ("train", "test"):
        path = anchor_path(cfg.anchor_root, "cifar100", split)
        if path.exists():
            continue
        ds = CIFAR100Dataset(cfg.data_root, split=split)
        embs = chunked_encode(lambda idx: anchors.encode_images(torch.stack([ds[i]["input"] for i in idx])),
                              list(range(len(ds))), 1000, f"cifar100/{split}")
        save_anchors(path, embs, labels=torch.tensor([int(ds[i]["label"]) for i in range(len(ds))]), class_names=ds.class_names)
    print("prepare: done", flush=True)


if __name__ == "__main__":
    main()
