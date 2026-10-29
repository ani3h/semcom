"""Stage 1, step 1: cache the raw anchors of the two new tasks, fit and cache every domain's centering on TRAIN data only,
and cache S anchors for all six tasks. Reports the random-pair cosine of every anchor space before and after correction.

    python experiments/week5b_stage1_space.py

Nothing here touches ESC-50 fold 5 for any fitted quantity (asserted); the folds appear only as the split labels of the
cached anchors. Writes outputs/week5b/space/{centerer_<task>.pt, anchors_S/<task>_<split>.pt, results.json}.
"""
import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from semcom.common_space.anchors import anchor_path, load_anchors, save_anchors
from semcom.common_space.captions import load_corpus
from semcom.common_space.domains import DOMAIN_SPECS, esc50_train_anchors, get_domain
from semcom.common_space.retrieval import random_pair_cosine
from semcom.data.audio_esc50 import ESC50Dataset
from semcom.data.text_captions import CaptionDataset

SPLITS = ("train", "val", "test")


def cache_caption_anchors(cfg) -> dict:
    """Raw CLIP-text anchors of each CaptionDataset split, row-aligned with the dataset (prompts excluded)."""
    out = {}
    for split in SPLITS:
        ds = CaptionDataset(cfg.data_root, split=split, seed=cfg.seed, prompt_repeat=0)
        raw = torch.load(Path(cfg.bridge_root) / f"{ds.corpus_split}_clip_text.pt").float()[torch.from_numpy(ds.corpus_indices)]
        assert raw.size(0) == len(ds)
        save_anchors(anchor_path(cfg.anchor_root, "caption_text", split), raw, n=len(ds))
        out[split] = len(ds)
    return out


def cache_esc50_anchors(cfg) -> dict:
    """Raw CLAP anchors of each ESC-50 split (train = folds 1-3, val = fold 4, test = fold 5), aligned with ESC50Dataset."""
    blob = load_anchors(anchor_path(cfg.anchor_root, "esc50", "all", "_clap_16k"))
    out = {}
    for split, folds in (("train", (1, 2, 3)), ("val", (4,)), ("test", (5,))):
        ds = ESC50Dataset(cfg.data_root, split=split, download=False)
        sel = torch.isin(blob["folds"], torch.tensor(folds))
        names = [n for n, s in zip(blob["filenames"], sel.tolist()) if s]
        assert names == list(ds.meta["filename"]), f"ESC-50 {split}: anchor rows are not aligned with the dataset"
        save_anchors(anchor_path(cfg.anchor_root, "esc50_audio", split), blob["anchors"][sel], filenames=names, n=len(ds))
        out[split] = len(ds)
    train_files = set(load_anchors(anchor_path(cfg.anchor_root, "esc50_audio", "train"))["filenames"])
    val_files = set(load_anchors(anchor_path(cfg.anchor_root, "esc50_audio", "val"))["filenames"])
    test_files = set(load_anchors(anchor_path(cfg.anchor_root, "esc50_audio", "test"))["filenames"])
    assert not (train_files & test_files) and not (val_files & test_files), "fold 5 leaked into train/val"
    out["fold5_guard"] = "train and val anchors share no file with the test fold"
    return out


@hydra.main(config_path="../configs", config_name="week5b", version_base=None)
def main(cfg: DictConfig) -> None:
    space = Path(cfg.out_root) / "space"
    (space / "anchors_S").mkdir(parents=True, exist_ok=True)
    report = {"new_anchor_caches": {"caption_text": cache_caption_anchors(cfg), "esc50_audio": cache_esc50_anchors(cfg)}, "domains": {}}
    esc50_train_anchors(cfg.anchor_root)  # asserts the fit set is exactly folds 1-3
    for task, spec in DOMAIN_SPECS.items():
        domain = get_domain(task, cfg.anchor_root, space, Path(cfg.out_root) / "bridge")
        raw = {s: load_anchors(anchor_path(cfg.anchor_root, task, s))["anchors"] for s in SPLITS}
        S = {s: domain.to_S(raw[s]) for s in SPLITS}
        for s in SPLITS:
            torch.save(S[s].half(), space / "anchors_S" / f"{task}_{s}.pt")
        row = {"fit": domain.fit_on, "n": {s: int(raw[s].size(0)) for s in SPLITS},
               "random_pair_cosine": {"raw_test": random_pair_cosine(raw["test"]), "S_test": random_pair_cosine(S["test"]),
                                      "raw_val": random_pair_cosine(raw["val"]), "S_val": random_pair_cosine(S["val"])}}
        report["domains"][task] = row
        print(f"[space] {task:13s} fit {row['fit']['fit']} (n={row['fit']['n_fit']}, bridge {row['fit']['bridge']}) | random-pair cosine "
              f"test: raw {row['random_pair_cosine']['raw_test']:.3f} -> S {row['random_pair_cosine']['S_test']:.3f}", flush=True)
    (space / "results.json").write_text(json.dumps(report, indent=2))
    (space / "manifest.json").write_text(json.dumps({t: d["fit"] for t, d in report["domains"].items()}, indent=2))
    print(f"[space] wrote {space}", flush=True)


if __name__ == "__main__":
    main()
