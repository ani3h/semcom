"""Stage 2: pre-build the clean-symbol caches (train and val) of every task and channel, once, before parallel head training.

    python experiments/week5b_s2_build_caches.py 'tasks=[caption_text,text,audio]'
"""
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from semcom.common_space.adapters import make_adapter
from semcom.common_space.anchors import anchor_path, load_anchors
from semcom.common_space.loader import AnchoredData, load_week4, make_dataset
from semcom.common_space.training import SymbolCache

ALL = ["cifar10", "mnist", "esc50_audio", "audio", "text", "caption_text"]


@hydra.main(config_path="../configs", config_name="week5b_s2", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for task in list(cfg.get("tasks", ALL)):
        common = OmegaConf.load(f"configs/common_s2/{task}.yaml")
        overrides = OmegaConf.to_container(common.dataset_overrides)
        for ch in ("awgn", "rayleigh"):
            w4 = load_week4(task, ch, cfg.w4_root, device)
            adapter = make_adapter(task, None, w4.spec)
            for split in ("train", "val"):
                t0 = time.time()
                ds = make_dataset(w4, split, cfg.data_root, overrides)
                blob = load_anchors(anchor_path(cfg.anchor_root, task, split))
                crops = (blob["utt_idx"], blob["starts"]) if (task == "audio" and split == "train") else None
                data = AnchoredData(ds, blob["anchors"], crops)
                path = Path(cfg.symcache_root) / f"{task}_{ch}_{split}.pt"
                c = SymbolCache.load_or_build(path, w4, adapter, data, device)
                print(f"[cache] {task}/{ch}/{split}: {tuple(c.sym.shape)} -> {path.name} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
