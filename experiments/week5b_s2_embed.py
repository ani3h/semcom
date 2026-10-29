"""Stage 2: build the embedding caches (items x 5 noise draws x SNR grid, in S) that every Stage 2 metric is computed from.

    python experiments/week5b_s2_embed.py kind=B 'tasks=[cifar10,mnist]' 'splits=[test]'      # pretrained encoder on the decoded output
    python experiments/week5b_s2_embed.py kind=A                                              # heads, every variant and seed
    python experiments/week5b_s2_embed.py kind=promptsA | kind=promptsB                       # class prompts transmitted by the caption-text transceiver

A and B use the same channel noise (see semcom.common_space.stage2), so a given item sees identical realisations under every method.
Output: outputs/week5b/stage2/emb/<kind>_<task>_<channel>_<split>[_<variant>_s<seed>].pt
"""
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from semcom.common_space.adapters import make_adapter
from semcom.common_space.anchors import PretrainedAnchors, anchor_path, load_anchors
from semcom.common_space.domains import get_domain
from semcom.common_space.loader import load_head_space, load_week4, make_dataset
from semcom.common_space.stage2 import CHUNK, DRAWS, EVAL_ITEMS, SNR_GRID, clean_symbols, embed_A, embed_B
from semcom.data.prompts import HELDOUT_TEMPLATES, TRAINING_TEMPLATES, template_prompts


def pick_device(name: str) -> torch.device:
    return torch.device(name) if name != "auto" else torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def prompt_items(root) -> list[dict]:
    """Every class prompt that the caption-text transceiver will transmit: training-time templates (a) and held-out paraphrases (b)."""
    rows = []
    for task in ("cifar10", "mnist", "esc50"):
        for set_name, templates in (("train", TRAINING_TEMPLATES), ("heldout", HELDOUT_TEMPLATES)):
            for p in template_prompts(task, templates, root):
                rows.append({"input": p["text"], "label": None, "raw": None, "task": task, "set": set_name, "class_idx": p["class_idx"], "template": p["template"]})
    return rows


def save(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)


@hydra.main(config_path="../configs", config_name="week5b_s2_eval", version_base=None)
def main(cfg: DictConfig) -> None:
    device = pick_device(cfg.device)
    emb = Path(cfg.out_root) / "emb"
    kind = cfg.kind
    pre = PretrainedAnchors(device=device) if kind in ("B", "promptsB") else None
    tasks = ["caption_text"] if kind.startswith("prompts") else list(cfg.tasks)
    splits = ["prompts"] if kind.startswith("prompts") else list(cfg.splits)
    for task in tasks:
        domain = get_domain(task, cfg.anchor_root, cfg.space_root, cfg.bridge_dir)
        for ch in cfg.channels:
            w4 = load_week4(task, ch, cfg.w4_root, device)
            adapter = make_adapter(task, pre, w4.spec)
            if kind in ("B", "promptsB") and not adapter.supports_direct:
                print(f"[embed] {task}: approach B not applicable (decode() returns class logits)", flush=True)
                continue
            channel = hydra.utils.instantiate(OmegaConf.load(f"configs/channel/{ch}.yaml"))
            for split in splits:
                t0 = time.time()
                if split == "prompts":
                    items, meta, chunk = prompt_items(cfg.data_root), None, CHUNK["prompts"]
                    meta = [{k: v for k, v in it.items() if k not in ("label", "raw")} for it in items]
                else:
                    ds = make_dataset(w4, split, cfg.data_root, {"prompt_repeat": 0} if task == "caption_text" else None)
                    n = min(len(ds), EVAL_ITEMS[split][task])
                    items, meta, chunk = [ds[i] for i in range(n)], None, CHUNK[task]
                chunks = clean_symbols(w4, adapter, items, device, chunk)
                if kind in ("A", "promptsA"):
                    a_raw = None if split == "prompts" else load_anchors(anchor_path(cfg.anchor_root, task, split))["anchors"][:len(items)]
                    for variant in cfg.variants:
                        for seed in cfg.seeds:
                            path = Path(cfg.heads_root) / task / ch / f"{variant}_s{seed}.pt"
                            name = f"{kind}_{task}_{ch}_{split}_{variant}_s{seed}.pt"
                            if (emb / name).exists():
                                continue
                            if not (Path(cfg.heads_root) / task / ch / f"train_{variant}_s{seed}.json").exists():
                                print(f"[embed] head not finished yet: {path.name}", flush=True)
                                continue
                            space, hmeta = load_head_space(path, w4, adapter, device)
                            res = embed_A(space.head, domain, hmeta["target"], chunks, channel, a_raw if hmeta["target"] == "raw" else None,
                                          base_seed=cfg.noise_seed)
                            save(emb / name, {**res, "snr": SNR_GRID, "draws": DRAWS, "meta": meta, "head": {"variant": variant, "seed": seed, "target": hmeta["target"], "epoch": hmeta["epoch"]}})
                            print(f"[embed] {name} [{time.time() - t0:.0f}s]", flush=True)
                else:
                    if (emb / f"{kind}_{task}_{ch}_{split}.pt").exists():
                        print(f"[embed] skip existing {kind}_{task}_{ch}_{split}.pt", flush=True)
                        continue
                    res = embed_B(w4, adapter, domain, chunks, channel, base_seed=cfg.noise_seed, audio_workers=int(cfg.audio_workers),
                                  progress=lambda d, snr: print(f"[embed]   {task}/{ch}/{split} draw {d} snr {snr:g} [{time.time() - t0:.0f}s]", flush=True))
                    name = f"{kind}_{task}_{ch}_{split}.pt"
                    save(emb / name, {**res, "snr": SNR_GRID, "draws": DRAWS, "meta": meta})
                    print(f"[embed] {name} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
