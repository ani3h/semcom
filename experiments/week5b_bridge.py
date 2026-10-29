"""Stage 0, step 2: train the CLAP-text -> CLIP-text bridge (linear and MLP) on caption text only and report
text-to-text agreement on held-out captions.

    python experiments/week5b_bridge.py

The linear-vs-MLP choice is made per seed on the caption VALIDATION split by a rule fixed in the config
(bridge.select_margin); the test captions and ESC-50 are never used for that choice.
"""
import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from semcom.common_space.bridge import agreement, ridge_fit, train_bridge
from semcom.common_space.captions import load_corpus
from semcom.common_space.correction import Centerer
from semcom.tracking.run_naming import make_run_name
from semcom.tracking.seed import set_seed
from semcom.tracking.wandb_logger import init_run


def pick_device(name: str) -> torch.device:
    return torch.device(name) if name != "auto" else torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(config_path="../configs", config_name="week5b", version_base=None)
def main(cfg: DictConfig) -> None:
    device = pick_device(cfg.device)
    root, out = Path(cfg.bridge_root), Path(cfg.out_root) / "bridge"
    out.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(Path(cfg.captions.root) / "corpus.json")
    emb = {s: {n: torch.load(root / f"{s}_{n}.pt").float() for n in ("clap_text", "clip_text")} for s in ("train", "val", "test")}

    cx = Centerer().fit(emb["train"]["clap_text"])  # statistics from TRAIN captions only
    cy = Centerer().fit(emb["train"]["clip_text"])
    cx.save(out / "centerer_clap_text.pt")
    cy.save(out / "centerer_clip_text.pt")
    X = {s: cx.transform(emb[s]["clap_text"]) for s in emb}
    Y = {s: cy.transform(emb[s]["clip_text"]) for s in emb}
    sources = {s: np.array([r["source"] for r in corpus[s]]) for s in corpus}
    print(f"[bridge] captions train/val/test = {len(X['train'])}/{len(X['val'])}/{len(X['test'])}; "
          f"random-pair-free check: mean cos of centered CLAP vs CLIP text (unmapped) = {(X['test'] * Y['test']).sum(-1).mean():.3f}", flush=True)

    b = cfg.bridge
    run = init_run(cfg, make_run_name("audio", "captions", "bridge", "none", cfg.seed), project=cfg.tracking.project,
                   mode=cfg.tracking.mode, group="audio", job_type="bridge")
    results = {"n": {s: len(X[s]) for s in X}, "gallery": b.gallery, "chance_recall@1": 1 / b.gallery, "seeds": {}}

    W = ridge_fit(X["train"], Y["train"] - Y["train"].mean(0), lam=1e-2)
    ridge_out = F.normalize(X["test"] @ W, dim=-1)
    results["ridge_reference"] = {"test": agreement(ridge_out, Y["test"], b.gallery)}
    print(f"[bridge] ridge regression reference on test: {results['ridge_reference']['test']}", flush=True)

    for seed in cfg.seeds:
        set_seed(seed)
        per_kind = {}
        for kind in b.kinds:
            model, hist = train_bridge(X["train"], Y["train"], X["val"], Y["val"], kind=kind, hidden=b.hidden, epochs=b.epochs,
                                       batch_size=b.batch_size, lr=b.lr, weight_decay=b.weight_decay, cos_weight=b.cos_weight,
                                       init_temp=b.init_temp, seed=seed, device=device, gallery=b.gallery,
                                       log=lambda m: print(m, flush=True))
            with torch.no_grad():
                pred = {s: model(X[s]) for s in ("val", "test")}
            res = {"val": agreement(pred["val"], Y["val"], b.gallery), "test": agreement(pred["test"], Y["test"], b.gallery),
                   "test_by_source": {src: agreement(pred["test"][sources["test"] == src], Y["test"][sources["test"] == src], b.gallery)
                                      for src in ("coco", "audiocaps")},
                   "n_params": sum(p.numel() for p in model.parameters())}
            per_kind[kind] = res
            torch.save({"state": model.state_dict(), "kind": kind, "hidden": b.hidden, "seed": seed}, out / f"{kind}_s{seed}.pt")
            run.log({f"{kind}/seed{seed}/test_recall@1": res["test"]["recall@1"], f"{kind}/seed{seed}/test_cos": res["test"]["cos"]})
        chosen = "mlp" if ("mlp" in per_kind and "linear" in per_kind and
                           per_kind["mlp"]["val"]["recall@1"] >= per_kind["linear"]["val"]["recall@1"] + b.select_margin) else "linear"
        results["seeds"][str(seed)] = {"kinds": per_kind, "chosen": chosen}
        print(f"[bridge] seed {seed}: val R@1 linear {per_kind['linear']['val']['recall@1']:.3f} vs mlp "
              f"{per_kind['mlp']['val']['recall@1']:.3f} -> chosen {chosen}; test R@1 {per_kind[chosen]['test']['recall@1']:.3f}", flush=True)

    (out / "results.json").write_text(json.dumps({**results, "config": OmegaConf.to_container(cfg.bridge)}, indent=2))
    run.finish()
    print(f"[bridge] done -> {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
