"""Train a common-space head on a frozen Week 4 transceiver (approach A).

    python experiments/week5_train_heads.py common=cifar10 channel=awgn
    python experiments/week5_train_heads.py common=cifar10 channel=awgn loss.cos_weight=0   # InfoNCE-only ablation

Only the head and the loss temperature train. The transceiver, the channel and the anchors are frozen;
the channel stays in the loop with SNR drawn uniformly from train.snr_range_db for every batch.
"""
import json
import math
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from semcom.common_space.adapters import make_adapter
from semcom.common_space.anchors import anchor_path, load_anchors
from semcom.common_space.heads import build_head
from semcom.common_space.learned import LearnedHeadSpace
from semcom.common_space.loader import AnchoredData, clean_symbol_sigma, load_week4, make_dataset
from semcom.common_space.losses import AnchorLoss
from semcom.common_space.retrieval import mean_cosine
from semcom.tracking.run_naming import make_run_name
from semcom.tracking.seed import set_seed
from semcom.tracking.wandb_logger import init_run


def pick_device(name: str) -> torch.device:
    return torch.device(name) if name != "auto" else torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def lr_lambda(warmup: int, total: int, min_frac: float):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress))
    return fn


@torch.no_grad()
def validate(space, loss_fn, data, w4, channel, snr_db, device, batch_size, n_batches, seed=999) -> dict[str, float]:
    space.head.eval()
    torch.manual_seed(seed)
    perm = np.random.default_rng(seed).permutation(len(data))
    sums, cos_sum, n = {"total": 0.0, "nce": 0.0, "cos": 0.0}, 0.0, 0
    for b in range(min(n_batches, len(data) // batch_size)):
        items, anchors = data.fetch(perm[b * batch_size:(b + 1) * batch_size])
        batch = w4.spec.to_device(w4.spec.collate(items), device)
        anchors = anchors.to(device)
        z = space.embed_received(batch, channel, snr_db)
        out = loss_fn(z, anchors)
        for k in sums:
            sums[k] += out[k].item()
        cos_sum += mean_cosine(z, anchors)
        n += 1
    return {**{f"val/{k}": v / n for k, v in sums.items()}, "val/cos_to_anchor": cos_sum / n}


@hydra.main(config_path="../configs", config_name="week5", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    task, channel_name = cfg.task, HydraConfig.get().runtime.choices["channel"]
    w = cfg.loss.cos_weight
    target = cfg.get("target", "raw")  # raw: Week 5 recipe, targets are the raw CLIP/CLAP anchors; centered: targets are the S anchors
    assert target in ("raw", "centered")
    base = "nce" if w == 0 else "cos" if w == 0.5 else f"cos{w:g}"  # "cos" = the approved main run
    variant = base if target == "raw" else f"cen{'' if w == 0.5 else f'{w:g}'}"
    channel = hydra.utils.instantiate(cfg.channel)

    w4 = load_week4(task, channel_name, cfg.w4_root, device)
    adapter = make_adapter(task, None, w4.spec)  # training uses cached anchors only
    ov = cfg.get("dataset_overrides")
    overrides = OmegaConf.to_container(ov) if ov is not None else {}
    train_ds, val_ds = (make_dataset(w4, s, cfg.data_root, overrides) for s in ("train", "val"))
    tr_blob, va_blob = (load_anchors(anchor_path(cfg.anchor_root, task, s)) for s in ("train", "val"))
    if target == "centered":  # train the whole objective in S: targets are the centered (and, for audio, bridged) anchors
        from semcom.common_space.domains import get_domain

        domain = get_domain(task, cfg.anchor_root, cfg.get("space_root", "outputs/week5b/space"), Path(cfg.get("bridge_root_s0", "outputs/week5b/bridge")))
        tr_blob["anchors"], va_blob["anchors"] = domain.to_S(tr_blob["anchors"]), domain.to_S(va_blob["anchors"])
    crops = (tr_blob["utt_idx"], tr_blob["starts"]) if task == "audio" else None
    train, val = AnchoredData(train_ds, tr_blob["anchors"], crops), AnchoredData(val_ds, va_blob["anchors"])

    tcfg = cfg.train
    sigma = clean_symbol_sigma(w4, adapter, train_ds, tcfg.sigma_items, device=device)
    head = build_head(cfg.head.kind, sigma=sigma, **{k: v for k, v in cfg.head.items() if k != "kind"}).to(device)
    loss_fn = AnchorLoss(**cfg.loss).to(device)
    space = LearnedHeadSpace(w4.model, head, adapter)
    params = list(head.parameters()) + list(loss_fn.parameters())
    n_params = sum(p.numel() for p in head.parameters())

    steps_per_epoch = len(train) // tcfg.batch_size
    total_steps = steps_per_epoch * tcfg.epochs
    opt = torch.optim.AdamW(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tcfg.warmup_steps, total_steps, tcfg.min_lr_frac))

    out_dir = Path(cfg.out_root) / task / channel_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = make_run_name(adapter.modality, cfg.dataset_name, channel_name, f"U{tcfg.snr_range_db[0]:g}-{tcfg.snr_range_db[1]:g}", cfg.seed)
    run = init_run(cfg, run_name, project=cfg.tracking.project, mode=cfg.tracking.mode, group=adapter.modality,
                   job_type="common-space-head", tags=["head", variant, task])
    run.define_metric("train/*", step_metric="global_step")
    run.define_metric("val/*", step_metric="epoch")
    run.summary["n_head_params"] = n_params
    run.summary["clean_symbol_sigma"] = sigma
    print(f"[{run_name}] variant={variant} device={device} head={cfg.head.kind} params={n_params / 1e6:.2f}M "
          f"sigma={sigma:.3f} train={len(train)} val={len(val)} steps={total_steps}", flush=True)

    rng = np.random.default_rng(cfg.seed)
    best, step, t0 = float("inf"), 0, time.time()
    for epoch in range(1, tcfg.epochs + 1):
        head.train()
        torch.manual_seed(cfg.seed * 1000 + epoch)
        perm = rng.permutation(len(train))
        for b in range(steps_per_epoch):
            items, anchors = train.fetch(perm[b * tcfg.batch_size:(b + 1) * tcfg.batch_size])
            batch = w4.spec.to_device(w4.spec.collate(items), device)
            snr = float(rng.uniform(*tcfg.snr_range_db))
            out = loss_fn(space.embed_received(batch, channel, snr), anchors.to(device))
            opt.zero_grad(set_to_none=True)
            out["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, tcfg.grad_clip)
            if torch.isfinite(grad_norm):
                opt.step()
            sched.step()
            step += 1
            if step % tcfg.log_every == 0:
                run.log({"global_step": step, "train/loss_total": out["total"].item(), "train/loss_nce": out["nce"].item(),
                         "train/loss_cos": out["cos"].item(), "train/temp": out["temp"].item(), "train/snr_db": snr})
        v = validate(space, loss_fn, val, w4, channel, tcfg.val_snr_db, device, tcfg.batch_size, tcfg.val_batches)
        run.log({"epoch": epoch, **v})
        print(f"  epoch {epoch:3d}/{tcfg.epochs} train_total {out['total'].item():.3f} | val@{tcfg.val_snr_db:g}dB "
              f"total {v['val/total']:.3f} nce {v['val/nce']:.3f} cos {v['val/cos']:.3f} "
              f"cos_to_anchor {v['val/cos_to_anchor']:.3f} temp {out['temp'].item():.3f} [{time.time() - t0:.0f}s]", flush=True)
        if v["val/total"] < best:
            best = v["val/total"]
            torch.save({"head": head.state_dict(), "head_cfg": OmegaConf.to_container(cfg.head), "sigma": sigma, "target": target,
                        "loss": loss_fn.state_dict(), "epoch": epoch, "best_val_total": best,
                        "cfg": OmegaConf.to_container(cfg, resolve=True)}, out_dir / f"head_{variant}.pt")

    (out_dir / f"train_{variant}.json").write_text(json.dumps(
        {"task": task, "channel": channel_name, "variant": variant, "target": target, "run_name": run_name, "n_head_params": n_params,
         "sigma": sigma, "best_val_total": best, "train_seconds": time.time() - t0, "final_temp": out["temp"].item()}, indent=2))
    run.finish()
    print(f"Done: best val total {best:.3f}; head saved to {out_dir / f'head_{variant}.pt'}")


if __name__ == "__main__":
    main()
