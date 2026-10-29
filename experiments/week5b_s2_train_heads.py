"""Stage 2: train one head (one variant, one seed) on a frozen transceiver.

    python experiments/week5b_s2_train_heads.py common_s2=caption_text channel=awgn variant=cen seed=43

Variants: cos5 = raw CLIP/CLAP anchors with cosine weight 5; cen = centered S anchors with cosine weight 0.5 (Stage 1 definitions).
Text and speech heads use up to 80 epochs with a plateau learning-rate decay and early stopping (patience 8); image heads keep the
40-epoch cosine recipe; ESC-50 heads keep 100 epochs cosine and take dropout / weight decay overrides. Only the head and the loss
temperature train: the transceiver, channel, anchors and (centered) S targets are frozen. Output: <out_root>/<task>/<channel>/<variant>_s<seed><tag>.pt
"""
import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from semcom.common_space.adapters import make_adapter
from semcom.common_space.anchors import anchor_path, load_anchors
from semcom.common_space.domains import get_domain
from semcom.common_space.heads import build_head
from semcom.common_space.learned import LearnedHeadSpace
from semcom.common_space.loader import AnchoredData, clean_symbol_sigma, load_week4, make_dataset
from semcom.common_space.losses import AnchorLoss
from semcom.common_space.training import (EarlyStopper, LRSchedule, SymbolCache, chunk_bounds, chunk_widths, chunked_head_forward, clip_and_guard,
                                          sort_within_batches, validate)
from semcom.tracking.run_naming import make_run_name
from semcom.tracking.seed import set_seed
from semcom.tracking.wandb_logger import init_run

VARIANTS = {"cos5": ("raw", 5.0), "cen": ("centered", 0.5)}


def pick_device(name: str) -> torch.device:
    return torch.device(name) if name != "auto" else torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(config_path="../configs", config_name="week5b_s2", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    task, channel_name, variant, tag = cfg.task, HydraConfig.get().runtime.choices["channel"], cfg.variant, cfg.tag
    target, cos_weight = VARIANTS[variant]
    channel = hydra.utils.instantiate(cfg.channel)
    tcfg = cfg.train

    w4 = load_week4(task, channel_name, cfg.w4_root, device)
    adapter = make_adapter(task, None, w4.spec)
    overrides = OmegaConf.to_container(cfg.dataset_overrides) if cfg.get("dataset_overrides") is not None else {}
    train_ds, val_ds = (make_dataset(w4, s, cfg.data_root, overrides) for s in ("train", "val"))
    tr_blob, va_blob = (load_anchors(anchor_path(cfg.anchor_root, task, s)) for s in ("train", "val"))
    if target == "centered":
        domain = get_domain(task, cfg.anchor_root, cfg.space_root, cfg.bridge_dir)
        tr_blob["anchors"], va_blob["anchors"] = domain.to_S(tr_blob["anchors"]), domain.to_S(va_blob["anchors"])
    crops = (tr_blob["utt_idx"], tr_blob["starts"]) if task == "audio" else None  # Valentini speech trains on fixed, cached crops
    train, val = AnchoredData(train_ds, tr_blob["anchors"], crops), AnchoredData(val_ds, va_blob["anchors"])

    sigma = clean_symbol_sigma(w4, adapter, train_ds, tcfg.sigma_items, device=device)
    head = build_head(cfg.head.kind, sigma=sigma, **{k: v for k, v in cfg.head.items() if k != "kind"}).to(device)
    loss_fn = AnchorLoss(cos_weight=cos_weight, **cfg.loss).to(device)
    params = list(head.parameters()) + list(loss_fn.parameters())
    n_params = sum(p.numel() for p in head.parameters())

    sym = Path(cfg.symcache_root)
    tr_cache = SymbolCache.load_or_build(sym / f"{task}_{channel_name}_train.pt", w4, adapter, train, device).to(device)
    va_cache = SymbolCache.load_or_build(sym / f"{task}_{channel_name}_val.pt", w4, adapter, val, device).to(device)
    tr_anchors = tr_blob["anchors"].to(device)  # resident on the device: per-step indexing needs no host transfer

    steps_per_epoch = len(train) // tcfg.batch_size
    opt = torch.optim.AdamW(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = LRSchedule(opt, tcfg.lr, tcfg.warmup_steps, tcfg.schedule, steps_per_epoch * tcfg.epochs, tcfg.min_lr_frac,
                       tcfg.get("lr_factor", 0.3), tcfg.get("lr_patience", 3))
    stopper = EarlyStopper(tcfg.patience)

    out_dir = Path(cfg.out_root) / task / channel_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{variant}_s{cfg.seed}{tag}"
    run_name = make_run_name(adapter.modality, cfg.dataset_name, channel_name, f"U{tcfg.snr_range_db[0]:g}-{tcfg.snr_range_db[1]:g}", cfg.seed)
    run = init_run(cfg, run_name, project=cfg.tracking.project, mode=cfg.tracking.mode, group=adapter.modality,
                   job_type="common-space-head-s2", tags=["head", "stage2", variant, task, f"seed{cfg.seed}"])
    run.define_metric("train/*", step_metric="global_step")
    run.define_metric("val/*", step_metric="epoch")
    print(f"[{run_name}] stage2 {stem} device={device} head={cfg.head.kind} dropout={cfg.head.dropout} wd={tcfg.weight_decay} params={n_params / 1e6:.2f}M "
          f"sigma={sigma:.3f} train={len(train)} val={len(val)} schedule={tcfg.schedule} max_epochs={tcfg.epochs} patience={tcfg.patience}", flush=True)

    rng = np.random.default_rng(cfg.seed)
    step, t0, history, stopped_early, start_epoch = 0, time.time(), [], False, 1
    resume_path = out_dir / f"{stem}.resume.pt"
    if resume_path.exists():  # continue an interrupted run from its last completed epoch (identical continuation: all RNG state is restored)
        rs = torch.load(resume_path, map_location=device, weights_only=False)
        head.load_state_dict(rs["head"])
        loss_fn.load_state_dict(rs["loss"])
        opt.load_state_dict(rs["opt"])
        sched.t, sched.scale, sched.bad_epochs = rs["sched"]
        sched._apply()
        stopper.best, stopper.best_epoch, stopper.bad = rs["stopper"]
        history, step, start_epoch = rs["history"], rs["step"], rs["epoch"] + 1
        rng.bit_generator.state = rs["rng"]
        t0 = time.time() - rs["elapsed"]
        print(f"  resumed {stem} after epoch {rs['epoch']} (best val {stopper.best:.3f} at epoch {stopper.best_epoch})", flush=True)
    for epoch in range(start_epoch, tcfg.epochs + 1):
        head.train()
        torch.manual_seed(cfg.seed * 1000 + epoch)
        perm = rng.permutation(len(train))
        k = int(tcfg.get("length_chunks", 1)) if tr_cache.lengths_np is not None else 1
        if k > 1:  # text heads: reorder inside each batch by length and run the head on k length-sorted chunks (same batches, far less padding)
            perm = sort_within_batches(perm, tr_cache.lengths_np, tcfg.batch_size, steps_per_epoch)
            bounds = chunk_bounds(tcfg.batch_size, k)
            cwidths = chunk_widths(perm, tr_cache.lengths_np, tcfg.batch_size, steps_per_epoch, k, cap=tr_cache.sym.size(1))
        perm_dev = torch.as_tensor(perm, dtype=torch.long, device=device)  # one host->device transfer per epoch
        widths = tr_cache.widths(perm, tcfg.batch_size, steps_per_epoch)
        for b in range(steps_per_epoch):
            idx = perm_dev[b * tcfg.batch_size:(b + 1) * tcfg.batch_size]
            snr = float(rng.uniform(*tcfg.snr_range_db))
            if k > 1:
                z = chunked_head_forward(head, channel, snr, tr_cache.sym, tr_cache.lengths, idx, bounds, cwidths[b])
            else:
                rx, side = tr_cache.get_dev(idx, None if widths is None else widths[b])
                z = head(channel(rx, snr), **side)
            out = loss_fn(z, tr_anchors[idx])
            opt.zero_grad(set_to_none=True)
            out["total"].backward()
            clip_and_guard(params, tcfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            if step % tcfg.log_every == 0:
                run.log({"global_step": step, "train/loss_total": out["total"].item(), "train/loss_nce": out["nce"].item(),
                         "train/loss_cos": out["cos"].item(), "train/temp": out["temp"].item(), "train/snr_db": snr, "train/lr": sched.lr})
        v = validate(head, loss_fn, va_blob["anchors"], channel, tcfg.val_snr_db, device, tcfg.batch_size, tcfg.val_batches, va_cache)
        improved, stop = stopper.update(v["val/total"], epoch)
        sched.epoch_end(improved)
        history.append({"epoch": epoch, "val_total": v["val/total"], "val_cos": v["val/cos_to_anchor"], "lr": sched.lr, "train_total": out["total"].item()})
        run.log({"epoch": epoch, **v})
        print(f"  epoch {epoch:3d}/{tcfg.epochs} train {out['total'].item():.3f} | val@{tcfg.val_snr_db:g}dB total {v['val/total']:.3f} nce {v['val/nce']:.3f} "
              f"cos_to_anchor {v['val/cos_to_anchor']:.3f} lr {sched.lr:.2e} {'*' if improved else ''}[{time.time() - t0:.0f}s]", flush=True)
        if improved:
            torch.save({"head": head.state_dict(), "head_cfg": OmegaConf.to_container(cfg.head), "sigma": sigma, "target": target, "variant": variant,
                        "seed": cfg.seed, "loss": loss_fn.state_dict(), "epoch": epoch, "best_val_total": v["val/total"],
                        "cfg": OmegaConf.to_container(cfg, resolve=True)}, out_dir / f"{stem}.pt")
        tmp = resume_path.with_suffix(".tmp")
        torch.save({"epoch": epoch, "head": head.state_dict(), "loss": loss_fn.state_dict(), "opt": opt.state_dict(), "sched": (sched.t, sched.scale, sched.bad_epochs),
                    "stopper": (stopper.best, stopper.best_epoch, stopper.bad), "history": history, "step": step, "rng": rng.bit_generator.state,
                    "elapsed": time.time() - t0}, tmp)
        tmp.replace(resume_path)
        if stop:
            stopped_early = True
            break

    (out_dir / f"train_{stem}.json").write_text(json.dumps(
        {"task": task, "channel": channel_name, "variant": variant, "target": target, "seed": int(cfg.seed), "tag": tag, "n_head_params": n_params,
         "dropout": cfg.head.dropout, "weight_decay": tcfg.weight_decay, "schedule": tcfg.schedule, "max_epochs": tcfg.epochs, "patience": tcfg.patience,
         "best_val_total": stopper.best, "best_epoch": stopper.best_epoch, "epochs_run": len(history), "stopped_early": stopped_early,
         "train_seconds": time.time() - t0, "history": history}, indent=2))
    resume_path.unlink(missing_ok=True)
    run.finish()
    print(f"Done {stem}: best val {stopper.best:.3f} at epoch {stopper.best_epoch}/{len(history)} (stopped early: {stopped_early})", flush=True)


if __name__ == "__main__":
    main()
