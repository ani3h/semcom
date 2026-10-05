"""Train one modality's transceiver with the channel in the loop, then sweep SNR.

    python experiments/week4_train.py task=cifar10 channel=awgn
    python experiments/week4_train.py task=text channel=rayleigh tracking.mode=offline

One model per (task, channel); SNR is drawn uniformly from train.snr_range_db
for every batch and the test metric is evaluated at each point of eval.snr_grid_db.
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
from torch.utils.data import DataLoader

from semcom.tracking.plots import plot_snr_curves
from semcom.tracking.run_naming import make_run_name
from semcom.tracking.seed import set_seed
from semcom.tracking.wandb_logger import init_run


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def lr_lambda(warmup: int, total: int, min_frac: float):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress))
    return fn


@torch.no_grad()
def val_loss(spec, model, loader, channel, snr_db, device, max_batches=20) -> float:
    model.eval()
    torch.manual_seed(999)
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        losses.append(spec.loss(model, spec.to_device(batch, device), channel, snr_db).item())
    return float(np.mean(losses))


@hydra.main(config_path="../configs", config_name="week4", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    choices = HydraConfig.get().runtime.choices
    task_name, channel_name = choices["task"], choices["channel"]

    spec = hydra.utils.instantiate(cfg.spec, _recursive_=False)
    channel = hydra.utils.instantiate(cfg.channel)
    train_ds, val_ds, test_ds = (hydra.utils.instantiate(cfg.dataset, split=s) for s in ("train", "val", "test"))
    spec.fit(train_ds)
    model = spec.build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    tcfg = cfg.train
    loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, drop_last=True,
                        collate_fn=spec.collate, num_workers=tcfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, collate_fn=spec.collate)
    steps_per_epoch = len(loader) if tcfg.get("max_steps_per_epoch") is None else min(len(loader), tcfg.max_steps_per_epoch)
    total_steps = steps_per_epoch * tcfg.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tcfg.warmup_steps, total_steps, tcfg.min_lr_frac))

    out_dir = Path(cfg.out_root) / task_name / channel_name
    out_dir.mkdir(parents=True, exist_ok=True)
    snr_label = f"U{tcfg.snr_range_db[0]:g}-{tcfg.snr_range_db[1]:g}"
    run_name = make_run_name(spec.modality, choices["dataset"], channel_name, snr_label, cfg.seed)
    run = init_run(cfg, run_name, project=cfg.tracking.project, mode=cfg.tracking.mode,
                   group=spec.modality, job_type="train")
    run.define_metric("train/*", step_metric="global_step")
    run.define_metric("val/*", step_metric="epoch")
    run.define_metric("eval/*", step_metric="snr_db")
    run.summary["n_params"] = n_params
    print(f"[{run_name}] device={device} params={n_params / 1e6:.2f}M steps={total_steps} "
          f"({steps_per_epoch}/epoch x {tcfg.epochs})", flush=True)

    rng = np.random.default_rng(cfg.seed)
    best_val, step, t0 = float("inf"), 0, time.time()
    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        running = []
        for i, batch in enumerate(loader):
            if i >= steps_per_epoch:
                break
            snr = float(rng.uniform(*tcfg.snr_range_db))
            loss = spec.loss(model, spec.to_device(batch, device), channel, snr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            if torch.isfinite(grad_norm):  # Rayleigh deep fades can produce a non-finite step
                opt.step()
            sched.step()
            step += 1
            running.append(loss.item())
            if step % tcfg.log_every == 0:
                run.log({"global_step": step, "train/loss": float(np.mean(running[-tcfg.log_every:])),
                         "train/snr_db": snr, "train/lr": sched.get_last_lr()[0]})
        v = val_loss(spec, model, val_loader, channel, tcfg.val_snr_db, device)
        torch.manual_seed(cfg.seed * 1000 + epoch)  # val_loss reseeds the global RNG; give training a fresh stream
        run.log({"epoch": epoch, "val/loss": v})
        print(f"  epoch {epoch:3d}/{tcfg.epochs} train {np.mean(running):.4f} val@{tcfg.val_snr_db:g}dB {v:.4f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
        if v < best_val:
            best_val = v
            torch.save({"model": model.state_dict(), "spec": spec.state_dict(),
                        "cfg": OmegaConf.to_container(cfg, resolve=True)}, out_dir / "best.pt")

    if (out_dir / "best.pt").exists():
        model.load_state_dict(torch.load(out_dir / "best.pt", weights_only=False)["model"])
    results: dict[float, dict[str, float]] = {}
    for snr in cfg.eval.snr_grid_db:
        res = spec.evaluate(model, test_ds, channel, float(snr), device)
        results[float(snr)] = res
        run.log({"snr_db": float(snr), **{f"eval/{k}": v for k, v in res.items()}})
        print(f"  test @ {snr:>4} dB: " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)

    (out_dir / "results.json").write_text(json.dumps(
        {"task": task_name, "channel": channel_name, "run_name": run_name, "metrics": spec.metric_names,
         "n_params": n_params, "train_seconds": time.time() - t0, "results": {str(k): v for k, v in results.items()}},
        indent=2))
    fig = plot_snr_curves({channel_name: results}, spec.metric_names, f"{task_name} ({channel_name})",
                          str(out_dir / "snr_curve.png"))
    run.log({"eval/snr_curve": wandb.Image(fig)})
    run.finish()
    print(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
