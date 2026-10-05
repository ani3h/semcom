"""Combine per-channel results into one task-metric-vs-SNR figure per task
(AWGN and Rayleigh as separate curves), SeER Fig. 4 style.

    python experiments/week4_plot.py                 # all tasks found under outputs/week4
    python experiments/week4_plot.py --wandb         # also log the figures to a W&B summary run
"""
import argparse
import json
from pathlib import Path

from semcom.tracking.plots import plot_snr_curves

TITLES = {
    "text": "Text (Europarl, DeepSC-style Transformer)",
    "cifar10": "CIFAR-10 reconstruction (Deep JSCC, 1/6)",
    "mnist": "MNIST classification (task-oriented JSCC)",
    "audio": "Speech reconstruction (Edinburgh, SE-ResNet)",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/week4")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    figures = {}
    for task_dir in sorted(p for p in Path(args.root).iterdir() if p.is_dir() and p.name != "logs"):
        by_channel, metrics = {}, None
        for res_file in sorted(task_dir.glob("*/results.json")):
            data = json.loads(res_file.read_text())
            by_channel[data["channel"]] = {float(s): m for s, m in data["results"].items()}
            metrics = [m for m in data["metrics"] if any(m in v for v in by_channel[data["channel"]].values())]
        if not by_channel:
            continue
        out = task_dir / "snr_curves.png"
        figures[task_dir.name] = plot_snr_curves(by_channel, metrics, TITLES.get(task_dir.name, task_dir.name), str(out))
        print(f"{task_dir.name}: {sorted(by_channel)} -> {out}")

    if args.wandb and figures:
        import wandb

        run = wandb.init(project="semcom", name="week4-snr-curves", group="summary", job_type="plot", mode="offline")
        run.log({f"snr_curves/{name}": wandb.Image(fig) for name, fig in figures.items()})
        run.finish()


if __name__ == "__main__":
    main()
