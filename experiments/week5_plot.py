"""Retrieval-vs-SNR figures for the common space, in the style of the Week 4 SNR curves.

    python experiments/week5_plot.py            # one figure per task under outputs/week5/<task>/
    python experiments/week5_plot.py --wandb    # also log them to an offline W&B summary run
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = {  # key -> (label, colour)
    "A_cos": ("A: head, InfoNCE + 0.5 cos", "#1d6fb8"),
    "A_nce": ("A: head, InfoNCE only", "#7fb2e0"),
    "A_cos5": ("A: head, InfoNCE + 5 cos (ablation)", "#e08a1d"),
    "B": ("B: pretrained on decoded output", "#2a9d4b"),
    "ceiling": ("clean-input ceiling", "#6b7280"),
}
CHANNEL_STYLE = {"awgn": ("AWGN", "o-"), "rayleigh": ("Rayleigh", "s--")}
PANELS = {  # task -> [(metric, title, chance level or None)]
    "cifar10": [("top1", "CIFAR-10 class retrieval, top-1", 0.1), ("top5", "top-5", 0.5), ("cos_to_anchor", "cosine to own anchor", None), ("margin", "margin: own minus unrelated cosine", None)],
    "mnist": [("top1", "MNIST class retrieval, top-1", 0.1), ("top5", "top-5", 0.5), ("cos_to_anchor", "cosine to own anchor", None), ("margin", "margin: own minus unrelated cosine", None)],
    "text": [("recall@1", "Text self-retrieval, recall@1", None), ("recall@5", "recall@5", None), ("cos_to_anchor", "cosine to own anchor", None), ("margin", "margin: own minus unrelated cosine", None)],
    "audio": [("recall@1", "Audio clean<->transmitted, recall@1", None), ("recall@5", "recall@5", None), ("cos_to_anchor", "cosine to own anchor", None), ("margin", "margin: own minus unrelated cosine", None)],
}


def plot_task(task: str, data: dict[str, dict], save_path: str):
    panels = PANELS[task]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.2), squeeze=False)
    n_eval = next(iter(data.values()))["n_eval"]
    for ax, (metric, title, chance) in zip(axes[0], panels):
        for channel, res in data.items():
            ch_label, style = CHANNEL_STYLE[channel]
            for method, (label, color) in METHODS.items():
                if method not in res["results"]:
                    continue
                by_snr = res["results"][method]
                snrs = sorted(by_snr, key=float)
                ys = [by_snr[s][metric] for s in snrs]
                if method == "ceiling":
                    if channel == "awgn":  # SNR-independent: draw once
                        ax.axhline(ys[0], color=color, ls=":", lw=1.4, label=label)
                    continue
                ax.plot([float(s) for s in snrs], ys, style, color=color, markersize=4.5, lw=1.5,
                        label=f"{label} [{ch_label}]")
        if chance is None and metric.startswith("recall@"):
            chance = int(metric.split("@")[1]) / n_eval
        if chance is not None:
            ax.axhline(chance, color="#c8312b", ls="-.", lw=0.9, alpha=0.6, label="chance")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks([0, 5, 10, 15, 20, 25])
        ax.grid(alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    note = "; approach B n/a for MNIST (decode() returns logits)" if task == "mnist" else ""
    fig.suptitle(f"{task}: common-space retrieval vs SNR (n={n_eval}){note}", fontsize=11)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(save_path, dpi=150)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/week5")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    figs = {}
    for task in PANELS:
        data = {ch: json.loads(p.read_text()) for ch in CHANNEL_STYLE if (p := Path(args.root) / task / ch / "results.json").exists()}
        if data:
            out = Path(args.root) / task / "retrieval_curves.png"
            figs[task] = plot_task(task, data, str(out))
            print(f"{task}: {sorted(data)} -> {out}")
    if args.wandb and figs:
        import wandb

        run = wandb.init(project="semcom", name="week5-retrieval-curves", group="summary", job_type="plot", mode="offline")
        run.log({f"retrieval_curves/{t}": wandb.Image(f) for t, f in figs.items()})
        run.finish()


if __name__ == "__main__":
    main()
