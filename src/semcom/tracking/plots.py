from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHANNEL_STYLE = {"awgn": ("AWGN", "tab:blue", "o-"), "rayleigh": ("Rayleigh", "tab:red", "s--")}


def plot_snr_curves(
    results: dict[str, dict[float, dict[str, float]]],
    metrics: list[str],
    title: str,
    save_path: Optional[str] = None,
):
    """Task metric vs SNR, one panel per metric, one curve per channel.
    results[channel][snr_db][metric] -> value."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for channel, by_snr in results.items():
            label, color, style = CHANNEL_STYLE.get(channel, (channel, None, "o-"))
            snrs = sorted(by_snr)
            ax.plot(snrs, [by_snr[s][metric] for s in snrs], style, color=color, label=label)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
