from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .awgn import AWGNChannel
from .rayleigh import RayleighChannel


def toy_qpsk_symbols(n: int = 256, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (n, 2), generator=g).float() * 2 - 1
    return bits  # (n, 2) real I/Q pairs -> n complex symbols under to_complex()


def plot_channel_effects(snrs_db: tuple[float, ...] = (0, 10, 20), save_path: Optional[str] = None):
    x = toy_qpsk_symbols().unsqueeze(0)  # (1, n, 2)
    channels = {"AWGN": AWGNChannel(), "Rayleigh": RayleighChannel()}

    fig, axes = plt.subplots(len(channels), len(snrs_db), figsize=(4 * len(snrs_db), 4 * len(channels)))
    for row, (name, ch) in enumerate(channels.items()):
        for col, snr in enumerate(snrs_db):
            y = ch(x, snr).squeeze(0)
            ax = axes[row][col]
            ax.scatter(y[:, 0].numpy(), y[:, 1].numpy(), s=8, alpha=0.6)
            ax.set_title(f"{name} @ {snr}dB")
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect("equal")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
