import numpy as np
import torch

from .base import Channel


def snr_sweep(
    channel: Channel, x: torch.Tensor, snr_range_db: tuple[float, float] = (0, 25), step_db: float = 1.0
) -> dict[float, torch.Tensor]:
    snrs = np.arange(snr_range_db[0], snr_range_db[1] + step_db, step_db)
    return {float(snr): channel(x, float(snr)) for snr in snrs}
