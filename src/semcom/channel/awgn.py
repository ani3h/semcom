import torch

from .base import Channel, to_complex, to_real


class AWGNChannel(Channel):
    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        xc = to_complex(x)
        signal_power = xc.abs().pow(2).mean(dim=-1, keepdim=True)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = (noise_power / 2).sqrt()
        noise = torch.complex(
            torch.randn_like(xc.real) * noise_std,
            torch.randn_like(xc.real) * noise_std,
        )
        return to_real(xc + noise)
