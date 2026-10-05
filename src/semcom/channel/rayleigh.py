import torch

from .base import Channel, to_complex, to_real


class RayleighChannel(Channel):
    """Flat Rayleigh fading + AWGN, with perfect-CSI equalization at the
    receiver (divide by the known fading gain) — the standard simplifying
    assumption in JSCC/SemCom literature so output shape tracks input shape.

    Zero-forcing makes post-equalization noise heavy-tailed (E[1/|h|^2]
    diverges for Rayleigh), so averaged MSE is dominated by rare deep fades."""

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        xc = to_complex(x)
        h = torch.complex(torch.randn_like(xc.real), torch.randn_like(xc.real)) / (2**0.5)
        faded = xc * h

        signal_power = faded.abs().pow(2).mean(dim=-1, keepdim=True)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = (noise_power / 2).sqrt()
        noise = torch.complex(
            torch.randn_like(xc.real) * noise_std,
            torch.randn_like(xc.real) * noise_std,
        )

        received = faded + noise
        equalized = received / h
        return to_real(equalized)
