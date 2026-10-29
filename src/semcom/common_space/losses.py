import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorLoss(nn.Module):
    """Symmetric InfoNCE between head outputs and frozen anchors, plus an optional
    cosine term. InfoNCE alone only constrains the ranking of anchors; the cosine
    term pins absolute distances, which the Week 6 Fidelity component needs.

    The temperature is learnable (CLIP-style log-scale parameter), initialised at
    init_temp and clamped to [min_temp, max_temp]. forward() returns every term
    separately so both can be logged.
    """

    def __init__(self, init_temp: float = 0.07, min_temp: float = 0.01, max_temp: float = 0.5,
                 cos_weight: float = 0.5):
        super().__init__()
        self.log_inv_temp = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))
        self.min_temp, self.max_temp, self.cos_weight = min_temp, max_temp, cos_weight

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(-self.log_inv_temp).clamp(self.min_temp, self.max_temp)

    def forward(self, z: torch.Tensor, anchors: torch.Tensor) -> dict[str, torch.Tensor]:
        """z, anchors: (B, D), both L2-normalised; row i of z matches row i of anchors."""
        logits = z @ anchors.T / self.temperature
        target = torch.arange(z.size(0), device=z.device)
        nce = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
        cos = (1.0 - (z * anchors).sum(-1)).mean()
        return {"total": nce + self.cos_weight * cos, "nce": nce, "cos": cos, "temp": self.temperature.detach()}
