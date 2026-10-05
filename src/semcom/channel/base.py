from abc import ABC, abstractmethod

import torch


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """View a real tensor's last dim as I/Q pairs -> complex tensor with
    last dim halved. Lets the same channel module take arbitrary-shaped
    "transmitted symbols" from any modality's encoder."""
    if x.shape[-1] % 2 != 0:
        raise ValueError("Last dimension must be even to view as complex I/Q pairs.")
    return torch.view_as_complex(x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2).contiguous())


def to_real(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(x).reshape(*x.shape[:-1], -1)


class Channel(ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor: ...

    def __call__(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        return self.forward(x, snr_db)
