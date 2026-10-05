from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class Transceiver(nn.Module, ABC):
    """Modality-specific semantic encoder/decoder pair behind one interface.

    encode(x) -> real "symbols" tensor whose last dim is even (the channel views
    it as I/Q pairs). Image/audio models flatten to one vector per sample so the
    channel's per-row power normalization is per sample; text keeps one row per
    token. Week 5's projection heads attach to the encode() output.

    decode(symbols, **side_info) -> task output (text token logits / generated ids,
    image reconstruction, class logits, waveform).

    forward(x, channel, snr_db, **side_info) runs encode -> channel -> decode.
    """

    modality: str

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, symbols: torch.Tensor, **side_info: Any) -> Any: ...

    def forward(self, x: torch.Tensor, channel, snr_db: float, **side_info: Any) -> Any:
        return self.decode(channel(self.encode(x), snr_db), **side_info)


class TaskSpec(ABC):
    """Everything modality-specific that the generic training/eval driver needs:
    model construction, batching, training loss, and the task metric(s)."""

    modality: str
    metric_names: list[str]
    primary_metric: str

    def fit(self, train_dataset) -> None:
        """Hook for statistics derived from the train split (vocab, audio scale)."""

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass

    @abstractmethod
    def build_model(self) -> Transceiver: ...

    @abstractmethod
    def collate(self, items: list[dict]) -> dict[str, torch.Tensor]: ...

    @abstractmethod
    def loss(self, model: Transceiver, batch: dict, channel, snr_db: float) -> torch.Tensor: ...

    @abstractmethod
    def evaluate(self, model: Transceiver, dataset, channel, snr_db: float, device) -> dict[str, float]: ...

    @staticmethod
    def to_device(batch: dict, device) -> dict:
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
