import torch
import torch.nn as nn

from ..channel.base import Channel
from ..encoders.base import Transceiver
from .adapters import Adapter
from .base import CommonSpace, l2_normalize


class LearnedHeadSpace(CommonSpace):
    """Approach A: a small trainable head maps the received symbols into the anchor space.
    The Week 4 transceiver is frozen; only the head (and the loss temperature) train."""

    def __init__(self, transceiver: Transceiver, head: nn.Module, adapter: Adapter):
        self.transceiver = transceiver.eval().requires_grad_(False)
        self.head = head
        self.adapter = adapter
        self.modality = adapter.modality

    def embed_source(self, batch: dict) -> torch.Tensor:
        with torch.no_grad():
            return l2_normalize(self.adapter.encode_source(batch))

    def received_symbols(self, batch: dict, channel: Channel, snr_db: float) -> torch.Tensor:
        with torch.no_grad():
            return channel(self.transceiver.encode(self.adapter.tx_input(batch)), snr_db)

    def embed_received(self, batch: dict, channel: Channel, snr_db: float) -> torch.Tensor:
        rx = self.received_symbols(batch, channel, snr_db)
        return self.head(rx, **self.adapter.head_side(batch))
