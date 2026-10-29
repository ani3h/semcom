import torch

from ..channel.base import Channel
from ..encoders.base import Transceiver
from .adapters import Adapter
from .base import CommonSpace, l2_normalize


class DirectPretrainedSpace(CommonSpace):
    """Approach B (baseline, no training): decode the transceiver's output, then embed that with the
    frozen pretrained CLIP/CLAP. Not available where decode() is not a reconstruction (MNIST)."""

    def __init__(self, transceiver: Transceiver, adapter: Adapter):
        if not adapter.supports_direct:
            adapter.embed_decoded(None)  # raises NotImplementedError with the reason
        self.transceiver = transceiver.eval().requires_grad_(False)
        self.adapter = adapter
        self.modality = adapter.modality

    def embed_source(self, batch: dict) -> torch.Tensor:
        with torch.no_grad():
            return l2_normalize(self.adapter.encode_source(batch))

    def embed_received(self, batch: dict, channel: Channel, snr_db: float) -> torch.Tensor:
        with torch.no_grad():
            out = self.transceiver(self.adapter.tx_input(batch), channel, snr_db, **self.adapter.decode_side(batch))
            return l2_normalize(self.adapter.embed_decoded(out))
