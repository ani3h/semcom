from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

EMBED_DIM = 512


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


class CommonSpace(ABC):
    """Maps a modality's source and received signals into one fixed-dim, L2-normalised space.

    `batch` is the dict produced by the Week 4 TaskSpec.collate() (already on the target
    device) plus whatever raw fields that adapter needs (text carries "texts").

    embed_source(batch): the frozen pretrained anchor of the clean original (CLIP for
      image/text, CLAP for audio) -- the same reference for every method.
    embed_received(batch, channel, snr_db): what comes out of the transceiver after the
      channel. Approach A (learned head) embeds the received symbols; approach B embeds
      the decoded output with the pretrained model.

    CLIP and CLAP are different, unaligned 512-d spaces: image and text share one, audio
    lives in the other. Cosine values are therefore only comparable within a modality.
    """

    dim: int = EMBED_DIM
    modality: str

    @abstractmethod
    def embed_source(self, batch: dict) -> torch.Tensor: ...

    @abstractmethod
    def embed_received(self, batch: dict, channel, snr_db: float) -> torch.Tensor: ...
