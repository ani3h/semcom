"""Per-modality glue between a Week 4 transceiver's batch format and the common space."""
import re
from abc import ABC, abstractmethod

import torch

from ..encoders.text.tokenizer import PAD_ID, UNK, WordTokenizer

_NO_SPACE_BEFORE = re.compile(r"\s+([,.;:!?%)\]])")
_NO_SPACE_AFTER = re.compile(r"([(\[$])\s+")


def detokenize(tokens: list[str]) -> str:
    """Tokenizer output (lowercase, punctuation as separate tokens) -> natural-looking text for
    CLIP. <unk> tokens are dropped (they are lost information, not words)."""
    text = " ".join(t for t in tokens if t != UNK)
    return _NO_SPACE_AFTER.sub(r"\1", _NO_SPACE_BEFORE.sub(r"\1", text)).strip()


class Adapter(ABC):
    modality: str
    supports_direct: bool = True

    def __init__(self, anchors):
        self.anchors = anchors  # PretrainedAnchors or a stub with the same encode_* methods

    @abstractmethod
    def tx_input(self, batch: dict) -> torch.Tensor:
        """What the transceiver's encode() consumes."""

    def head_side(self, batch: dict) -> dict:
        """Extra inputs a head needs (e.g. the real-token mask for text)."""
        return {}

    def decode_side(self, batch: dict) -> dict:
        """Side information the transceiver's decode() needs."""
        return {}

    @abstractmethod
    def encode_source(self, batch: dict) -> torch.Tensor:
        """Frozen pretrained anchor of the clean original input."""

    @abstractmethod
    def embed_decoded(self, output) -> torch.Tensor:
        """Approach B: embed the transceiver's decoded output with the pretrained model."""


class ImageReconAdapter(Adapter):
    """CIFAR-10: decode() returns a reconstructed image."""

    modality = "image"

    def tx_input(self, batch):
        return batch["x"]

    def encode_source(self, batch):
        return self.anchors.encode_images(batch["x"])

    def embed_decoded(self, output):
        return self.anchors.encode_images(output.clamp(0, 1))


class ImageClassAdapter(ImageReconAdapter):
    """MNIST: decode() returns class logits, so there is nothing to feed a pretrained encoder."""

    supports_direct = False

    def embed_decoded(self, output):
        raise NotImplementedError("MNIST decode() returns class logits, not a reconstruction: approach B does not apply")


class TextAdapter(Adapter):
    modality = "text"

    def __init__(self, anchors, tokenizer: WordTokenizer):
        super().__init__(anchors)
        self.tokenizer = tokenizer

    def tx_input(self, batch):
        return batch["src"]

    def head_side(self, batch):
        return {"mask": batch["src"] != PAD_ID}

    def decode_side(self, batch):
        return {"src_mask": batch["src"] != PAD_ID}

    def encode_source(self, batch):
        return self.anchors.encode_text(batch["texts"])  # the raw original, not its tokenizer round trip

    def roundtrip(self, texts: list[str]) -> list[str]:
        """What the transceiver can at best deliver for these sentences (tokenizer loss only)."""
        return [detokenize(self.tokenizer.ids_to_tokens(self.tokenizer.encode(t))) for t in texts]

    def embed_decoded(self, output):
        texts = [detokenize(self.tokenizer.ids_to_tokens(row)) for row in output.cpu()]
        return self.anchors.encode_text(texts)


class AudioAdapter(Adapter):
    modality = "audio"

    def __init__(self, anchors, sample_rate: int = 16000):
        super().__init__(anchors)
        self.sample_rate = sample_rate

    def tx_input(self, batch):
        return batch["x"]

    def encode_source(self, batch):
        return self.anchors.encode_audio(batch["x"], self.sample_rate)

    def embed_decoded(self, output):
        return self.anchors.encode_audio(output, self.sample_rate)


def make_adapter(task: str, anchors, spec=None) -> Adapter:
    if task == "cifar10":
        return ImageReconAdapter(anchors)
    if task == "mnist":
        return ImageClassAdapter(anchors)
    if task in ("text", "caption_text"):
        return TextAdapter(anchors, spec.tokenizer)
    if task in ("audio", "esc50_audio"):
        return AudioAdapter(anchors)
    raise ValueError(f"unknown task {task!r}")
