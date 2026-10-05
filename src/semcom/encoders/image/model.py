import torch
import torch.nn as nn

from ..base import Transceiver


def _conv(cin: int, cout: int, k: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2), nn.PReLU(cout))


def _deconv(cin: int, cout: int, k: int, stride: int, last: bool = False) -> nn.Sequential:
    layers = [nn.ConvTranspose2d(cin, cout, k, stride, k // 2, output_padding=stride - 1)]
    layers.append(nn.Sigmoid() if last else nn.PReLU(cout))
    return nn.Sequential(*layers)


class JSCCImage(Transceiver):
    """Bourtsoulatze-style deep JSCC for 32x32 RGB reconstruction (CIFAR-10).

    Encoder: 5 conv layers (two stride-2) -> (c, 8, 8) feature map flattened to
    one 2k-vector per image, k = c*64/2 complex channel uses. With c=16 that is
    512 complex symbols for 3072 source values = bandwidth ratio 1/6. The
    original uses GDN nonlinearities; this baseline uses PReLU only.
    """

    modality = "image"

    def __init__(self, channels: int = 16, width: int = 32, in_ch: int = 3):
        super().__init__()
        self.channels = channels
        self.enc = nn.Sequential(
            _conv(in_ch, 16, 5, 2), _conv(16, width, 5, 2), _conv(width, width, 5, 1),
            _conv(width, width, 5, 1), _conv(width, channels, 5, 1),
        )
        self.dec = nn.Sequential(
            _deconv(channels, width, 5, 1), _deconv(width, width, 5, 1), _deconv(width, width, 5, 1),
            _deconv(width, 16, 5, 2), _deconv(16, in_ch, 5, 2, last=True),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x).flatten(1)

    def decode(self, symbols: torch.Tensor) -> torch.Tensor:
        return self.dec(symbols.view(symbols.size(0), self.channels, 8, 8))


class JSCCClassifier(Transceiver):
    """Task-oriented JSCC for 28x28 MNIST: conv encoder -> 4x4x16 = 256 real
    values (128 complex symbols, ~1/6 of 784 pixels); the receiver is a small
    classifier head on the received symbols (decode() returns class logits)."""

    modality = "image"

    def __init__(self, num_classes: int = 10, channels: int = 16, in_ch: int = 1):
        super().__init__()
        self.channels = channels
        self.enc = nn.Sequential(
            _conv(in_ch, 16, 5, 2), _conv(16, 32, 5, 2), _conv(32, 32, 3, 1), _conv(32, channels, 3, 2),
        )
        self.head = nn.Sequential(
            nn.Conv2d(channels, 32, 3, 1, 1), nn.PReLU(32), nn.Flatten(),
            nn.Linear(32 * 16, 128), nn.PReLU(), nn.Linear(128, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x).flatten(1)

    def decode(self, symbols: torch.Tensor) -> torch.Tensor:
        return self.head(symbols.view(symbols.size(0), self.channels, 4, 4))
