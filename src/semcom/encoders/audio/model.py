import torch
import torch.nn as nn

from ..base import Transceiver


class SEResBlock(nn.Module):
    """Residual 1-D conv block with squeeze-excitation channel attention.
    GroupNorm rather than BatchNorm: activations' statistics depend on the
    random training SNR, which would make BN running stats mismatch at eval."""

    def __init__(self, ch: int, kernel: int = 5, reduction: int = 8, groups: int = 8):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad)
        self.norm1 = nn.GroupNorm(groups, ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad)
        self.norm2 = nn.GroupNorm(groups, ch)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Conv1d(ch, ch // reduction, 1), nn.ReLU(),
            nn.Conv1d(ch // reduction, ch, 1), nn.Sigmoid(),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h * self.se(h))


class SEAudioTransceiver(Transceiver):
    """DeepSC-S-style speech transceiver: the waveform is cut into frame_len-sample
    frames, passed through SE-ResNet blocks, and a conv layer emits
    symbols_per_frame real values per frame. All frames of one clip are flattened
    into a single row for the channel (power normalised per clip). The decoder
    mirrors the encoder and regresses the waveform (MSE).

    Inputs are multiplied by a fixed `scale` buffer (1 / train-set RMS, set by
    the task's fit()) so the network sees roughly unit-variance signals; decode()
    undoes it, so the model's interface stays in waveform units."""

    modality = "audio"

    def __init__(
        self, frame_len: int = 128, channels: int = 128, num_blocks: int = 4,
        symbols_per_frame: int = 64, kernel: int = 5, scale: float = 1.0,
    ):
        super().__init__()
        assert symbols_per_frame % 2 == 0, "symbols are I/Q pairs"
        self.frame_len = frame_len
        self.symbols_per_frame = symbols_per_frame
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.enc_in = nn.Conv1d(frame_len, channels, kernel, padding=kernel // 2)
        self.enc_blocks = nn.Sequential(*[SEResBlock(channels, kernel) for _ in range(num_blocks)])
        self.enc_out = nn.Conv1d(channels, symbols_per_frame, 1)
        self.dec_in = nn.Conv1d(symbols_per_frame, channels, kernel, padding=kernel // 2)
        self.dec_blocks = nn.Sequential(*[SEResBlock(channels, kernel) for _ in range(num_blocks)])
        self.dec_out = nn.Conv1d(channels, frame_len, 1)

    def encode(self, wave: torch.Tensor) -> torch.Tensor:
        """wave: (B, N) with N divisible by frame_len -> (B, T * symbols_per_frame)."""
        B, N = wave.shape
        assert N % self.frame_len == 0, f"length {N} not divisible by frame_len {self.frame_len}"
        frames = (wave * self.scale).view(B, N // self.frame_len, self.frame_len).transpose(1, 2)
        h = self.enc_out(self.enc_blocks(self.enc_in(frames)))
        return h.transpose(1, 2).reshape(B, -1)

    def decode(self, symbols: torch.Tensor) -> torch.Tensor:
        B = symbols.size(0)
        h = symbols.view(B, -1, self.symbols_per_frame).transpose(1, 2)
        frames = self.dec_out(self.dec_blocks(self.dec_in(h)))
        return frames.transpose(1, 2).reshape(B, -1) / self.scale
