import torch
import torch.nn as nn

from .base import EMBED_DIM, l2_normalize


class SymbolConditioner(nn.Module):
    """Input conditioning for received symbols: x -> clip * tanh(x / sigma / clip).

    sigma is the (constant) std of the transceiver's clean symbols, so inputs are roughly
    unit-scale at any SNR; the soft clip bounds Rayleigh outliers (up to ~500 sigma, because
    the zero-forcing equalizer amplifies deep fades) to +-clip. It is near-identity for
    |x| << clip*sigma. Heads cannot use the fading gain: the channel returns only equalized
    symbols. This clip shapes the Rayleigh results and is part of the method."""

    def __init__(self, sigma: float = 1.0, clip: float = 4.0):
        super().__init__()
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        self.clip = float(clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clip * torch.tanh(x / self.sigma / self.clip)


class MLPHead(nn.Module):
    """Flat received symbols (B, D) -> unit 512-d embedding (linear or one-hidden-layer MLP)."""

    def __init__(self, in_dim: int, out_dim: int = EMBED_DIM, hidden: int = 1024, kind: str = "mlp",
                 dropout: float = 0.1, sigma: float = 1.0, clip: float = 4.0):
        super().__init__()
        self.cond = SymbolConditioner(sigma, clip)
        self.net = nn.Linear(in_dim, out_dim) if kind == "linear" else nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, rx: torch.Tensor, mask=None) -> torch.Tensor:
        return l2_normalize(self.net(self.cond(rx)))


class TextAttentionHead(nn.Module):
    """Per-token symbols (B, L, sym_dim) + real-token mask -> unit embedding: token projection,
    learned positions, one small self-attention layer, masked mean pooling, linear out."""

    def __init__(self, sym_dim: int = 16, out_dim: int = EMBED_DIM, d_model: int = 256, nhead: int = 4,
                 ff: int = 512, max_len: int = 64, dropout: float = 0.1, sigma: float = 1.0, clip: float = 4.0):
        super().__init__()
        self.cond = SymbolConditioner(sigma, clip)
        self.proj = nn.Linear(sym_dim, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.out = nn.Linear(d_model, out_dim)

    def forward(self, rx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.cond(rx)) + self.pos(torch.arange(rx.size(1), device=rx.device))
        h = self.layer(h, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        return l2_normalize(self.out(pooled))


class AudioHead(nn.Module):
    """Flattened frame symbols (B, T*frame_dim) -> unit embedding: temporal conv over frames,
    mean + std pooling over time, linear out."""

    def __init__(self, frame_dim: int = 64, out_dim: int = EMBED_DIM, hidden: int = 256, kernel: int = 5,
                 dropout: float = 0.1, sigma: float = 1.0, clip: float = 4.0):
        super().__init__()
        self.frame_dim = frame_dim
        self.cond = SymbolConditioner(sigma, clip)
        self.conv = nn.Conv1d(frame_dim, hidden, kernel, padding=kernel // 2)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, out_dim)

    def forward(self, rx: torch.Tensor, mask=None) -> torch.Tensor:
        x = self.cond(rx).view(rx.size(0), -1, self.frame_dim).transpose(1, 2)
        h = nn.functional.gelu(self.conv(x))
        return l2_normalize(self.out(self.drop(torch.cat([h.mean(-1), h.std(-1)], dim=-1))))


HEADS = {"linear": MLPHead, "mlp": MLPHead, "text_attention": TextAttentionHead, "audio_conv": AudioHead}


def build_head(kind: str, sigma: float, **kwargs) -> nn.Module:
    if kind not in HEADS:
        raise ValueError(f"unknown head kind {kind!r}; choose from {sorted(HEADS)}")
    if kind in ("linear", "mlp"):
        kwargs["kind"] = kind
    return HEADS[kind](sigma=sigma, **kwargs)
