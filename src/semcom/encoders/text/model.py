import math

import torch
import torch.nn as nn

from ..base import Transceiver
from .tokenizer import BOS_ID, EOS_ID, PAD_ID


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class TextTransceiver(Transceiver):
    """DeepSC-style: Transformer semantic encoder -> dense channel encoder
    (symbols_per_token real values per token) -> channel -> dense channel
    decoder -> Transformer decoder. The receiver is given the sentence-length
    mask (src_mask) as side information, as in DeepSC's reference code."""

    modality = "text"

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_ff: int = 512,
        dropout: float = 0.1,
        symbols_per_token: int = 16,
        max_len: int = 128,
    ):
        super().__init__()
        assert symbols_per_token % 2 == 0, "symbols are I/Q pairs"
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, max_len)

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.channel_encoder = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, symbols_per_token))
        self.channel_decoder = nn.Sequential(nn.Linear(symbols_per_token, 256), nn.ReLU(), nn.Linear(256, d_model))

        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=nn.LayerNorm(d_model))
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.tgt_embed.weight
        # The output projection is tied to tgt_embed, so N(0,1) default embedding init
        # would give huge initial logits (loss >> ln(vocab)); use std d_model^-0.5.
        for emb in (self.src_embed, self.tgt_embed):
            nn.init.normal_(emb.weight, mean=0.0, std=d_model**-0.5)
            with torch.no_grad():
                emb.weight[PAD_ID].zero_()

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """src: (B, L) token ids (padded) -> (B, L, symbols_per_token) real symbols."""
        mask = src != PAD_ID
        h = self.pos(self.src_embed(src) * math.sqrt(self.d_model))
        h = self.encoder(h, src_key_padding_mask=~mask)
        sym = self.channel_encoder(h)
        # Padded positions are not information, but they must not be all-zero rows:
        # the channel's noise scale is sqrt(signal power), whose gradient is
        # infinite at zero power. Send a constant row; the receiver masks it out.
        return torch.where(mask.unsqueeze(-1), sym, torch.ones_like(sym))

    def decode(self, symbols: torch.Tensor, src_mask: torch.Tensor, tgt_in: torch.Tensor | None = None):
        """With tgt_in: teacher-forced logits (B, T, V). Without: greedy token ids (B, <=L)."""
        memory = self.channel_decoder(symbols)
        if tgt_in is not None:
            return self._logits(tgt_in, memory, src_mask)
        return self._greedy(memory, src_mask)

    def _logits(self, tgt_in: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        T = tgt_in.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_in.device), diagonal=1)
        h = self.pos(self.tgt_embed(tgt_in) * math.sqrt(self.d_model))
        h = self.decoder(h, memory, tgt_mask=causal, memory_key_padding_mask=~src_mask)
        return self.out(h)

    @torch.no_grad()
    def _greedy(self, memory: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        B = memory.size(0)
        ys = torch.full((B, 1), BOS_ID, dtype=torch.long, device=memory.device)
        done = torch.zeros(B, dtype=torch.bool, device=memory.device)
        for _ in range(memory.size(1)):
            nxt = self._logits(ys, memory, src_mask)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done |= nxt == EOS_ID
            if done.all():
                break
        return ys[:, 1:]
