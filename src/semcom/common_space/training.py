"""Head-training utilities for Stage 2: learning-rate schedules, early stopping, a symbol cache and validation.

The Week 4 / Stage 1 transceivers are frozen, so the clean symbols `encode(x)` of a training item never change (eval mode, no dropout).
SymbolCache computes them once; each training step then only applies the channel and the head, which makes text heads several times
faster and gives numerically identical results to the uncached path for the same noise seed.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .retrieval import mean_cosine


def cosine_factor(warmup: int, total: int, min_frac: float):
    """Linear warmup then cosine decay to min_frac (the Week 5 / Stage 1 schedule)."""
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress))
    return fn


class LRSchedule:
    """Per-step learning rate. kind="cosine": identical to torch LambdaLR(cosine_factor). kind="plateau": linear warmup, then a
    constant rate that is multiplied by `factor` whenever `patience` consecutive epochs bring no validation improvement (floored
    at min_frac of the base rate), so a head that has stopped improving is annealed before early stopping ends it."""

    def __init__(self, opt, base_lr: float, warmup: int, kind: str = "cosine", total_steps: int = 1, min_frac: float = 0.05,
                 factor: float = 0.3, patience: int = 3):
        assert kind in ("cosine", "plateau")
        self.opt, self.base, self.warmup, self.kind = opt, base_lr, warmup, kind
        self.min_frac, self.factor, self.patience = min_frac, factor, patience
        self.cos = cosine_factor(warmup, total_steps, min_frac)
        self.t, self.scale, self.bad_epochs = 0, 1.0, 0
        self._apply()

    def _factor(self) -> float:
        if self.kind == "cosine":
            return self.cos(self.t)
        return ((self.t + 1) / self.warmup if self.t < self.warmup else 1.0) * self.scale

    def _apply(self) -> None:
        for g in self.opt.param_groups:
            g["lr"] = self.base * self._factor()

    def step(self) -> None:
        self.t += 1
        self._apply()

    def epoch_end(self, improved: bool) -> None:
        if self.kind != "plateau":
            return
        self.bad_epochs = 0 if improved else self.bad_epochs + 1
        if self.bad_epochs >= self.patience:
            self.scale = max(self.scale * self.factor, self.min_frac)
            self.bad_epochs = 0
            self._apply()

    @property
    def lr(self) -> float:
        return self.opt.param_groups[0]["lr"]


class EarlyStopper:
    """Stops after `patience` consecutive epochs without a strictly lower validation loss. patience=None never stops."""

    def __init__(self, patience: int | None):
        self.patience, self.best, self.best_epoch, self.bad = patience, float("inf"), 0, 0

    def update(self, value: float, epoch: int) -> tuple[bool, bool]:
        """Returns (improved, should_stop)."""
        improved = value < self.best
        if improved:
            self.best, self.best_epoch, self.bad = value, epoch, 0
        else:
            self.bad += 1
        return improved, self.patience is not None and self.bad >= self.patience


class SymbolCache:
    """Clean transceiver symbols for every item of an AnchoredData set, kept on `device`.
    Text keeps a padded (N, Lmax, D) tensor plus the token counts, so a batch is sliced to its own longest sentence (the same
    width the uncached path produces); other modalities are (N, D)."""

    def __init__(self, sym: torch.Tensor, lengths: torch.Tensor | None):
        self.sym, self.lengths = sym, lengths
        self.lengths_np = None if lengths is None else lengths.cpu().numpy()  # host copy: batch widths never need a device sync

    def __len__(self) -> int:
        return self.sym.size(0)

    @classmethod
    @torch.no_grad()
    def build(cls, w4, adapter, data, device, chunk: int = 512, width: int | None = None) -> "SymbolCache":
        syms, lens = [], []
        for s in range(0, len(data), chunk):
            items, _ = data.fetch(range(s, min(s + chunk, len(data))))
            batch = w4.spec.to_device(w4.spec.collate(items), device)
            sym = w4.model.encode(adapter.tx_input(batch))
            side = adapter.head_side(batch)
            if "mask" in side:
                lens.append(side["mask"].sum(1).cpu())
                width = width or w4.spec.tokenizer.max_len + 1
                sym = F.pad(sym, (0, 0, 0, width - sym.size(1)), value=1.0)  # pad rows are the constant row the encoder emits
            syms.append(sym.cpu())
        return cls(torch.cat(syms), torch.cat(lens) if lens else None)

    @classmethod
    def load_or_build(cls, path, w4, adapter, data, device, **kw) -> "SymbolCache":
        path = Path(path)
        if path.exists():
            blob = torch.load(path)
            if blob["sym"].size(0) == len(data):
                return cls(blob["sym"], blob["lengths"])
        cache = cls.build(w4, adapter, data, device, **kw)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{np.random.randint(1 << 30)}.tmp")
        torch.save({"sym": cache.sym, "lengths": cache.lengths}, tmp)
        tmp.replace(path)  # atomic: concurrent builders cannot leave a half-written file
        return cache

    def to(self, device) -> "SymbolCache":
        self.sym = self.sym.to(device)
        self.lengths = None if self.lengths is None else self.lengths.to(device)
        return self

    def widths(self, perm: np.ndarray, batch_size: int, n_steps: int, bucket: int = 8) -> list[int] | None:
        """Per-step padded width, computed on the host: the longest sentence in the batch rounded UP to a multiple of `bucket` (capped at the
        cache width). Padded positions are masked everywhere (a head's output does not depend on how much padding follows), so bucketing is
        numerically equivalent while limiting the number of distinct tensor shapes, which keeps the MPS allocator's memory from growing.
        None for fixed-size symbols."""
        if self.lengths_np is None:
            return None
        cap = self.sym.size(1)
        return [min(cap, -(-int(self.lengths_np[perm[b * batch_size:(b + 1) * batch_size]].max()) // bucket) * bucket) for b in range(n_steps)]

    def get_dev(self, idx: torch.Tensor, width: int | None = None) -> tuple[torch.Tensor, dict]:
        """Sync-free variant of get() for a device-resident index tensor and a host-known width (identical outputs)."""
        rx = self.sym[idx]
        if self.lengths is None:
            return rx, {}
        rx = rx[:, :width]
        return rx, {"mask": torch.arange(width, device=rx.device).unsqueeze(0) < self.lengths[idx].unsqueeze(1)}

    def get(self, idx) -> tuple[torch.Tensor, dict]:
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=self.sym.device)
        rx = self.sym[idx]
        if self.lengths is None:
            return rx, {}
        lens = self.lengths[idx]
        width = int(lens.max())
        return rx[:, :width], {"mask": torch.arange(width, device=rx.device).unsqueeze(0) < lens.unsqueeze(1)}


@torch.no_grad()
def validate(head, loss_fn, anchors: torch.Tensor, channel, snr_db: float, device, batch_size: int, n_batches: int, cache: SymbolCache,
             seed: int = 999) -> dict[str, float]:
    """Deterministic validation (fixed item order and noise) using cached clean symbols."""
    head.eval()
    torch.manual_seed(seed)
    perm = np.random.default_rng(seed).permutation(anchors.size(0))
    sums, cos_sum, n = {"total": 0.0, "nce": 0.0, "cos": 0.0}, 0.0, 0
    for b in range(min(n_batches, anchors.size(0) // batch_size)):
        idx = perm[b * batch_size:(b + 1) * batch_size]
        rx, side = cache.get(idx)
        a = anchors[torch.as_tensor(idx)].to(device)
        z = head(channel(rx, snr_db), **side)
        out = loss_fn(z, a)
        for k in sums:
            sums[k] += out[k].item()
        cos_sum += mean_cosine(z, a)
        n += 1
    return {**{f"val/{k}": v / n for k, v in sums.items()}, "val/cos_to_anchor": cos_sum / n}


def clip_and_guard(params, max_norm: float) -> torch.Tensor:
    """Gradient clipping with no host sync. Non-finite gradient entries (rare: Rayleigh deep fades) become 0 instead of the whole step being
    skipped, so optimizer state is never corrupted; returns the (device) total norm."""
    for p in params:
        if p.grad is not None:
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.nn.utils.clip_grad_norm_(params, max_norm)


def sort_within_batches(perm: np.ndarray, lengths: np.ndarray, batch_size: int, n_steps: int) -> np.ndarray:
    """Reorder the items of every batch by descending sentence length. Batch MEMBERSHIP is unchanged (so in-batch negatives are the same
    as in the unsorted path); only the order inside a batch changes, which the loss does not depend on."""
    out = perm.copy()
    for b in range(n_steps):
        sl = slice(b * batch_size, (b + 1) * batch_size)
        out[sl] = perm[sl][np.argsort(-lengths[perm[sl]], kind="stable")]
    return out


def chunk_bounds(batch_size: int, k: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, batch_size, k + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def chunk_widths(perm_sorted: np.ndarray, lengths: np.ndarray, batch_size: int, n_steps: int, k: int, bucket: int = 4, cap: int | None = None) -> list[list[int]]:
    """Padded width of each length-sorted chunk of each batch (longest sentence in the chunk, rounded up to `bucket`, capped)."""
    out = []
    for b in range(n_steps):
        ln = lengths[perm_sorted[b * batch_size:(b + 1) * batch_size]]
        w = [-(-int(ln[a:c].max()) // bucket) * bucket for a, c in chunk_bounds(batch_size, k)]
        out.append([min(x, cap) if cap else x for x in w])
    return out


def chunked_head_forward(head, channel, snr: float, sym: torch.Tensor, lengths: torch.Tensor, idx: torch.Tensor, bounds, widths) -> torch.Tensor:
    """Run a per-sentence head on length-sorted chunks of one batch (rows already sorted by descending length) and concatenate.
    Equivalent to one padded pass because the head never mixes sentences and masks all padding, but costs far fewer FLOPs."""
    rx, ln = sym[idx], lengths[idx]
    zs = []
    for (a, b), w in zip(bounds, widths):
        mask = torch.arange(w, device=rx.device).unsqueeze(0) < ln[a:b].unsqueeze(1)
        zs.append(head(channel(rx[a:b, :w], snr), mask=mask))
    return torch.cat(zs)
