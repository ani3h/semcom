"""Text-bridged mapping from CLAP space to CLIP space (approach (b) for the shared space S).

Trained ONLY on caption text: x = normalize(CLAP_text(c) - mu_clap_text) -> y = normalize(CLIP_text(c) - mu_clip_text).
Applied to audio by feeding x = normalize(CLAP_audio(a) - mu_clap_audio): the per-modality centering is what closes
the CLAP audio-vs-text modality gap (a mean shift), an assumption the Stage 0 gate tests.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import EMBED_DIM, l2_normalize
from .losses import AnchorLoss
from .retrieval import recall_at_k


class TextBridge(nn.Module):
    def __init__(self, kind: str = "mlp", dim: int = EMBED_DIM, hidden: int = 1024):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(dim, dim)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        else:
            raise ValueError(f"unknown bridge kind {kind!r}")
        self.kind, self.dim, self.hidden = kind, dim, hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.net(x))


def ridge_fit(x: torch.Tensor, y: torch.Tensor, lam: float = 1e-2) -> torch.Tensor:
    """Closed-form least-squares reference map (a regression baseline, not the trained bridge)."""
    return torch.linalg.solve(x.T @ x + lam * torch.eye(x.size(1)), x.T @ y)


@torch.no_grad()
def agreement(bridge_out: torch.Tensor, target: torch.Tensor, gallery: int = 1000, max_galleries: int = 5) -> dict:
    """Text-to-text agreement on held-out captions, all in the centered target space: mean cosine and
    caption-retrieval recall@1/5 within disjoint galleries of `gallery` captions (chance 1/gallery)."""
    n = bridge_out.size(0)
    cos = (bridge_out * target).sum(-1).mean().item()
    recs = []
    for g in range(min(max_galleries, n // gallery)):
        sl = slice(g * gallery, (g + 1) * gallery)
        recs.append(recall_at_k(bridge_out[sl], target[sl], ks=(1, 5, 10)))
    out = {"cos": cos, "n": n, "gallery": gallery}
    for k in recs[0]:
        out[k] = float(np.mean([r[k] for r in recs]))
    return out


def train_bridge(x_tr, y_tr, x_val, y_val, kind: str = "mlp", hidden: int = 1024, epochs: int = 12,
                 batch_size: int = 1024, lr: float = 1e-3, weight_decay: float = 1e-4, cos_weight: float = 0.5,
                 init_temp: float = 0.07, seed: int = 42, device="cpu", gallery: int = 1000, log=print):
    """x_*: centered+normalised CLAP-text embeddings, y_*: centered+normalised CLIP-text embeddings.
    Loss is the same symmetric InfoNCE + cosine as the Week 5 heads. Returns (best-by-val bridge on cpu, history)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dim = x_tr.size(1)
    bridge = TextBridge(kind, dim=dim, hidden=hidden).to(device)
    loss_fn = AnchorLoss(init_temp=init_temp, cos_weight=cos_weight).to(device)
    params = list(bridge.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    n = x_tr.size(0)
    steps_per_epoch = n // batch_size
    total = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(s, total) / total)))
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    best, best_state, history, t0 = (-1.0, -1.0), None, [], time.time()
    for epoch in range(1, epochs + 1):
        bridge.train()
        perm = torch.randperm(n, generator=gen)
        for b in range(steps_per_epoch):
            idx = perm[b * batch_size:(b + 1) * batch_size].to(device)
            out = loss_fn(bridge(x_tr[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            out["total"].backward()
            opt.step()
            sched.step()
        bridge.eval()
        with torch.no_grad():
            val = agreement(bridge(x_val.to(device)).cpu(), y_val, gallery)
        history.append({"epoch": epoch, "train_loss": out["total"].item(), **{f"val_{k}": v for k, v in val.items()}})
        log(f"  [{kind} seed {seed}] epoch {epoch:2d}/{epochs} loss {out['total'].item():.3f} val cos {val['cos']:.3f} "
            f"R@1 {val['recall@1']:.3f} R@5 {val['recall@5']:.3f} temp {out['temp'].item():.3f} [{time.time() - t0:.0f}s]")
        if (val["recall@1"], val["cos"]) > best:  # validation recall@1; ties (saturated retrieval) broken by cosine
            best, best_state = (val["recall@1"], val["cos"]), {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}
    final = TextBridge(kind, dim=dim, hidden=hidden)
    final.load_state_dict(best_state)
    return final.eval(), history
