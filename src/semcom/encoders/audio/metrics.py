import numpy as np
import torch
from pystoi import stoi as _stoi

try:
    from pesq import NoUtterancesError, pesq as _pesq
except ImportError:  # optional: C extension may be unavailable
    _pesq = None


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB for (B, N) waveforms."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    target = alpha * ref
    noise = est - target
    return 10 * torch.log10(target.pow(2).sum(-1).clamp_min(eps) / noise.pow(2).sum(-1).clamp_min(eps))


def stoi_mean(est: torch.Tensor, ref: torch.Tensor, sr: int) -> float:
    return float(np.mean([_stoi(r, e, sr, extended=False) for e, r in zip(est.numpy(), ref.numpy())]))


def pesq_mean(est: torch.Tensor, ref: torch.Tensor, sr: int) -> float | None:
    """Mean wideband PESQ over clips PESQ can score; None if unavailable."""
    if _pesq is None:
        return None
    scores = []
    for e, r in zip(est.numpy(), ref.numpy()):
        try:
            scores.append(_pesq(sr, r, e, "wb"))
        except NoUtterancesError:
            continue
    return float(np.mean(scores)) if scores else None
