"""Fidelity candidates computed in the shared space S, and the statistics used to choose between them.

For items i = 1..N, S anchors a_i (unit, centered) and received embeddings z_i in S (unit):
    s_i      = <z_i, a_i>                              cosine to the own anchor
    b_i      = mean_{j != i} <z_i, a_j>                the item's baseline similarity to unrelated anchors
    margin_i = s_i - b_i
    F_abs_i  = margin_i / (1 - b_aa)                   HEADLINE. b_aa = mean unrelated anchor-anchor cosine, so the divisor is
                                                       the anchor's own margin; no clean-channel rescale, so it keeps encoder loss
    F_ret_i  = margin_i / m_clean                      retained fraction of the clean (noise-free) margin: channel loss only
    C        = m_clean / (1 - b_aa)                    clean ceiling per domain, so F_abs = F_ret * C (exactly)
plus the two baselines: raw cosine (cosine to the anchor in the space the head was trained in) and centered cosine (s_i in S).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

CANDIDATES = ("raw cosine", "centered cosine", "F_abs", "F_ret")


@dataclass
class Reference:
    m_clean: float  # mean margin of the noise-free (identity channel) embeddings
    b_aa: float  # mean cosine between different items' anchors

    @property
    def ceiling(self) -> float:
        return self.m_clean / (1.0 - self.b_aa)


def anchor_baseline(a: torch.Tensor) -> float:
    n = a.size(0)
    sims = a @ a.T
    return ((sims.sum() - sims.diag().sum()) / (n * (n - 1))).item()


def unrelated_baseline(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Per-item mean cosine between z_i and the anchors of the OTHER items, (N,)."""
    n = z.size(0)
    sims = z @ a.T
    return (sims.sum(1) - sims.diag()) / (n - 1)


def candidate_scores(z_raw, a_raw, z_s, a_s, ref: Reference, perm: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """All candidates for matched pairs (i, i), or for mismatched pairs (i, perm[i]) when perm is given. (N,) each."""
    idx = torch.arange(z_s.size(0)) if perm is None else perm
    base = unrelated_baseline(z_s, a_s)
    cen = (z_s * a_s[idx]).sum(-1)
    margin = cen - base
    return {"raw cosine": (z_raw * a_raw[idx]).sum(-1), "centered cosine": cen,
            "F_abs": margin / (1.0 - ref.b_aa), "F_ret": margin / ref.m_clean}


def make_reference(z_clean_s: torch.Tensor, a_s: torch.Tensor) -> Reference:
    margin = (z_clean_s * a_s).sum(-1) - unrelated_baseline(z_clean_s, a_s)
    return Reference(m_clean=margin.mean().item(), b_aa=anchor_baseline(a_s))


def derangement(n: int, seed: int = 0) -> torch.Tensor:
    """A fixed permutation without fixed points, used to pair each item with a wrong anchor."""
    g = torch.Generator().manual_seed(seed)
    return (torch.arange(n) + torch.randint(1, n, (1,), generator=g)) % n


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(matched score > mismatched score) (Mann-Whitney), ties counted half."""
    ranks = pd.concat([pd.Series(np.ravel(pos)), pd.Series(np.ravel(neg))]).rank()
    n1, n0 = np.size(pos), np.size(neg)
    return float((ranks.iloc[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def snr_at_fraction(snrs: list[float], means: list[float], clean: float, frac: float = 0.9) -> dict:
    """Lowest SNR at which the mean score reaches frac * clean (linear interpolation between grid points)."""
    target = frac * clean
    if clean <= 0:
        return {"status": "undefined", "snr": None}
    if means[0] >= target:
        return {"status": "at_or_below_lowest_grid_point", "snr": float(snrs[0])}
    for k in range(1, len(snrs)):
        if means[k] >= target:
            t = (target - means[k - 1]) / (means[k] - means[k - 1])
            return {"status": "interpolated", "snr": float(snrs[k - 1] + t * (snrs[k] - snrs[k - 1]))}
    return {"status": "not_reached_by_grid_max", "snr": None}


def analyze(per_snr: dict[float, np.ndarray], mismatched: dict[float, np.ndarray], clean: np.ndarray, n_boot: int = 500,
            seed: int = 0) -> dict:
    """Statistics for one candidate. per_snr[snr] and mismatched[snr] are (draws, N) arrays; clean is (N,)."""
    snrs = sorted(per_snr)
    item_mean = {s: per_snr[s].mean(0) for s in snrs}  # average over noise draws, (N,)
    means = [float(item_mean[s].mean()) for s in snrs]
    rng = np.random.default_rng(seed)
    n = len(clean)
    boots = rng.integers(0, n, size=(n_boot, n))
    steps = []
    for lo, hi in zip(snrs[:-1], snrs[1:]):
        d = item_mean[hi] - item_mean[lo]
        steps.append({"from": lo, "to": hi, "delta": float(d.mean()), "ci_lo": float(np.quantile(d[boots].mean(1), 0.025))})
    pooled = pd.DataFrame({"F": np.concatenate([per_snr[s].ravel() for s in snrs]),
                           "snr": np.concatenate([np.full(per_snr[s].size, s) for s in snrs])})
    rng_range = means[-1] - means[0]
    clean_mean = float(np.mean(clean))
    mid = min(snrs, key=lambda s: abs(s - 10.0))
    return {"means": dict(zip(map(str, snrs), means)), "clean": clean_mean, "strictly_increasing": all(b > a for a, b in zip(means, means[1:])),
            "steps_significant": int(sum(s["ci_lo"] > 0 for s in steps)), "n_steps": len(steps), "steps": steps,
            "dynamic_range": float(rng_range), "spearman_pooled": float(pooled["F"].corr(pooled["snr"], method="spearman")),
            "snr_at_90pct_clean": snr_at_fraction(snrs, means, clean_mean),
            "auc": {str(s): auc(per_snr[s], mismatched[s]) for s in snrs},
            "item_sd_over_range_at_10dB": float(item_mean[mid].std() / rng_range) if rng_range > 0 else None}


def choose_variant(val_stats: dict[str, dict[str, dict]]) -> tuple[str, dict]:
    """Pre-registered rule, applied to VALIDATION statistics of F_abs only (single seed):
    1. most (task, channel) pairs whose mean F_abs is strictly increasing over the SNR grid;
    2. ties: larger median dynamic range; 3. ties: larger median AUC at the lowest SNR.
    val_stats[variant][pair] = analyze(...) of F_abs. Returns (chosen, per-variant summary)."""
    summary = {}
    for variant, pairs in val_stats.items():
        ranges = [p["dynamic_range"] for p in pairs.values()]
        lowest = [p["auc"][min(p["auc"], key=float)] for p in pairs.values()]
        summary[variant] = {"n_pairs": len(pairs), "n_strictly_increasing": sum(p["strictly_increasing"] for p in pairs.values()),
                            "median_dynamic_range": float(np.median(ranges)), "median_auc_lowest_snr": float(np.median(lowest))}
    chosen = max(summary, key=lambda v: (summary[v]["n_strictly_increasing"], round(summary[v]["median_dynamic_range"], 4),
                                         summary[v]["median_auc_lowest_snr"]))
    return chosen, summary
