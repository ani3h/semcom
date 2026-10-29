"""Statistics for the Stage 2 evaluation: percentile bootstrap over items, seed spread, class-level retrieval scores, and the
amended variant-selection rule. Pure numpy / torch, no model code.

Conventions
-----------
Per-item scores live in arrays whose LAST axis is the item axis. Noise draws and head seeds are averaged per item first, so an item is
the unit of resampling; the seed-to-seed spread is reported separately (mean and sd over the per-seed means), never mixed into a CI.
"""
import numpy as np
import pandas as pd
import torch

from .fidelity import auc as auc_stat
from .fidelity import snr_at_fraction

N_BOOT = 1000


def boot_idx(n: int, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """(n_boot, n) item indices shared by every statistic of a table, so related numbers are resampled together."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def ci(draws: np.ndarray, point: float, alpha: float = 0.05) -> dict:
    return {"point": float(point), "lo": float(np.quantile(draws, alpha / 2)), "hi": float(np.quantile(draws, 1 - alpha / 2))}


def mean_ci(x: np.ndarray, idx: np.ndarray) -> dict:
    """Mean over the last axis with a percentile bootstrap over items. x: (..., N) -> dicts of arrays-free scalars when 1-D."""
    return ci(x[idx].mean(-1), x.mean())


def mean_ci_curve(x: np.ndarray, idx: np.ndarray) -> list[dict]:
    """x: (S, N) per-item scores at S conditions -> one CI dict per condition (same resampled items for all conditions)."""
    d = x[:, idx].mean(-1)  # (S, n_boot)
    return [ci(d[s], x[s].mean()) for s in range(x.shape[0])]


def seed_spread(per_seed: np.ndarray) -> dict:
    """per_seed: (seeds,) scalar per seed -> mean and sd (ddof=1) over seeds, kept separate from the item bootstrap."""
    per_seed = np.asarray(per_seed, dtype=float)
    return {"mean": float(per_seed.mean()), "sd": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0, "values": per_seed.tolist()}


def balanced_acc(correct: np.ndarray, labels: np.ndarray, classes: np.ndarray | None = None) -> float:
    """Mean per-class accuracy; correct (N,) 0/1, labels (N,)."""
    classes = np.unique(labels) if classes is None else classes
    return float(np.mean([correct[labels == c].mean() for c in classes if (labels == c).any()]))


def balanced_acc_ci(correct: np.ndarray, labels: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Class-stratified percentile bootstrap of balanced accuracy. correct: (N,) per-item accuracy in [0,1] (draw/seed averaged)."""
    rng = np.random.default_rng(seed)
    groups = [np.where(labels == c)[0] for c in np.unique(labels)]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = np.mean([correct[rng.choice(g, size=len(g), replace=True)].mean() for g in groups])
    return ci(draws, balanced_acc(correct, labels))


def curve_balanced_acc_ci(correct: np.ndarray, labels: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> list[dict]:
    """correct: (S, N) per-item accuracy at S conditions; the same stratified resamples are applied to every condition."""
    rng = np.random.default_rng(seed)
    groups = [np.where(labels == c)[0] for c in np.unique(labels)]
    S = correct.shape[0]
    draws = np.empty((S, n_boot))
    for b in range(n_boot):
        draws[:, b] = np.mean([correct[:, rng.choice(g, size=len(g), replace=True)].mean(-1) for g in groups], axis=0)
    return [ci(draws[s], balanced_acc(correct[s], labels)) for s in range(S)]


def diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """CI of mean(a) - mean(b) for two independent item sets (e.g. training templates vs held-out paraphrases)."""
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(-1)
    db = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(-1)
    return ci(da - db, a.mean() - b.mean())


# ---------------------------------------------------------------------------------------------------------------------------
def class_correct(query: torch.Tensor, protos: torch.Tensor, labels: torch.Tensor, k: int = 1) -> np.ndarray:
    """1/0 per query: is the true class among the k nearest class prototypes (cosine). query (N,D), protos (C,D)."""
    scores = query.float() @ protos.float().T
    true = scores.gather(1, labels.view(-1, 1))
    return (1 + (scores > true).sum(1) <= k).float().numpy()


def self_retrieval(query: torch.Tensor, gallery: torch.Tensor, ks=(1, 5)) -> dict[str, np.ndarray]:
    """Per-item retrieval of the item's own clean anchor among the gallery (row i <-> row i): recall@k indicators, cosine, margin."""
    sims = query.float() @ gallery.float().T
    true = sims.diag().unsqueeze(1)
    rank = 1 + (sims > true).sum(1)
    n = sims.size(0)
    base = (sims.sum(1) - sims.diag()) / (n - 1)
    out = {f"recall@{k}": (rank <= k).float().numpy() for k in ks if k <= n}
    out["cos_to_anchor"] = sims.diag().numpy()
    out["margin"] = (sims.diag() - base).numpy()
    return out


# ---------------------------------------------------------------------------------------------------------------------------
def f_curve_stats(F: np.ndarray, F_mis: np.ndarray, clean: np.ndarray, snrs: list[float], idx: np.ndarray, auc_snrs=(0.0, 10.0)) -> dict:
    """Fidelity statistics with item-bootstrap CIs.

    F:     (S, N) per-item F, already averaged over noise draws and head seeds.
    F_mis: (S, N) per-item F for each item paired with a WRONG anchor (same averaging), the AUC negatives.
    clean: (N,) per-item F of the noise-free signal.
    Returns point + CI for: mean per SNR, clean mean, dynamic range, SNR at 90% of clean, AUC at auc_snrs, plus the point-only strictly-increasing
    flag, the share of bootstrap resamples in which the mean curve is strictly increasing, pooled Spearman with SNR, and item-noise/range at 10 dB.
    """
    S, N = F.shape
    snr_arr = np.asarray(snrs, dtype=float)
    means = F.mean(1)
    bm = F[:, idx].mean(-1)  # (S, n_boot)
    bclean = clean[idx].mean(-1)
    rng_pt = means[-1] - means[0]

    def pooled_spearman(Fm, rows):
        y = np.repeat(snr_arr[:, None], Fm.shape[1], axis=1)
        return float(pd.Series(Fm[:, rows].ravel()).corr(pd.Series(y[:, rows].ravel()), method="spearman"))

    spear_pt = pooled_spearman(F, np.arange(N))
    n_b = min(idx.shape[0], 200)  # Spearman over S*N values per resample is the costliest statistic
    spear_draws = np.array([pooled_spearman(F, idx[b]) for b in range(n_b)])
    snr90 = [snr_at_fraction(list(snr_arr), list(bm[:, b]), float(bclean[b]))["snr"] for b in range(idx.shape[0])]
    snr90 = np.array([np.nan if v is None else v for v in snr90], dtype=float)
    pt90 = snr_at_fraction(list(snr_arr), list(means), float(clean.mean()))
    out = {"snrs": list(map(float, snrs)), "means": [ci(bm[s], means[s]) for s in range(S)], "clean": ci(bclean, clean.mean()),
           "dynamic_range": ci(bm[-1] - bm[0], rng_pt), "strictly_increasing": bool(np.all(np.diff(means) > 0)),
           "frac_resamples_strictly_increasing": float(np.mean(np.all(np.diff(bm, axis=0) > 0, axis=0))),
           "spearman_pooled": ci(spear_draws, spear_pt),
           "snr_at_90pct_clean": {"point": pt90["snr"], "status": pt90["status"],
                                  "lo": None if np.all(np.isnan(snr90)) else float(np.nanquantile(snr90, 0.025)),
                                  "hi": None if np.all(np.isnan(snr90)) else float(np.nanquantile(snr90, 0.975)),
                                  "frac_resamples_reached": float(np.mean(~np.isnan(snr90)))}}
    mid = int(np.argmin(np.abs(snr_arr - 10.0)))
    out["item_sd_over_range_at_10dB"] = float(F[mid].std() / rng_pt) if rng_pt > 0 else None
    out["auc"] = {}
    for a in auc_snrs:
        s = int(np.argmin(np.abs(snr_arr - a)))
        pt = auc_stat(F[s], F_mis[s])
        draws = np.array([auc_stat(F[s][idx[b]], F_mis[s][idx[b]]) for b in range(min(idx.shape[0], 300))])
        out["auc"][str(a)] = ci(draws, pt)
    return out


# ---------------------------------------------------------------------------------------------------------------------------
def pair_criteria(F: np.ndarray, F_mis: np.ndarray, snrs: list[float]) -> dict:
    """The three scale-free criteria of the amended rule for ONE (pair, seed), computed from per-item F averaged over draws.
    F, F_mis: (S, N). Returns auc_at_0dB, pooled_spearman, item_noise_over_range (point estimates)."""
    snr_arr = np.asarray(snrs, dtype=float)
    s0 = int(np.argmin(np.abs(snr_arr - 0.0)))
    mid = int(np.argmin(np.abs(snr_arr - 10.0)))
    means = F.mean(1)
    rng = means[-1] - means[0]
    y = np.repeat(snr_arr[:, None], F.shape[1], axis=1)
    return {"auc_at_0dB": float(auc_stat(F[s0], F_mis[s0])),
            "pooled_spearman": float(pd.Series(F.ravel()).corr(pd.Series(y.ravel()), method="spearman")),
            "item_noise_over_range": float(F[mid].std() / rng) if rng > 0 else float("inf")}


AMENDED_CRITERIA = (("auc_at_0dB", "higher"), ("pooled_spearman", "higher"), ("item_noise_over_range", "lower"))


def choose_final_variant(criteria: dict[str, dict[str, list[dict]]]) -> tuple[str, dict]:
    """The AMENDED rule, applied exactly as declared (validation split, scale-free criteria only).

    criteria[variant][pair] = list over seeds of pair_criteria(...) dicts. For each criterion: per pair take the mean over seeds, then the
    median over pairs; the variant that is better on at least 2 of the 3 criteria is the headline variant. Dynamic range, C and win counts
    are NOT inputs. Exactly two candidates are required; returns None only if exact ties leave neither with 2 criteria.
    """
    variants = list(criteria)
    assert len(variants) == 2, "the amended rule compares exactly two co-primary variants"
    medians = {v: {} for v in variants}
    for v in variants:
        for crit, _ in AMENDED_CRITERIA:
            per_pair = [np.mean([s[crit] for s in seeds]) for seeds in criteria[v].values()]
            medians[v][crit] = float(np.median(per_pair))
    wins = {v: 0 for v in variants}
    detail = {}
    for crit, better in AMENDED_CRITERIA:
        a, b = variants
        win = a if (medians[a][crit] > medians[b][crit]) == (better == "higher") else b
        if medians[a][crit] == medians[b][crit]:
            win = None
        detail[crit] = {"better": better, "winner": win, **{v: medians[v][crit] for v in variants}}
        if win:
            wins[win] += 1
    chosen = next((v for v in variants if wins[v] >= 2), None)  # None only if a tie leaves neither variant with 2 criteria
    return chosen, {"criteria": detail, "criteria_won": wins, "n_pairs": len(criteria[variants[0]])}
