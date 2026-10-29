"""Stage 2 evaluation functions on cached embeddings. Pure torch / numpy so they can be unit-tested on synthetic data.

Shapes: z is (D noise draws, S SNR points, N items, 512) float16; anchors are (N, 512). "Per-item" arrays are (S, N) after averaging over the
D draws. Each item is the unit of bootstrap resampling; head seeds and noise draws are averaged per item first.
"""
import numpy as np
import torch

from .fidelity import anchor_baseline, derangement
from .stage2_stats import (balanced_acc, ci, curve_balanced_acc_ci, mean_ci_curve, pair_criteria, seed_spread)

FCANDS = ("F_abs", "F_ret", "centered cosine", "raw cosine")


# ---------------------------------------------------------------------------------------------------------------------------
# Fidelity from embeddings
def f_arrays(z: torch.Tensor, z_clean: torch.Tensor, a_S: torch.Tensor, raw: dict | None = None) -> dict:
    """Per-item Fidelity candidates from embeddings in S.

    raw: optional {"cos": (D,S,N), "cos_mis": (D,S,N), "cos_clean": (N,)} for heads trained against raw anchors (the raw-cosine baseline in the
    head's own space); heads trained in S use their centered cosine, and approach B has no raw baseline (omitted).
    Returns {cand: {"F": (S,N), "F_mis": (S,N), "clean": (N,)}} plus {"ref": {"m_clean", "b_aa", "C"}}. F_mis pairs every item with the same
    fixed wrong anchor (derangement), the negatives of the matched-vs-mismatched AUC.
    """
    D, S, N, _ = z.shape
    a = a_S.float()
    perm = derangement(N)
    ar = torch.arange(N)
    own, base, mis = np.zeros((S, N)), np.zeros((S, N)), np.zeros((S, N))
    for s in range(S):
        for d in range(D):
            sims = z[d, s].float() @ a.T
            dg = sims.diag()
            own[s] += dg.numpy() / D
            base[s] += ((sims.sum(1) - dg) / (N - 1)).numpy() / D
            mis[s] += sims[ar, perm].numpy() / D
    simc = z_clean.float() @ a.T
    own_c = simc.diag().numpy()
    base_c = ((simc.sum(1) - simc.diag()) / (N - 1)).numpy()
    b_aa = anchor_baseline(a)
    m_clean = float((own_c - base_c).mean())
    margin, margin_mis, margin_c = own - base, mis - base, own_c - base_c
    out = {"F_abs": {"F": margin / (1 - b_aa), "F_mis": margin_mis / (1 - b_aa), "clean": margin_c / (1 - b_aa)},
           "F_ret": {"F": margin / m_clean, "F_mis": margin_mis / m_clean, "clean": margin_c / m_clean},
           "centered cosine": {"F": own, "F_mis": mis, "clean": own_c}}
    if raw is not None:
        out["raw cosine"] = {"F": raw["cos"].mean(0).numpy(), "F_mis": raw["cos_mis"].mean(0).numpy(), "clean": raw["cos_clean"].numpy()}
    out["ref"] = {"m_clean": m_clean, "b_aa": float(b_aa), "C": m_clean / (1 - b_aa)}
    return out


def seed_summary(F: np.ndarray, F_mis: np.ndarray, clean: np.ndarray, snrs: list[float]) -> dict:
    """Point summaries of ONE head seed (for the seed-to-seed spread): mean per SNR, clean mean, range, the three scale-free criteria and the
    SNR at 90% of clean."""
    from .fidelity import snr_at_fraction

    means = F.mean(1)
    crit = pair_criteria(F, F_mis, snrs)
    snr90 = snr_at_fraction(list(map(float, snrs)), list(means), float(clean.mean()))
    return {"means": means.tolist(), "clean": float(clean.mean()), "dynamic_range": float(means[-1] - means[0]), **crit,
            "snr_at_90pct_clean": snr90["snr"], "strictly_increasing": bool(np.all(np.diff(means) > 0))}


# ---------------------------------------------------------------------------------------------------------------------------
# Class-level retrieval from embeddings
def topk_correct(scores: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    """(..., N, C) scores, (N,) labels -> (..., N) 1/0: is the true class among the k best scores."""
    true = scores.gather(-1, labels.expand(*scores.shape[:-2], -1).unsqueeze(-1))
    return ((scores > true).sum(-1) + 1 <= k).float()


def class_prototypes(emb: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    protos = torch.stack([emb[labels == c].float().mean(0) for c in range(n_classes)])
    return torch.nn.functional.normalize(protos, dim=-1)


def class_correct(z: torch.Tensor, protos: torch.Tensor, labels: torch.Tensor, ks=(1, 5), sel: torch.Tensor | None = None) -> dict[int, np.ndarray]:
    """Per-item class-retrieval correctness at every (SNR), averaged over the noise draws. z: (D,S,N,512) -> {k: (S,N)}.
    sel: optional item subset (indices); labels must then index the SELECTED items."""
    zs = z if sel is None else z[:, :, sel]
    scores = zs.float() @ protos.float().T  # (D,S,N,C)
    return {k: topk_correct(scores, labels, k).mean(0).numpy() for k in ks if k <= protos.size(0)}


def class_correct_static(emb: torch.Tensor, protos: torch.Tensor, labels: torch.Tensor, ks=(1, 5), sel: torch.Tensor | None = None) -> dict[int, np.ndarray]:
    """Same for a noise-free embedding (N,512) (the clean-anchor ceiling and the noise-free transceiver): {k: (N,)}."""
    e = emb if sel is None else emb[sel]
    scores = e.float() @ protos.float().T
    return {k: topk_correct(scores, labels, k).numpy() for k in ks if k <= protos.size(0)}


def self_retrieval_arrays(z: torch.Tensor, a_S: torch.Tensor, ks=(1, 5)) -> dict[str, np.ndarray]:
    """Per-item self-retrieval of the item's own clean anchor among the N anchors: recall@k indicators, cosine, margin -> (S,N) each."""
    D, S, N, _ = z.shape
    a = a_S.float()
    out = {f"recall@{k}": np.zeros((S, N)) for k in ks if k <= N}
    out.update({"cos_to_anchor": np.zeros((S, N)), "margin": np.zeros((S, N))})
    for s in range(S):
        for d in range(D):
            sims = z[d, s].float() @ a.T
            dg = sims.diag().unsqueeze(1)
            rank = 1 + (sims > dg).sum(1)
            for k in ks:
                if k <= N:
                    out[f"recall@{k}"][s] += (rank <= k).float().numpy() / D
            out["cos_to_anchor"][s] += dg.squeeze(1).numpy() / D
            out["margin"][s] += (dg.squeeze(1) - (sims.sum(1) - dg.squeeze(1)) / (N - 1)).numpy() / D
    return out


def self_retrieval_static(z_clean: torch.Tensor, a_S: torch.Tensor, ks=(1, 5)) -> dict[str, np.ndarray]:
    return {k: v[0] for k, v in self_retrieval_arrays(z_clean.unsqueeze(0).unsqueeze(0), a_S, ks).items()}


# ---------------------------------------------------------------------------------------------------------------------------
def summarize(per_seed: list[np.ndarray], idx: np.ndarray, labels: np.ndarray | None = None, balanced: bool = False) -> dict:
    """Curve statistics of a per-item score (S,N) measured under one or more head seeds.

    point/CI: mean over items of the seed-averaged per-item score, percentile bootstrap over items (balanced accuracy: class-stratified).
    spread: mean and sd over the per-seed curve means, reported separately (never mixed into the CI)."""
    avg = np.mean(per_seed, axis=0)
    if balanced:
        curve = curve_balanced_acc_ci(avg, labels, n_boot=idx.shape[0])
        per = np.array([[balanced_acc(x[s], labels) for s in range(x.shape[0])] for x in per_seed])
    else:
        curve = mean_ci_curve(avg, idx)
        per = np.array([x.mean(1) for x in per_seed])
    out = {"curve": curve, "n_items": int(avg.shape[1]), "n_seeds": len(per_seed)}
    if len(per_seed) > 1:
        out["spread"] = [seed_spread(per[:, s]) for s in range(per.shape[1])]
    return out


def summarize_static(x: np.ndarray, idx: np.ndarray, labels: np.ndarray | None = None, balanced: bool = False) -> dict:
    """A single-value reference (clean ceiling, noise-free transceiver): point + CI."""
    if balanced:
        c = curve_balanced_acc_ci(x[None], labels, n_boot=idx.shape[0])[0]
    else:
        c = ci(x[idx].mean(-1), x.mean())
    return {"value": c, "n_items": int(x.shape[0])}


def paired_difference(per_seed_a: list[np.ndarray], per_seed_b: list[np.ndarray], idx: np.ndarray, labels: np.ndarray | None = None, balanced: bool = False) -> list[dict]:
    """Paired difference A - B of a per-item score at every SNR: the same items and the same channel noise, so the item-level difference is
    resampled (balanced accuracy: class-stratified). A is averaged over its head seeds, B over its own (a single deterministic decoder)."""
    diff = np.mean(per_seed_a, axis=0) - np.mean(per_seed_b, axis=0)
    if balanced:
        return curve_balanced_acc_ci(diff, labels, n_boot=idx.shape[0])
    return mean_ci_curve(diff, idx)
