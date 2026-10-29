"""Statistics for the Stage 0 cross-modal gate: zero-shot classification, class-level retrieval, bootstrap CIs."""
import numpy as np
import torch

# ESC-50 classes mapped onto CIFAR-10 classes (pre-registered, before any audio->image number was computed).
# airplane/cat/dog/frog are exact name matches; the CIFAR-10 "bird" class is paired with every ESC-50 bird class.
ESC50_TO_CIFAR10 = {"airplane": "airplane", "cat": "cat", "dog": "dog", "frog": "frog",
                    "chirping_birds": "bird", "crow": "bird", "hen": "bird", "rooster": "bird"}


def topk_correct(scores: torch.Tensor, labels: torch.Tensor, k: int) -> np.ndarray:
    """(N, C) similarity scores -> per-sample 0/1 array: is the true class among the top-k."""
    true = scores.gather(1, labels.view(-1, 1))
    rank = 1 + (scores > true).sum(dim=1)
    return (rank <= k).float().numpy()


def balanced_accuracy(correct: np.ndarray, labels: np.ndarray) -> float:
    """Mean per-class accuracy (chance = 1/#classes even when classes are imbalanced)."""
    return float(np.mean([correct[labels == c].mean() for c in np.unique(labels)]))


def bootstrap_ci(correct: np.ndarray, labels: np.ndarray | None = None, balanced: bool = False, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap over samples (resampling within class when balanced): (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    stat = (lambda c, l: balanced_accuracy(c, l)) if balanced else (lambda c, l: float(c.mean()))
    point = stat(correct, labels)
    draws = np.empty(n_boot)
    if balanced:
        groups = [np.where(labels == c)[0] for c in np.unique(labels)]
        for b in range(n_boot):
            idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
            draws[b] = stat(correct[idx], labels[idx])
    else:
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            draws[b] = stat(correct[idx], None)
    return point, float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def class_prototypes(emb: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Unit-norm mean embedding per class, (n_classes, D)."""
    protos = torch.stack([emb[labels == c].mean(0) for c in range(n_classes)])
    return torch.nn.functional.normalize(protos, dim=-1)


def r_precision(queries: torch.Tensor, gallery: torch.Tensor, query_labels: torch.Tensor,
                gallery_labels: torch.Tensor) -> dict[str, float]:
    """Class-level query->gallery retrieval. P@1 and R-precision (precision at k = gallery items of the query's
    class, so 1.0 is perfect), averaged over queries."""
    sims = queries @ gallery.T
    order = sims.argsort(dim=1, descending=True)
    ranked = gallery_labels[order] == query_labels.view(-1, 1)
    r = torch.stack([(gallery_labels == c).sum() for c in query_labels]).clamp_min(1)
    pos = torch.arange(ranked.size(1)).view(1, -1)
    within_r = (pos < r.view(-1, 1)).float()
    return {"p@1": ranked[:, 0].float().mean().item(), "r_precision": ((ranked.float() * within_r).sum(1) / r).mean().item()}
