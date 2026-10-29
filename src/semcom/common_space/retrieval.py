import torch


def recall_at_k(queries: torch.Tensor, gallery: torch.Tensor, ks=(1, 5, 10)) -> dict[str, float]:
    """Row i of `queries` should retrieve row i of `gallery` (cosine similarity on
    L2-normalised embeddings). Rank = 1 + number of gallery items scoring strictly higher
    than the true match."""
    sims = queries @ gallery.T
    true = sims.diag().unsqueeze(1)
    rank = 1 + (sims > true).sum(dim=1)
    return {f"recall@{k}": (rank <= k).float().mean().item() for k in ks if k <= gallery.size(0)}


def class_topk(embeddings: torch.Tensor, prompt_embeddings: torch.Tensor, labels: torch.Tensor,
               ks=(1, 5)) -> dict[str, float]:
    """Image->text retrieval at class level: each embedding retrieves among the class prompts."""
    sims = embeddings @ prompt_embeddings.T
    true = sims.gather(1, labels.view(-1, 1))
    rank = 1 + (sims > true).sum(dim=1)
    return {f"top{k}": (rank <= k).float().mean().item() for k in ks if k <= prompt_embeddings.size(0)}


def mean_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a * b).sum(-1).mean().item()


def random_pair_cosine(emb: torch.Tensor, n_pairs: int = 20000, seed: int = 0) -> float:
    """Mean cosine between random *unrelated* pairs: the baseline similarity of a space.
    CLIP and CLAP spaces differ here, which Week 6 needs for scale normalisation."""
    g = torch.Generator().manual_seed(seed)
    n = emb.size(0)
    i = torch.randint(0, n, (n_pairs,), generator=g)
    j = torch.randint(0, n - 1, (n_pairs,), generator=g)
    j = j + (j >= i).long()  # j != i
    return (emb[i] * emb[j]).sum(-1).mean().item()


def unrelated_cosine(emb: torch.Tensor, anchors: torch.Tensor) -> float:
    """Mean cosine between an embedding and the anchors of *other* items (off-diagonal pairs):
    the baseline an own-anchor cosine has to be judged against. Needs len(emb) > 1."""
    n = emb.size(0)
    sims = emb @ anchors.T
    return ((sims.sum() - sims.diag().sum()) / (n * (n - 1))).item()


def alignment(emb: torch.Tensor, anchors: torch.Tensor) -> dict[str, float]:
    """cos_to_anchor (own source), cos_unrelated (other items' sources) and their difference, the
    margin, which is what a normalised Fidelity should be built on (cosine scales differ per space)."""
    own, other = mean_cosine(emb, anchors), unrelated_cosine(emb, anchors)
    return {"cos_to_anchor": own, "cos_unrelated": other, "margin": own - other}
