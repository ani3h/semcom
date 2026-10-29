from pathlib import Path

import torch
import torch.nn.functional as F


class Centerer:
    """Per-domain mean centering of an anchor space: x -> normalize(x - mu).

    mu is fitted on TRAIN anchors only and then frozen; transform() never touches the fit
    statistics. Centering removes the shared offset that makes every pair of CLIP/CLAP
    embeddings look similar (random-pair cosine 0.77-0.95), so cosine becomes usable as a
    distance. Each domain (dataset / modality) gets its own centerer.
    """

    def __init__(self, mu: torch.Tensor | None = None, n_fit: int = 0):
        self.mu, self.n_fit = mu, n_fit

    def fit(self, train_anchors: torch.Tensor) -> "Centerer":
        self.mu = train_anchors.detach().float().mean(dim=0).cpu()
        self.n_fit = train_anchors.size(0)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mu is None:
            raise RuntimeError("Centerer.fit() must be called on train anchors first")
        return F.normalize(x.float() - self.mu.to(x.device), dim=-1)

    def state_dict(self) -> dict:
        return {"mu": self.mu, "n_fit": self.n_fit}

    @classmethod
    def from_state(cls, state: dict) -> "Centerer":
        return cls(state["mu"], state["n_fit"])

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path) -> "Centerer":
        return cls.from_state(torch.load(path))
