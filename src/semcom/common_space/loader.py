from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from .heads import build_head
from .learned import LearnedHeadSpace


@dataclass
class Week4Model:
    task: str
    channel: str
    spec: object
    model: torch.nn.Module
    cfg: object


def load_week4(task: str, channel: str, root: str = "outputs/week4", device="cpu") -> Week4Model:
    """Rebuild a trained Week 4 transceiver from its checkpoint: eval mode, frozen, on `device`."""
    ck = torch.load(Path(root) / task / channel / "best.pt", weights_only=False, map_location="cpu")
    cfg = OmegaConf.create(ck["cfg"])
    spec = hydra.utils.instantiate(cfg.spec, _recursive_=False)
    spec.load_state_dict(ck["spec"])
    model = spec.build_model()
    model.load_state_dict(ck["model"])
    model.eval().requires_grad_(False).to(device)
    return Week4Model(task, channel, spec, model, cfg)


def make_dataset(w4: Week4Model, split: str, data_root: str | None = None, overrides: dict | None = None):
    """The dataset a Week 4 transceiver was trained on. `overrides` replaces dataset-config keys (e.g. prompt_repeat=0 so
    heads never see the class prompts the text transceiver was trained on)."""
    extra = {} if data_root is None else {"root": str(data_root)}
    extra.update(overrides or {})
    return hydra.utils.instantiate(w4.cfg.dataset, split=split, **extra)


class AnchoredData:
    """Dataset items paired with their cached anchors, fetched by index (no DataLoader).

    Anchors are keyed by dataset index (splits are seeded and cached on disk). For audio
    training the loader's random crops cannot be cached, so `crops=(utterance_idx, starts)`
    pins a fixed set of crops, one anchor each."""

    def __init__(self, ds, anchors: torch.Tensor, crops: tuple | None = None):
        self.ds, self.anchors, self.crops = ds, anchors, crops
        expected = len(ds) if crops is None else len(crops[0])
        if anchors.size(0) != expected:
            raise ValueError(f"{anchors.size(0)} cached anchors but {expected} items: stale cache?")

    def __len__(self) -> int:
        return self.anchors.size(0)

    def fetch(self, idxs) -> tuple[list[dict], torch.Tensor]:
        idxs = [int(i) for i in idxs]
        if self.crops is None:
            items = [self.ds[i] for i in idxs]
        else:
            utt, starts = self.crops
            items = [{"input": self.ds.get_crop(int(utt[i]), int(starts[i])), "label": None, "raw": None}
                     for i in idxs]
        return items, self.anchors[idxs]


@torch.no_grad()
def clean_symbol_sigma(w4: Week4Model, adapter, dataset, n: int = 2000, batch_size: int = 500, device="cpu") -> float:
    """Std of the transceiver's clean (pre-channel) symbols over the first n items; the constant the
    head's input conditioner divides by (real tokens only for text)."""
    total, total_sq, count = 0.0, 0.0, 0
    for s in range(0, min(n, len(dataset)), batch_size):
        items = [dataset[i] for i in range(s, min(s + batch_size, n, len(dataset)))]
        batch = w4.spec.to_device(w4.spec.collate(items), device)
        sym = w4.model.encode(adapter.tx_input(batch))
        mask = adapter.head_side(batch).get("mask")
        sym = sym[mask] if mask is not None else sym
        total += sym.sum().item()
        total_sq += sym.pow(2).sum().item()
        count += sym.numel()
    mean = total / count
    return max((total_sq / count - mean**2) ** 0.5, 1e-6)


def load_head_space(path, w4: Week4Model, adapter, device) -> tuple[LearnedHeadSpace, dict]:
    """A trained head on its frozen Week 4 transceiver. Returns (space, checkpoint metadata incl. "target": raw | centered)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = build_head(ck["head_cfg"]["kind"], sigma=ck["sigma"], **{k: v for k, v in ck["head_cfg"].items() if k != "kind"})
    head.load_state_dict(ck["head"])
    return LearnedHeadSpace(w4.model, head.to(device).eval(), adapter), {"target": ck.get("target", "raw"), "epoch": ck.get("epoch")}
