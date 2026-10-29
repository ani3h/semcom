"""Per-domain maps from raw anchor space (CLIP / CLAP) into the shared space S.

Every domain (dataset) has its own mean-centering fitted on TRAIN anchors only and cached with provenance. Audio domains
(speech, event audio) then pass through the seed-42 CLAP->CLIP bridge; image and text domains are native to the CLIP space.
S anchors are therefore: centered CLIP (image, text) and bridged, centered CLAP (audio).

    domain = get_domain("esc50_audio")
    anchors_S = domain.to_S(raw_clap_anchors)
"""
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .anchors import anchor_path, load_anchors
from .bridge import TextBridge
from .correction import Centerer

ANCHOR_ROOT = "data_cache/anchors"
SPACE_ROOT = "outputs/week5b/space"
BRIDGE_DIR = "outputs/week5b/bridge"

# task -> (where the centering statistics are fitted, whether audio goes through the bridge)
DOMAIN_SPECS = {
    "cifar10": {"fit": "train anchors", "bridge": False},
    "mnist": {"fit": "train anchors", "bridge": False},
    "text": {"fit": "train anchors", "bridge": False},  # Europarl, CLIP text
    "audio": {"fit": "train anchors", "bridge": True},  # Valentini speech, CLAP
    "caption_text": {"fit": "train captions (Stage 0 centerer)", "bridge": False},
    "esc50_audio": {"fit": "ESC-50 folds 1-3", "bridge": True},
}
RAW_SPACE = {"cifar10": "clip", "mnist": "clip", "text": "clip", "caption_text": "clip", "audio": "clap", "esc50_audio": "clap"}


@dataclass
class Domain:
    name: str
    centerer: Centerer
    bridge: TextBridge | None
    fit_on: dict

    def to_S(self, raw: torch.Tensor) -> torch.Tensor:
        x = self.centerer.transform(raw)
        if self.bridge is not None:
            self.bridge.to(x.device)
            with torch.no_grad():
                x = self.bridge(x)
        return x


def load_bridge(bridge_dir=BRIDGE_DIR, seed: int = 42) -> TextBridge:
    kind = json.loads((Path(bridge_dir) / "results.json").read_text())["seeds"][str(seed)]["chosen"]
    ck = torch.load(Path(bridge_dir) / f"{kind}_s{seed}.pt")
    bridge = TextBridge(ck["kind"], hidden=ck["hidden"])
    bridge.load_state_dict(ck["state"])
    return bridge.eval().requires_grad_(False)


def esc50_train_anchors(anchor_root=ANCHOR_ROOT) -> torch.Tensor:
    """CLAP anchors of the ESC-50 TRAIN folds (1-3) only; asserts that no validation/test fold leaks in."""
    blob = load_anchors(anchor_path(anchor_root, "esc50", "all", "_clap_16k"))
    sel = blob["folds"] <= 3
    assert int(sel.sum()) == 1200 and int((blob["folds"][sel] >= 4).sum()) == 0, "ESC-50 centering must use folds 1-3 only"
    return blob["anchors"][sel]


def get_domain(task: str, anchor_root=ANCHOR_ROOT, space_root=SPACE_ROOT, bridge_dir=BRIDGE_DIR, bridge_seed: int = 42) -> Domain:
    """Load (or fit once on train data and cache) the centering for `task`'s domain and attach the bridge for audio."""
    if task not in DOMAIN_SPECS:
        raise ValueError(f"unknown domain {task!r}; choose from {sorted(DOMAIN_SPECS)}")
    path = Path(space_root) / f"centerer_{task}.pt"
    if path.exists():
        centerer = Centerer.load(path)
    else:
        if task == "caption_text":
            centerer = Centerer.load(Path(bridge_dir) / "centerer_clip_text.pt")
        elif task == "esc50_audio":
            centerer = Centerer().fit(esc50_train_anchors(anchor_root))
        else:
            centerer = Centerer().fit(load_anchors(anchor_path(anchor_root, task, "train"))["anchors"])
        centerer.save(path)
    bridge = load_bridge(bridge_dir, bridge_seed) if DOMAIN_SPECS[task]["bridge"] else None
    return Domain(task, centerer, bridge, {"fit": DOMAIN_SPECS[task]["fit"], "n_fit": centerer.n_fit,
                                           "bridge": f"seed {bridge_seed}" if bridge is not None else None})
