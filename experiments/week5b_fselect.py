"""Stage 1, step 4: Fidelity statistics for every head variant, and the variant choice for Stage 2.

    python experiments/week5b_fselect.py                       # all six tasks
    python experiments/week5b_fselect.py 'fselect_tasks=[cifar10]'   # one task (writes its own file; merge in the report)

For every task x channel x head variant x split (val, test): F_abs (headline), F_ret, the clean ceiling C, and the raw / centered cosine
baselines, each with mean per SNR (strictly increasing?), dynamic range, pooled Spearman with SNR, the SNR at which it reaches 90% of
its clean value, and matched-vs-mismatched AUC. The variant is chosen on VALIDATION F_abs only, by the pre-registered rule in
semcom.common_space.fidelity.choose_variant, on a single head seed. Test statistics are reported, never used to choose.
"""
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from semcom.channel.base import Channel
from semcom.common_space.adapters import make_adapter
from semcom.common_space.anchors import anchor_path, load_anchors
from semcom.common_space.domains import get_domain
from semcom.common_space.fidelity import CANDIDATES, analyze, candidate_scores, derangement, make_reference
from semcom.common_space.loader import load_head_space, load_week4, make_dataset

ALL_TASKS = ("cifar10", "mnist", "text", "audio", "caption_text", "esc50_audio")
CHUNK = {"esc50_audio": 100, "audio": 250}


class Clean(Channel):
    """Identity channel: the noise-free (infinite-SNR) reference."""

    def forward(self, x, snr_db):
        return x


def pick_device(name: str) -> torch.device:
    return torch.device(name) if name != "auto" else torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def embed(space, w4, items, channel, snr, seed, device, chunk) -> torch.Tensor:
    torch.manual_seed(seed)
    out = []
    for s in range(0, len(items), chunk):
        batch = w4.spec.to_device(w4.spec.collate(items[s:s + chunk]), device)
        out.append(space.embed_received(batch, channel, snr).cpu())
    return torch.cat(out)


@hydra.main(config_path="../configs", config_name="week5b", version_base=None)
def main(cfg: DictConfig) -> None:
    device = pick_device(cfg.device)
    space_dir, bridge_dir = Path(cfg.out_root) / "space", Path(cfg.out_root) / "bridge"
    out_dir = Path(cfg.out_root) / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(cfg.get("fselect_tasks", ALL_TASKS))
    variants = OmegaConf.to_container(cfg.stage1.variants)
    grid, draws, nboot = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0], int(cfg.stage1.noise_draws), int(cfg.stage1.n_boot)
    for task in tasks:
        res = {"task": task, "channels": {}, "missing": []}
        domain = get_domain(task, cfg.anchor_root, space_dir, bridge_dir)
        for ch_name in ("awgn", "rayleigh"):
            w4 = load_week4(task, ch_name, cfg.w4_root, device)
            channel = hydra.utils.instantiate(OmegaConf.load(f"configs/channel/{ch_name}.yaml"))
            adapter = make_adapter(task, None, w4.spec)
            res["channels"][ch_name] = {}
            for split in ("val", "test"):
                ds = make_dataset(w4, split, cfg.data_root, {"prompt_repeat": 0} if task == "caption_text" else None)
                n = min(len(ds), int(cfg.stage1.f_items))
                items = [ds[i] for i in range(n)]
                a_raw = load_anchors(anchor_path(cfg.anchor_root, task, split))["anchors"][:n]
                a_S = domain.to_S(a_raw)
                perm = derangement(n)
                for vname, v in variants.items():
                    path = Path(cfg.out_root).parent / "week5" / task / ch_name / f"{v['file']}.pt"
                    if not path.exists():
                        res["missing"].append(str(path))
                        continue
                    space, meta = load_head_space(path, w4, adapter, device)
                    centered = meta["target"] == "centered"
                    to_S = (lambda z: z) if centered else domain.to_S
                    a_train = a_S if centered else a_raw  # anchors in the space the head was trained in
                    z_clean = embed(space, w4, items, Clean(), 0.0, 0, device, CHUNK.get(task, 1000))
                    ref = make_reference(to_S(z_clean), a_S)
                    clean = {c: s.numpy() for c, s in candidate_scores(z_clean, a_train, to_S(z_clean), a_S, ref).items()}
                    per = {c: {} for c in CANDIDATES}
                    mis = {c: {} for c in CANDIDATES}
                    for snr in grid:
                        rows = {c: [] for c in CANDIDATES}
                        mrows = {c: [] for c in CANDIDATES}
                        for d in range(draws):
                            z = embed(space, w4, items, channel, snr, int(cfg.seed) * 10_000 + 1000 * d + int(snr), device, CHUNK.get(task, 1000))
                            zs = to_S(z)
                            for c, s in candidate_scores(z, a_train, zs, a_S, ref).items():
                                rows[c].append(s.numpy())
                            for c, s in candidate_scores(z, a_train, zs, a_S, ref, perm).items():
                                mrows[c].append(s.numpy())
                        for c in CANDIDATES:
                            per[c][snr], mis[c][snr] = np.stack(rows[c]), np.stack(mrows[c])
                    stats = {c: analyze(per[c], mis[c], clean[c], n_boot=nboot) for c in CANDIDATES}
                    res["channels"][ch_name].setdefault(split, {})[vname] = {
                        "target": meta["target"], "reference": {"m_clean": ref.m_clean, "b_aa": ref.b_aa, "C": ref.ceiling}, "stats": stats, "n_items": n}
                    s = stats["F_abs"]
                    print(f"[fselect] {task}/{ch_name}/{split}/{vname}: F_abs "
                          + " ".join(f"{s['means'][str(g)]:.3f}" for g in grid) + f" | clean {s['clean']:.3f} C {ref.ceiling:.3f} | increasing {s['strictly_increasing']}", flush=True)
        (out_dir / f"fselect_{task}.json").write_text(json.dumps(res, indent=2))
    print("[fselect] done", flush=True)


if __name__ == "__main__":
    main()
