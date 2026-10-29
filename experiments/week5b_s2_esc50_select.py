"""Stage 2: apply the pre-registered ESC-50 head-regularisation rule and write the winner.

    python experiments/week5b_s2_esc50_select.py

Rule (declared before any Stage 2 seed ran): seed 42 only, both variants and both channels (4 cases); compare the best validation loss
of the baseline (dropout 0.1, wd 1e-4), R1 (0.3, 1e-3) and R2 (0.5, 1e-2) heads; the setting with the lowest MEAN RANK over the 4 cases
wins, ties go to baseline. The winner is then used for all three seeds.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("outputs/week5b/stage2/heads/esc50_audio")
SETTINGS = {"base": {"dropout": 0.1, "weight_decay": 1e-4}, "R1": {"dropout": 0.3, "weight_decay": 1e-3}, "R2": {"dropout": 0.5, "weight_decay": 1e-2}}


def main() -> None:
    cases, ranks = {}, {k: [] for k in SETTINGS}
    for variant in ("cos5", "cen"):
        for ch in ("awgn", "rayleigh"):
            vals = {k: json.loads((ROOT / ch / f"train_{variant}_s42_{k}.json").read_text()) for k in SETTINGS}
            order = sorted(SETTINGS, key=lambda k: vals[k]["best_val_total"])
            cases[f"{variant}/{ch}"] = {k: {"best_val_total": vals[k]["best_val_total"], "best_epoch": vals[k]["best_epoch"]} for k in SETTINGS}
            for k in SETTINGS:
                ranks[k].append(order.index(k) + 1)
    mean_rank = {k: float(np.mean(v)) for k, v in ranks.items()}
    best = min(mean_rank.values())
    winners = [k for k, v in mean_rank.items() if abs(v - best) < 1e-9]
    chosen = "base" if "base" in winners else winners[0]  # ties go to baseline
    out = {"rule": "lowest mean rank of best validation loss over (variant, channel) at seed 42; ties to baseline", "settings": SETTINGS, "cases": cases,
           "ranks": ranks, "mean_rank": mean_rank, "chosen": chosen, "chosen_setting": SETTINGS[chosen]}
    # seed 42 of the winning setting IS the head used for seed 42 (identical recipe): expose it under the standard name
    import shutil

    for variant in ("cos5", "cen"):
        for ch in ("awgn", "rayleigh"):
            for ext in (".pt", ".json"):
                src = ROOT / ch / (("train_" if ext == ".json" else "") + f"{variant}_s42_{chosen}{ext}")
                dst = ROOT / ch / (("train_" if ext == ".json" else "") + f"{variant}_s42{ext}")
                shutil.copyfile(src, dst)
    Path("outputs/week5b/stage2/esc50_setting.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"mean_rank": mean_rank, "chosen": chosen, "setting": SETTINGS[chosen]}))


if __name__ == "__main__":
    main()
