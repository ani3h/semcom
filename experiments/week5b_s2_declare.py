"""Write the Stage 2 protocol record BEFORE any Stage 2 seed is trained or evaluated.

    python experiments/week5b_s2_declare.py

The record is timestamped and hashes the Stage 1 variant choice and the held-out prompt templates, so the report can show that the
post-hoc amendment and every selection rule below pre-date the Stage 2 numbers. It refuses to overwrite an existing record.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from semcom.data.prompts import HELDOUT_TEMPLATES, TRAINING_TEMPLATES, heldout_hash

OUT = Path("outputs/week5b/stage2/protocol_declared.json")

RECORD = {
    "amendment": {
        "kind": "POST-HOC AMENDMENT, declared before any Stage 2 seed was run",
        "reason": "Stage 1 compared the head variants on one seed. All three were strictly increasing on all 12 (task, channel) pairs, so the "
                  "pre-registered rule was decided by its tie-breaker, the dynamic range of F_abs, which depends on the scale of F_abs (the clean "
                  "ceiling C) and picked cos5. The scale-free criteria (AUC at 0 dB, pooled Spearman, item-noise/range) favoured cen. After seeing that "
                  "single-seed result, cos5 and cen are carried into Stage 2 as co-primary variants.",
        "final_variant_rule": {
            "split": "validation",
            "candidates": ["cos5", "cen"],
            "criteria": {"auc_at_0dB": "higher is better; matched-vs-mismatched AUC of F_abs at 0 dB",
                         "pooled_spearman": "higher is better; Spearman between F_abs and SNR pooled over items, draws and SNR points",
                         "item_noise_over_range": "lower is better; standard deviation over items of the draw-averaged F_abs at 10 dB divided by "
                                                  "(mean F_abs at 25 dB minus mean F_abs at 0 dB)"},
            "aggregation": "for each criterion, take each (task, channel) pair's mean over the 3 head seeds, then the median over the 12 pairs; "
                           "the variant that is better on at least 2 of the 3 criteria is the final headline variant",
            "context_only_not_selection": ["dynamic range of F_abs", "clean ceiling C", "per-pair win counts", "strictly-increasing counts"],
            "applied_exactly_as_written_after_3_seed_numbers_exist": True,
        },
        "original_rule_output_kept_for_transparency": "the Stage 1 pre-registered rule (strictly increasing pairs, then median dynamic range, then median AUC "
                                                      "at 0 dB) is re-applied to the 3-seed means and reported unchanged next to the amended rule",
    },
    "head_training": {
        "seeds": [42, 43, 44], "seed_0_means": 42, "variants": {"cos5": "raw anchors, cosine weight 5", "cen": "centered S anchors, cosine weight 0.5"},
        "image_heads": "40 epochs, cosine schedule (unchanged); seed 42 reuses the Stage 1 heads, seeds 43 and 44 are new",
        "text_and_speech_heads": "Europarl text, caption text, Valentini speech: up to 80 epochs, best-validation checkpoint, early stopping with patience 8, "
                                 "plus a plateau learning-rate decay (factor 0.3 after 3 epochs without improvement) so the heads are annealed before stopping",
        "esc50_heads": {"settings": {"baseline": {"dropout": 0.1, "weight_decay": 1e-4}, "R1": {"dropout": 0.3, "weight_decay": 1e-3},
                                      "R2": {"dropout": 0.5, "weight_decay": 1e-2}},
                        "comparison": "seed 42 only, both variants and both channels (4 cases), best validation loss (identical objective for all settings); "
                                      "the setting with the lowest mean rank over the 4 cases wins, ties go to baseline; the winner is applied to all 3 seeds",
                        "other": "100 epochs, cosine schedule, best-validation checkpoint (as in Stage 1)"},
        "no_encoder_fine_tuning": True,
    },
    "evaluation": {
        "noise_draws_per_item": 5, "bootstrap": "percentile, 95%, over items", "seed_spread": "mean +/- sd over the 3 head seeds, reported separately",
        "snr_grid_db": {"f_and_within_modality": [0, 5, 10, 15, 20, 25], "cross_modal": [0, 5, 10, 15, 25]},
        "rayleigh_soft_clip": "4*tanh(x/4) on head inputs (unchanged)",
        "audio_image_pair": "class level over the 5 shared ESC-50/CIFAR-10 classes, balanced accuracy, chance 0.2",
        "transceivers_single_seed": "the Week 4 and Stage 1 transceivers are single-seed, so every interval excludes transceiver-training variance",
        "text_prompt_optimism": {"training_templates": TRAINING_TEMPLATES, "heldout_templates": HELDOUT_TEMPLATES, "heldout_templates_sha256": heldout_hash(),
                                 "headline": "held-out paraphrase prompts (b) are the headline for any claim involving transmitted text"},
        "imagebind_ablation": "optional, run last after everything else is reported, scratch venv, nothing derived from its weights is shipped",
    },
}


def main() -> None:
    if OUT.exists():
        sys.exit(f"{OUT} already exists; the declaration is written once")
    s1 = Path("outputs/week5b/stage1/variant_choice.json")
    RECORD["declared_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    RECORD["stage1_variant_choice_sha256"] = hashlib.sha256(s1.read_bytes()).hexdigest()
    RECORD["stage1_variant_choice"] = json.loads(s1.read_text())["chosen"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(RECORD, indent=2))
    print(f"wrote {OUT} at {RECORD['declared_at_utc']}")


if __name__ == "__main__":
    main()
