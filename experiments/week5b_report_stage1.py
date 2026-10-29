"""Generate docs/week5b_stage1.md from the Stage 1 results files (no hand-copied numbers) and write the variant choice for Stage 2.

    python experiments/week5b_report_stage1.py
"""
import json
from pathlib import Path

import numpy as np
import torch

from semcom.common_space.fidelity import choose_variant

ROOT = Path("outputs/week5b")
HEADS = Path("outputs/week5")
OUT = Path("docs/week5b_stage1.md")
TASKS = ("cifar10", "mnist", "text", "audio", "caption_text", "esc50_audio")
NAMES = {"cifar10": "CIFAR-10 image", "mnist": "MNIST image", "text": "Europarl text", "audio": "Valentini speech",
         "caption_text": "Caption text (new)", "esc50_audio": "ESC-50 event audio (new)"}
VARIANTS = {"cos": "cos 0.5 (main, raw targets)", "cos5": "cos 5 (ablation, raw targets)", "cen": "cos 0.5 toward the centered S anchor"}
CANDS = ("F_abs", "F_ret", "centered cosine", "raw cosine")
GRID = ["0.0", "5.0", "10.0", "15.0", "20.0", "25.0"]


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def snr90(d):
    return {"interpolated": f"{d['snr']:.1f} dB", "at_or_below_lowest_grid_point": "≤ 0 dB", "not_reached_by_grid_max": "> 25 dB", "undefined": "n/a"}[d["status"]]


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def val_curve(path: Path) -> list[float]:
    """Validation total loss per epoch parsed from a head-training log (empty if the log is missing)."""
    import re

    if not path.exists():
        return []
    return [float(m.group(1)) for m in re.finditer(r"epoch\s+\d+/\d+ .*?val@\S+ total ([0-9.]+)", path.read_text())]


def head_log(task: str, channel: str, variant: str) -> Path:
    w, tgt = {"cos": ("0_5", "raw"), "cos5": ("5", "raw"), "cen": ("0_5", "centered")}[variant]
    new = ROOT / "logs" / f"head1_{task}_{channel}_target_{tgt}_loss_cos_weight_{w}.log"
    old = HEADS / "logs" / f"head_{task}_{channel}_cw{w.replace('_', '.')}.log"
    return new if new.exists() else old


def main() -> None:
    sanity = json.loads((ROOT / "stage1" / "sanity.json").read_text())
    space = json.loads((ROOT / "space" / "results.json").read_text())
    fsel = {t: json.loads((ROOT / "stage1" / f"fselect_{t}.json").read_text()) for t in TASKS if (ROOT / "stage1" / f"fselect_{t}.json").exists()}
    L = []

    # ------------------------------------------------------------------ selection (pure function of the validation results)
    val_stats = {v: {} for v in VARIANTS}
    for t, r in fsel.items():
        for ch, splits in r["channels"].items():
            for v, d in splits.get("val", {}).items():
                val_stats[v][f"{t}/{ch}"] = d["stats"]["F_abs"]
    complete = {v: p for v, p in val_stats.items() if len(p) == len(TASKS) * 2}
    chosen, summary = choose_variant(complete) if len(complete) >= 2 else (None, {})
    if chosen:
        (ROOT / "stage1" / "variant_choice.json").write_text(json.dumps({"chosen": chosen, "rule": "most (task,channel) pairs with strictly increasing mean "
                                                                         "F_abs on validation, then median dynamic range, then median AUC at 0 dB", "single_head_seed": True,
                                                                         "summary": summary}, indent=2))

    # ------------------------------------------------------------------ header and verdict
    ver = sanity["verdict"]
    L.append("# Week 5b, Stage 1: new transceivers, shared-space anchors, heads and Fidelity selection\n")
    L.append(f"**Transceiver sanity gate: {'PASS (proceed)' if sanity['proceed'] else 'STOP'}.** " +
             "; ".join(f"{k}: " + ", ".join(f"{m} {v[m]:.3f}" for m in v if m != 'broken') + (" (BROKEN)" if v["broken"] else " (ok)") for k, v in ver.items()) +
             ". **Variant chosen for Stage 2: " + (f"`{chosen}` ({VARIANTS[chosen]})" if chosen else "not yet determined (F selection incomplete)") + "**, chosen on a single head seed. "
             "All numbers below are generated from `outputs/week5b/{stage1,space}/*.json`.\n")
    L.append("Both new transceivers are single-seed, so every interval in this document and in Stage 2 excludes transceiver-training variance.\n")

    # ------------------------------------------------------------------ text vocabulary
    vr = sanity["tasks"]["caption_text"]["awgn"]["vocab_report"]
    L.append("## 1. Caption-domain text transceiver: vocabulary\n")
    L.append(table(["item", "value"], [["vocabulary size (incl. 4 special tokens)", vr["vocab_size"]], ["training sentences (caption train split minus a 5,000-caption monitoring hold-out, plus prompt sentences)", vr["n_training_sentences"]],
                                       ["distinct class-prompt words forced into the vocabulary", vr["n_required_prompt_words"]],
                                       ["of those, absent from the training sentences", vr["prompt_words_absent_from_training_sentences"]]] +
                  [[f"`<unk>` tokens in prompt set `{k}` ({vr['prompt_set_sizes'][k]} prompts)", v] for k, v in vr["unk_per_prompt_set"].items()]))
    L.append("\nThe vocabulary is the union of the words seen at least 3 times in the caption train split and every word of every class-prompt set (CIFAR-10, CIFAR-100, MNIST, ESC-50 "
             "with all three templates); training asserts zero `<unk>` on every prompt set. The held-out caption splits (corpus val and test) were not used for the vocabulary or for any "
             "training or checkpoint selection of the transceiver. **Deviation:** the 280 distinct prompt sentences were also added to the transceiver's training split (×10, train split only), "
             "because many prompt words occur fewer than 3 times, or never, in the captions, and a vocabulary entry that is never trained cannot be transmitted. Heads and anchors never see the prompts. "
             "Consequence: prompt sets are *in the training distribution* of the text transceiver, so prompt recovery below is optimistic for unseen text.\n")

    # ------------------------------------------------------------------ sanity tables
    L.append("## 2. Transceiver sanity gate (reconstruction quality per SNR and channel)\n")
    th = sanity["thresholds"]
    L.append(f"Pre-registered stop thresholds (judgement calls, fixed before any result; validation split, 25 dB AWGN): text BLEU-1 ≥ {th['text']['min_bleu1']} and token accuracy ≥ {th['text']['min_token_acc']}; "
             f"audio SI-SDR ≥ {th['audio']['min_si_sdr_db']} dB, cosine to the anchor in S ≥ {th['audio']['min_s_cosine']} and decoded→original recall@1 ≥ {th['audio']['min_recall_at_1']}. "
             "Decisions use the validation split; the test split is reported only (ESC-50 fold 5 is touched by no training or selection).\n")
    for task, cols in (("caption_text", ["token_acc", "bleu1", "exact_match", "cos_to_anchor", "margin", "recall@1", "recall@5"]),
                       ("esc50_audio", ["si_sdr", "lsd", "cos_to_anchor", "margin", "recall@1", "recall@5", "recall@10", "clap_raw_cosine"])):
        for split in ("val", "test"):
            rows = []
            for ch in ("awgn", "rayleigh"):
                d = sanity["tasks"][task][ch]["splits"][split]
                for snr in GRID:
                    rows.append([ch, int(float(snr))] + [f"{d[snr][c]:.3f}" for c in cols])
            n = sanity["tasks"][task]["awgn"]["splits"][split]["0.0"]["n"]
            L.append(f"**{NAMES[task]}, {split} split (n = {n})**; cos/margin/recall are the decoded signal's CLIP/CLAP embedding, mapped into S, against the original's S anchor "
                     f"(recall within a gallery of {n}).\n")
            L.append(table(["channel", "SNR (dB)"] + cols, rows) + "\n")
    L.append("**Class-prompt exact recovery by the text transceiver** (fraction of prompts decoded exactly; averaged over noise draws; prompts were in its training data):\n")
    for ch in ("awgn", "rayleigh"):
        pr = sanity["tasks"]["caption_text"][ch]["prompt_exact_recovery"]
        sets = list(pr[GRID[0]])
        L.append(f"*{ch}*\n")
        L.append(table(["prompt set"] + [f"{int(float(s))} dB" for s in GRID], [[k] + [f"{pr[s][k]:.2f}" for s in GRID] for k in sets]) + "\n")

    # ------------------------------------------------------------------ anchors
    L.append("## 3. Anchors in S and centering statistics\n")
    L.append("Image and text anchors are centered CLIP; audio anchors are bridged, centered CLAP (seed-42 linear bridge). Each domain's centering mean is fitted on its TRAIN anchors only and cached "
             "(`outputs/week5b/space/centerer_<task>.pt`, provenance in `manifest.json`); ESC-50 uses folds 1-3 (asserted), and the new tasks' anchor caches assert that fold 5 shares no file with train or validation.\n")
    L.append(table(["domain", "centering fitted on", "n", "bridge", "random-pair cosine, raw test", "random-pair cosine, S test"],
                   [[NAMES[t], d["fit"]["fit"], d["fit"]["n_fit"], d["fit"]["bridge"] or "-", f"{d['random_pair_cosine']['raw_test']:.3f}", f"{d['random_pair_cosine']['S_test']:.3f}"]
                    for t, d in space["domains"].items()]))

    # ------------------------------------------------------------------ heads
    L.append("\n## 4. Heads (1 seed, 3 variants, 6 tasks × 2 channels)\n")
    rows = []
    for t in TASKS:
        for ch in ("awgn", "rayleigh"):
            for v, label in VARIANTS.items():
                f = HEADS / t / ch / f"train_{v}.json"
                if f.exists():
                    j = json.loads(f.read_text())
                    ck = torch.load(HEADS / t / ch / f"head_{v}.pt", map_location="cpu", weights_only=False)
                    total = {"cifar10": 40, "mnist": 40, "text": 20, "audio": 30, "caption_text": 20, "esc50_audio": 100}[t]
                    rows.append([NAMES[t], ch, v, j.get("target", "raw"), f"{j['n_head_params'] / 1e6:.2f}M", f"{j['best_val_total']:.3f}", f"{ck['epoch']} / {total}", f"{j['train_seconds'] / 60:.1f} min"])
    L.append(table(["task", "channel", "variant", "target space", "head params", "best val loss", "best epoch / epochs", "train time"], rows))
    L.append("\nA best epoch far below the total means the head overfit its training set and the checkpoint with the lowest validation loss was kept.\n")
    L.append("**Loss-curve check** (parsed from the training logs): relative improvement of the validation loss over the last 5 epochs. Because the learning rate follows a cosine decay, the best epoch is "
             "usually the last one; a large improvement over the last epochs would suggest the head was cut short, a value near zero that it had converged. Image tasks used the 40-epoch setting.\n")
    rows, flagged = [], []
    for t in TASKS:
        for ch in ("awgn", "rayleigh"):
            for v in VARIANTS:
                c = val_curve(head_log(t, ch, v))
                if len(c) >= 6:
                    gain = (c[-6] - c[-1]) / c[-6]
                    rows.append([NAMES[t], ch, v, len(c), f"{c[-1]:.3f}", f"{100 * gain:.1f}%"])
                    if gain > 0.02 and int(np.argmin(c)) >= len(c) - 2:
                        flagged.append(f"{NAMES[t]} {ch} {v}")
    L.append(table(["task", "channel", "variant", "epochs logged", "final val loss", "improvement over last 5 epochs"], rows))
    L.append("\nHeads still improving by more than 2% over their last 5 epochs and with their best epoch at the end: " + (", ".join(flagged) if flagged else "none") + ".\n")
    L.append("\nVariant semantics: `cos` and `cos5` keep the Week 5 recipe (InfoNCE + 0.5 / 5 × cosine against the RAW CLIP/CLAP anchors) and are scored after mapping their output into S; `cen` trains the whole "
             "objective (InfoNCE + 0.5 × cosine) toward the centered S anchor. For the four tasks that already had Week 5 heads, `cos` and `cos5` are those heads (same code, same seed 42, same epochs: "
             "40 image / 20 Europarl text / 30 speech) and were **reused rather than retrained**; only `cen` was trained for them. All three variants were trained for the two new tasks "
             "(20 caption-text epochs, 100 ESC-50 epochs because it has only 1,200 training clips). The loss curves did not call for a change from the 40-epoch image setting.\n")

    # ------------------------------------------------------------------ F selection
    L.append("## 5. Fidelity selection\n")
    L.append("Definitions (all in S, per item): `margin = cos(z, a_own) − mean cos(z, a_other)`; **F_abs = margin / (1 − b_aa)** (headline; b_aa is the mean unrelated anchor-anchor cosine, so the divisor is the "
             "anchor's own margin and there is no clean-channel rescale); **F_ret = margin / m_clean** (retained fraction of the noise-free margin, channel loss only); **C = m_clean / (1 − b_aa)** "
             "(clean ceiling per domain), so F_abs = F_ret × C exactly; baselines: centered cosine (cosine to the anchor in S) and raw cosine (cosine to the anchor in the space the head was trained in; "
             "identical to centered cosine for the `cen` variant). Statistics use 5 independent noise draws per item over the grid 0-25 dB on the validation split (test is shown separately). "
             "'Strictly increasing' is on the point estimates of the mean per SNR; 'sig. steps' counts adjacent SNR steps whose paired bootstrap 95% lower bound is above 0. Pooled Spearman is over every "
             "(item, draw, SNR) with no threshold. **C partly reflects instrument weakness** (for example CLIP on MNIST digits, CLAP on speech), so comparisons between systems are valid within a domain only.\n")
    if chosen:
        L.append("### Variant choice (validation, F_abs, single head seed)\n")
        L.append("Pre-registered rule: the variant with the most (task, channel) pairs whose mean F_abs is strictly increasing over the SNR grid; ties broken by the larger median dynamic range, then by the larger median "
                 "matched-vs-mismatched AUC at 0 dB.\n")
        L.append(table(["variant", "strictly increasing / pairs", "median dynamic range", "median AUC at 0 dB"],
                       [[f"{v} ({VARIANTS[v]})" + (" **← chosen**" if v == chosen else ""), f"{s['n_strictly_increasing']} / {s['n_pairs']}", f"{s['median_dynamic_range']:.3f}", f"{s['median_auc_lowest_snr']:.3f}"]
                        for v, s in summary.items()]))
        L.append(f"\nThe choice of `{chosen}` was made on one head seed, with one noise realisation set, so near-ties should not be read as a real ranking; Stage 2 repeats the chosen variant over 3 seeds "
                 "and reports seed-to-seed spread separately.\n")
    def collect(split, cand):
        out = {v: {} for v in VARIANTS}
        for t, r in fsel.items():
            for ch, splits in r["channels"].items():
                for v, d in splits.get(split, {}).items():
                    out[v][f"{t}/{ch}"] = (d["stats"][cand], d["reference"]["C"])
        return out

    if chosen:
        L.append("### How fragile is the choice? (the same statistics under other criteria, both splits)\n")
        L.append("All three variants are strictly increasing on every pair, so the pre-registered rule is decided by its first tie-breaker, the dynamic range of F_abs. That range depends on the "
                 "scale of F_abs (the clean ceiling C), not only on how well F separates conditions, so the table also shows scale-free criteria. 'sd/range' is the item-level standard deviation "
                 "of F_abs at 10 dB divided by the dynamic range (lower means a cleaner signal); 'won' counts the (task, channel) pairs, out of 12, on which that variant is best.\n")
        for split in ("val", "test"):
            data = collect(split, "F_abs")
            rows = []
            wins = {}
            for crit, key in (("range", lambda s_: s_["dynamic_range"]), ("AUC 0 dB", lambda s_: s_["auc"]["0.0"]), ("Spearman", lambda s_: s_["spearman_pooled"]),
                              ("sd/range (lower)", lambda s_: -s_["item_sd_over_range_at_10dB"])):
                w = {v: 0 for v in VARIANTS}
                for k in data["cos"]:
                    w[max(VARIANTS, key=lambda v_: key(data[v_][k][0]))] += 1
                wins[crit] = w
            for v in VARIANTS:
                st = [x[0] for x in data[v].values()]
                rows.append([v, f"{sum(x['strictly_increasing'] for x in st)}/{len(st)}", f"{np.median([x[1] for x in data[v].values()]):.3f}", f"{np.median([x['dynamic_range'] for x in st]):.3f}",
                             f"{np.median([x['auc']['0.0'] for x in st]):.3f}", f"{np.median([x['auc']['10.0'] for x in st]):.3f}", f"{np.median([x['spearman_pooled'] for x in st]):.3f}",
                             f"{np.median([x['item_sd_over_range_at_10dB'] for x in st]):.2f}", " / ".join(f"{wins[c][v]}" for c in wins)])
            L.append(f"**{split} split** (pairs won: range / AUC 0 dB / Spearman / sd-range)\n")
            L.append(table(["variant", "strictly increasing", "median C", "median range", "median AUC 0 dB", "median AUC 10 dB", "median pooled Spearman", "median sd/range", "pairs won"], rows) + "\n")
        L.append(f"**Reading:** the rule picks `{chosen}`, and the validation and test splits agree on that. But `cen` (training toward the centered anchor) is best on every scale-free criterion; `cos5` wins only "
                 "the scale-dependent range, because its heads sit closer to the anchors in absolute terms (higher C). Both are real advantages: a larger usable F_abs range versus cleaner separation of "
                 "conditions. The selection was made on one head seed and the gaps in the scale-free criteria are large and consistent across tasks, while the range gap is a scale effect, so I would "
                 "carry **both `cos5` and `cen`** into Stage 2 unless you prefer one; heads train in minutes, so this costs about twice the Stage 2 head compute, not twice the wall time.\n")
        L.append("### The four Fidelity candidates compared (chosen variant, validation)\n")
        rows = []
        for cand in CANDS:
            data = collect("val", cand)[chosen]
            st = [x[0] for x in data.values()]
            sn = [x["snr_at_90pct_clean"] for x in st]
            n_reached = sum(1 for d in sn if d["status"] == "interpolated")
            rows.append([cand, f"{sum(x['strictly_increasing'] for x in st)}/{len(st)}", f"{np.mean([x['steps_significant'] for x in st]):.2f}/5", f"{np.median([x['dynamic_range'] for x in st]):.3f}",
                         f"{np.median([x['spearman_pooled'] for x in st]):.3f}", f"{np.median([x['auc']['0.0'] for x in st]):.3f}", f"{np.median([x['auc']['10.0'] for x in st]):.3f}",
                         f"{np.median([d['snr'] for d in sn if d['snr'] is not None]):.1f} dB ({n_reached} interpolated, {len(sn) - n_reached} at/below 0 dB or unreached)"])
        L.append(table(["candidate", "strictly increasing", "mean sig. steps", "median range", "median pooled Spearman", "median AUC 0 dB", "median AUC 10 dB", "median SNR at 90% of clean"], rows))
        L.append("\nF_abs and F_ret are monotone transforms of the same margin, so their AUC and Spearman coincide; they differ in scale (F_ret is 1 by construction at the clean value, F_abs keeps the "
                 "ceiling C). The two cosine baselines differ from the margin in what they subtract: the centered cosine keeps the item's own similarity to unrelated anchors in the score.\n")
        L.append("### Clean ceiling C (= mean F_abs of the noise-free signal) per domain and variant, validation\n")
        rows = []
        data = {v: collect("val", "F_abs")[v] for v in VARIANTS}
        for t in TASKS:
            for ch in ("awgn", "rayleigh"):
                k = f"{t}/{ch}"
                if all(k in data[v] for v in VARIANTS):
                    rows.append([NAMES[t], ch] + [f"{data[v][k][1]:.3f}" for v in VARIANTS])
        L.append(table(["task", "channel"] + list(VARIANTS), rows))
        L.append("\nC partly reflects instrument weakness (CLIP on MNIST digits, CLAP on speech and on 1,200-clip ESC-50 training data) and head quality, so it is a property of the (transceiver, head, anchor space) triple: "
                 "comparisons are valid within a domain only.\n")
    pick = chosen or "cos"
    for split in ("val", "test"):
        for cand in CANDS:
            rows = []
            for t in TASKS:
                for ch in ("awgn", "rayleigh"):
                    d = fsel.get(t, {}).get("channels", {}).get(ch, {}).get(split, {}).get(pick)
                    if not d:
                        continue
                    s = d["stats"][cand]
                    rows.append([NAMES[t], ch] + [f"{s['means'][g]:.3f}" for g in GRID] + [f"{s['clean']:.3f}", "yes" if s["strictly_increasing"] else "no", f"{s['steps_significant']}/{s['n_steps']}",
                                                                                          f"{s['dynamic_range']:.3f}", f"{s['spearman_pooled']:.3f}", snr90(s["snr_at_90pct_clean"]),
                                                                                          f"{s['auc']['0.0']:.3f}", f"{s['auc']['10.0']:.3f}", f"{d['reference']['C']:.3f}" if cand in ("F_abs", "F_ret") else "-"])
            if rows:
                L.append(f"**{cand}, variant `{pick}`, {split} split**\n")
                L.append(table(["task", "channel"] + [f"{int(float(g))} dB" for g in GRID] + ["clean", "increasing", "sig. steps", "range", "Spearman", "SNR at 90% of clean", "AUC 0 dB", "AUC 10 dB", "C"], rows) + "\n")
    L.append("**All three variants side by side: F_abs on validation** (increasing / dynamic range / AUC at 0 dB)\n")
    rows = []
    for t in TASKS:
        for ch in ("awgn", "rayleigh"):
            row = [NAMES[t], ch]
            for v in VARIANTS:
                d = fsel.get(t, {}).get("channels", {}).get(ch, {}).get("val", {}).get(v)
                row.append("-" if not d else f"{'inc' if d['stats']['F_abs']['strictly_increasing'] else 'NOT inc'} / {d['stats']['F_abs']['dynamic_range']:.3f} / {d['stats']['F_abs']['auc']['0.0']:.3f}")
            rows.append(row)
    L.append(table(["task", "channel"] + list(VARIANTS), rows) + "\n")

    # ------------------------------------------------------------------ caveats
    L.append("## 6. Caveats and deviations from the approved plan\n")
    L.append("- **Single seed everywhere.** Heads are single-seed (variant chosen on one), and both new transceivers are single-seed, so intervals exclude transceiver-training variance.\n"
             "- **Deviation (prompts in the text transceiver's training set)** as described in section 1; held-out caption splits were not used. The prompt-recovery numbers are therefore optimistic for unseen text.\n"
             "- **Deviation (Week 5 heads reused)** for `cos` and `cos5` on the four older tasks (section 4); `cen` is new for all six tasks.\n"
             "- **Interpretation of the third variant.** I read 'cosine toward the centered anchor' as training the whole objective in S (InfoNCE and cosine both against the centered S anchor), keeping the two "
             "Week 5 variants against raw anchors so they remain comparable with Week 5. A cosine-weight-5 variant toward the centered anchor (the missing cell of a 2×2) was not trained.\n"
             "- **Selection and test.** F statistics were computed on both splits, but the choice used validation only. For ESC-50 the validation split is fold 4 (used also to select head and transceiver "
             "checkpoints) and test is fold 5, which no training, centering statistic or selection touched.\n"
             "- **Speech audio stays in the comparison with a weak instrument:** its S anchors have a residual random-pair cosine (table in section 3), so its F_abs scale is not comparable to the others.\n"
             "- **ESC-50 heads see only 1,200 training clips** and overfit (best epochs in section 4); event audio's clean ceiling C is the lowest of the six domains (section 5). Its F_abs saturates early in SNR, so it separates "
             "low SNRs but not high ones.\n"
             "- **LSD** barely improves above 10 dB on ESC-50 and stays high; it is dominated by low-energy spectral bins, and I did not investigate its composition further, so treat it as a relative measure only (silent reference frames are excluded).\n"
             "- **Not started:** Stage 2 (3 seeds for the chosen variant, cross-modal channel experiments, ImageBind ablation).\n")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(L))
    print(f"wrote {OUT}; chosen variant: {chosen}")


if __name__ == "__main__":
    main()
