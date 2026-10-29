"""Generate docs/week5b_stage0.md from outputs/week5b/{bridge,stage0}/results.json (no hand-copied numbers).

    python experiments/week5b_report.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("outputs/week5b")
OUT = Path("docs/week5b_stage0.md")


def ci(d, digits=3):
    return f"{d['point']:.{digits}f} [{d['lo']:.{digits}f}, {d['hi']:.{digits}f}]"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    gate = json.loads((ROOT / "stage0" / "results.json").read_text())
    bridge = json.loads((ROOT / "bridge" / "results.json").read_text())
    prot, dec = gate["protocol"], gate["protocol"]["decision_seed"]
    g, S = prot["gate"], gate["seeds"]
    d = S[dec]
    seeds = prot["seeds"]
    verdict = "PASS" if gate["gate_pass"] else "FAIL"
    mean_sd = lambda vals: f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}"  # noqa: E731

    g1_pts = [S[str(s)]["G1"]["top1"]["point"] for s in seeds]
    g2_pts = [S[str(s)]["G2"]["all_folds"]["balanced_top1"]["point"] for s in seeds]
    g2_los = [S[str(s)]["G2"]["all_folds"]["balanced_top1"]["lo"] for s in seeds]
    l2_pts = [S[str(s)]["rungs"]["L2 audio->bridge vs CLAP-text prompts->bridge"]["top1"]["point"] for s in seeds]
    passes = {str(s): (S[str(s)]["G1"]["pass"], S[str(s)]["G2"]["pass"]) for s in seeds}

    L = []
    L.append(f"# Week 5b, Stage 0: clean cross-modal gate for the shared space S\n")
    L.append(f"**Overall gate (decision seed {dec}): {verdict}.** Generated from `outputs/week5b/stage0/results.json`; nothing here involves "
             f"a channel. Per-seed pass flags (G1, G2): " + ", ".join(f"seed {s}: {a}/{b}" for s, (a, b) in passes.items()) + ".\n")

    L.append("## 1. Verdict against the pre-registered criteria\n")
    g1, g2 = d["G1"], d["G2"]
    L.append(table(["gate", "criterion", "result (decision seed)", "pass", f"mean ± sd over seeds {seeds}"], [
        ["G1 audio→text, ESC-50 test fold (400 clips, 50 classes, chance 0.02)",
         f"top-1 ≥ {g['g1_abs']} and ≥ {g['g1_rel']} × native CLAP",
         f"S top-1 {ci(g1['top1'])}; native {g1['native_top1']:.3f}; ratio {g1['ratio_to_native']:.2f}",
         "yes" if g1["pass"] else "no", f"S top-1 {mean_sd(g1_pts)}"],
        ["G2 audio→image, balanced top-1 over 5 shared classes (320 clips, chance 0.2)",
         f"point ≥ {g['g2_min_ratio']} × chance and lower 95% bound > chance",
         f"{ci(g2['all_folds']['balanced_top1'])}", "yes" if g2["pass"] else "no",
         f"point {mean_sd(g2_pts)}; lower bounds {mean_sd(g2_los)}"],
    ]))
    L.append(f"\nThe native figure uses the same 400 clips, the same 50 class names and the same template (`{prot['template']}`) as the S-space test. "
             "The template was fixed before any audio number was computed; the secondary prompt variants below show how much it matters. "
             "G1 is judged on the test fold (the only audio-fitted quantity, the audio mean, is fit on folds 1-3); G2 uses all five folds "
             "because the test fold holds only 64 mapped clips. The test-fold-only G2 number is reported too and, being perfect, has a "
             "degenerate bootstrap interval, so it is not used for the decision.\n")

    L.append("## 2. How S is built\n")
    L.append("S is the per-domain mean-centered CLIP space (512-d, unit norm). Image and text are native to it. Audio enters through CLAP, "
             "per-modality mean-centering (ESC-50 train folds 1-3), and a linear CLAP→CLIP bridge trained only on caption text "
             "(centered CLAP-text embedding → centered CLIP-text embedding; symmetric InfoNCE + 0.5·cosine, learnable temperature). "
             f"Statistics fit on train data only: audio mean on {prot['audio_centerer_fit_clips']} clips, text means on {prot['text_centerers_fit_captions'][0]} train captions.\n")

    L.append("## 3. Text-to-text agreement of the bridge (held-out captions)\n")
    rows = []
    for s in seeds:
        for kind, k in bridge["seeds"][str(s)]["kinds"].items():
            by = k["test_by_source"]
            rows.append([s, kind + (" (chosen)" if bridge["seeds"][str(s)]["chosen"] == kind else ""), f"{k['n_params'] / 1e3:.0f}k",
                         f"{k['val']['recall@1']:.3f}", f"{k['test']['cos']:.3f}", f"{k['test']['recall@1']:.3f}", f"{k['test']['recall@5']:.3f}",
                         f"{by['coco']['recall@1']:.3f}", f"{by['audiocaps']['recall@1']:.3f}"])
    r = bridge["ridge_reference"]["test"]
    rows.append(["-", "ridge regression (reference)", "-", "-", f"{r['cos']:.3f}", f"{r['recall@1']:.3f}", f"{r['recall@5']:.3f}", "-", "-"])
    L.append(table(["seed", "bridge", "params", "val R@1", "test cos (centered)", "test R@1", "test R@5", "R@1 COCO", "R@1 AudioCaps"], rows))
    n = bridge["n"]
    L.append(f"\nCaptions train/val/test = {n['train']}/{n['val']}/{n['test']} (COCO sub-sampled, all AudioCaps kept, captions duplicated across splits removed; "
             f"split by source item). Retrieval is within disjoint galleries of {bridge['gallery']} captions (chance {bridge['chance_recall@1']:.3f}). "
             f"The linear-vs-MLP choice follows the pre-registered rule (MLP only if it beats linear's validation R@1 by {bridge['config']['select_margin']}) "
             "and was made on validation captions only. Centered cosine is moderate because training optimises ranking (InfoNCE) with a 0.5-weight cosine term.\n")

    L.append("## 4. ESC-50 audio→text: ladder and failure breakdown\n")
    rows = [[k, ci(v["top1"]), ci(v["top5"])] for k, v in gate["native"].items()]
    rows += [[k, ci(v["top1"]), ci(v["top5"])] for k, v in d["rungs"].items()]
    L.append(table(["method (test fold, decision seed)", "top-1 [95% CI]", "top-5 [95% CI]"], rows))
    L.append(f"\nNative zero-shot on all 2000 clips: {ci(gate['native_all_folds_top1'])}. L1 on all 2000 clips: {ci(d['L1_all_folds']['top1'])}. "
             f"L2 over seeds: {mean_sd(l2_pts)}.\n")
    L.append("Reading the ladder: L0 is native CLAP. L1 is the gate. L2 sends the CLAP-text prompts through the same bridge, so the audio side and the "
             "prompt side are treated identically; the gap L0→L2 is the cost of the audio→bridge path, and the gap L2→L1 is the cost of using CLIP's "
             "own text encoder for the prompts. L3/L4 ablate the per-modality centering of audio.\n")
    pv = d["prompt_variants_L1_top1"]
    L.append("Prompt sensitivity (top-1, test fold), native → S: " + "; ".join(f"`{k}`: {gate['prompt_variants_native_top1'][k]:.3f} → {pv[k]:.3f}" for k in pv) + ".\n")
    b = d["breakdown"]
    L.append(f"Failure breakdown (decision seed): {b['classes_below_60pct_of_native']} of {b['classes_with_native_acc_gt_0']} classes with non-zero native accuracy "
             f"fall below {g['g1_rel']:.0%} of their native per-class accuracy; {b['classes_never_predicted']} classes are never predicted; the most-predicted class "
             f"takes {b['max_prediction_share']:.1%} of predictions (no collapse onto a few prompts).\n")
    L.append(table(["class", "n", "native acc", "S acc"], [[x["class"], x["n"], f"{x['native']:.2f}", f"{x['S']:.2f}"] for x in b["largest_drops"]]))
    L.append("\nMost frequent confusions in S: " + "; ".join(f"{c['true']} → {c['pred']} ({c['count']})" for c in b["top_confusions"][:8]) + ".\n")

    L.append("## 5. Audio→image (class level) and controls\n")
    L.append(f"ESC-50 classes mapped to CIFAR-10 before any result was computed: `{json.dumps(prot['esc50_to_cifar10'])}`. Balanced top-1 is used because "
             f"the 'bird' class holds {d['G2']['all_folds']['majority_class_share']:.0%} of the clips, so a constant 'bird' predictor already scores that "
             "in plain accuracy. Audio queries are classified by cosine to per-class CIFAR-10 test-image prototypes in S.\n")
    rows = [["audio(S) → image prototypes, all folds (the gate)", ci(d["G2"]["all_folds"]["balanced_top1"])],
            ["audio(S) → image prototypes, test fold only", ci(d["G2"]["test_fold"]["balanced_top1"]) + f" (n={d['G2']['test_fold']['n']})"]]
    rows += [[k, ci(v)] for k, v in d["G2_references"].items()]
    c = gate["controls"]
    rows += [["control: untrained bridge, audio→image", ci(c["G2_untrained_bridge_balanced_top1"])],
             ["control: untrained bridge, audio→text 50-way (G1 setting)", ci(c["G1_untrained_bridge_top1"])],
             ["control: permuted labels, audio→image (mean / max of 200)", f"{c['G2_permuted_labels_balanced_mean_max'][0]:.3f} / {c['G2_permuted_labels_balanced_mean_max'][1]:.3f}"],
             ["control: permuted labels, audio→text (mean / max of 200)", f"{c['G1_permuted_labels_top1_mean_max'][0]:.3f} / {c['G1_permuted_labels_top1_mean_max'][1]:.3f}"]]
    L.append(table(["setting", "balanced top-1 / top-1 [95% CI]"], rows))
    L.append("\nThe reference rows are not like-for-like: the S route is scored against CLIP-text *photo* prompts ('a photo of a {c}'), the native row against "
             "CLAP-text *sound* prompts, and the two prompt styles separate these five classes differently, so a higher S number than native here says nothing "
             "about the bridge being better than CLAP. An untrained bridge can fall below chance on a balanced metric because a random map is systematically, "
             "not randomly, wrong; the point of that control is that it is not above chance.\n")
    L.append(f"\nChance: G1 {c['chance']['G1']:.2f}, G2 {c['chance']['G2']:.2f}. Prediction histogram over the 5 classes (airplane, bird, cat, dog, frog), decision seed: "
             f"{d['G2']['all_folds']['pred_hist']}.\n")

    L.append("## 6. Clean image↔text checks in S (no audio, no channel)\n")
    rows = []
    for name, v in gate["clean_image_text_checks"].items():
        for tag in ("raw CLIP", "S (centered)"):
            x = v[tag]
            rows.append([name, tag, v["n_classes"], ci(x["i2t_top1"]), ci(x["i2t_top5"]) if x["i2t_top5"] else "-", f"{x['t2i_p@1']:.3f}", f"{x['t2i_r_precision']:.3f}"])
    L.append(table(["dataset", "space", "classes", "image→text top-1 [CI]", "top-5 [CI]", "text→image P@1", "text→image R-precision"], rows))
    L.append("\nCentering helps image→text most where CLIP is weakest (MNIST top-1 up, CIFAR-10 slightly up, CIFAR-100 within noise), but it does not "
             "uniformly improve text→image: MNIST's R-precision does not improve. With ten queries, text→image P@1 is a coarse number; R-precision is the informative one.\n")
    L.append("\nText→image is class-level: each class prompt queries the test images; R-precision is precision at k = images per class (1.0 is perfect). "
             "Centering is per domain: each image dataset has its own mean (fit on its train anchors), text uses the caption-domain CLIP-text mean.\n")

    L.append("## 7. Caveats, deviations and what was not run\n")
    L.append("- **Deviations from the approved plan:** (1) the bridge and the gate are repeated over three seeds because they cost minutes, with seed 42 as the "
             "pre-declared decision seed; (2) G1 is judged on the ESC-50 test fold and G2 on all folds (stated above); (3) the untrained-bridge and "
             "permuted-label controls were added; (4) COCO captions were sub-sampled to 150k for training and duplicates across splits were removed.\n"
             "- **Audio→image is class-level and goes through a text-trained bridge.** No paired audio-image data exists here, so this shows that audio and "
             "image class semantics land in the same neighbourhood of S, not that fine-grained audio-visual alignment holds.\n"
             "- **Only five classes overlap** between ESC-50 and CIFAR-10 (four exact names plus 'bird' mapped to four ESC-50 bird classes), so G2 is an easy 5-way task; "
             "its high score should not be read as general audio-image alignment.\n"
             "- **The S-space audio→text number is limited mostly by CLIP's text encoder** on near-synonym sound names (see the L1/L2 gap and the confusions), "
             "an instrument limit that cross-system comparisons inherit.\n"
             "- **Intervals** are percentile bootstraps over test clips only; they do not cover prompt wording (see the variants row) or caption-split variation. "
             "Seed spread is shown separately.\n"
             "- **Not run:** audio→caption retrieval on AudioCaps/Clotho (no audio was downloaded; Clotho audio was skipped by decision), COCO images, ImageBind "
             "(deferred to the end, optional, nothing derived from its weights may be shipped), and every channel experiment (Stage 1 and 2 are blocked on your review).\n")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(L))
    print(f"wrote {OUT} ({len(L)} blocks)")


if __name__ == "__main__":
    main()
