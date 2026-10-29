"""Generate docs/week5b_stage2.md and the Stage 2 figures from outputs/week5b/stage2/{metrics,heads}/*.json (no hand-copied numbers).

    python experiments/week5b_s2_report.py

An authored findings text (docs/week5b_stage2_findings.md), if present, is included verbatim at the top; every table and figure is generated.
Plots: AWGN solid, Rayleigh dashed, one colour per method, legend below the panels, 95% bootstrap CI bands.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("outputs/week5b/stage2")
HEADS = ROOT / "heads"
OUT = Path("docs/week5b_stage2.md")
FIGS = Path("docs/figures_week5b_stage2")
TASKS = ("cifar10", "mnist", "text", "audio", "caption_text", "esc50_audio")
NAMES = {"cifar10": "CIFAR-10 image", "mnist": "MNIST image", "text": "Europarl text", "audio": "Valentini speech",
         "caption_text": "Caption text", "esc50_audio": "ESC-50 event audio"}
CH = (("awgn", "AWGN"), ("rayleigh", "Rayleigh"))
COL = {"A_cos5": "#1d6fb8", "A_cen": "#e08a1d", "B": "#2a9d4b"}
LAB = {"A_cos5": "A: head, cos5 (raw targets)", "A_cen": "A: head, cen (centered targets)", "B": "B: pretrained encoder on the decoded output"}
SNRS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
CROSS = [0.0, 5.0, 10.0, 15.0, 25.0]


def ld(name):
    p = ROOT / "metrics" / name
    return json.loads(p.read_text()) if p.exists() else None


def f3(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def c3(c, d=3):
    return "-" if c is None else f"{c['point']:.{d}f} [{c['lo']:.{d}f}, {c['hi']:.{d}f}]"


def pm(s, d=3):
    return "-" if s is None else f"{s['mean']:.{d}f} ± {s['sd']:.{d}f}"


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def snr90(d):
    if d["point"] is None:
        return {"not_reached_by_grid_max": "> 25 dB", "undefined": "n/a"}.get(d["status"], d["status"])
    s = f"{d['point']:.1f} dB"
    if d.get("lo") is not None:
        s += f" [{d['lo']:.1f}, {d['hi']:.1f}]"
    return s + ("" if d["status"] == "interpolated" else " (≤ lowest grid point)")


def legend_below(fig, ncol=4):
    h, l = {}, {}
    for ax in fig.axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            h.setdefault(ll, hh)
    fig.legend(list(h.values()), list(h.keys()), loc="lower center", ncol=ncol, fontsize=7.5, frameon=False)


def band(ax, x, curve, color, style, label=None):
    pt = np.array([c["point"] for c in curve])
    lo = np.array([c["lo"] for c in curve])
    hi = np.array([c["hi"] for c in curve])
    ax.plot(x, pt, style, color=color, markersize=3.5, lw=1.4, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0)


def style_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("SNR (dB)", fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)


# ===========================================================================================================================
def head_tables(L):
    rows, esc = [], []
    for t in TASKS:
        for ch, _ in CH:
            for var in ("cos5", "cen"):
                for seed in (42, 43, 44):
                    f = HEADS / t / ch / f"train_{var}_s{seed}.json"
                    if f.exists():
                        j = json.loads(f.read_text())
                        rows.append([NAMES[t], ch, var, seed, j["schedule"], f"{j['best_epoch']} / {j['epochs_run']} (max {j['max_epochs']})",
                                     "yes" if j["stopped_early"] else "no", f"{j['best_val_total']:.3f}", f"{j['dropout']}", f"{j['weight_decay']:g}", f"{j['train_seconds'] / 60:.1f}"])
    L.append(table(["task", "channel", "variant", "seed", "schedule", "best epoch / epochs run", "early stop", "best val loss", "dropout", "weight decay", "min"], rows))
    s = ROOT / "esc50_setting.json"
    if s.exists():
        d = json.loads(s.read_text())
        for case, v in d["cases"].items():
            esc.append([case] + [f"{v[k]['best_val_total']:.3f} (ep {v[k]['best_epoch']})" for k in ("base", "R1", "R2")])
        esc.append(["**mean rank**"] + [f"**{d['mean_rank'][k]:.2f}**" for k in ("base", "R1", "R2")])
        L.append("\n**ESC-50 head regularisation comparison (seed 42 only; validation loss is the identical objective for all three settings; lower is better):**\n")
        L.append(table(["case", "baseline (dropout 0.1, wd 1e-4)", "R1 (0.3, 1e-3)", "R2 (0.5, 1e-2)"], esc))
        L.append(f"\nThe lowest mean rank wins and ties go to the baseline: **{d['chosen']}** (dropout {d['chosen_setting']['dropout']}, weight decay {d['chosen_setting']['weight_decay']:g}) "
                 "was applied to all three ESC-50 seeds. The winner's best epochs are close to the 100-epoch cap, so it may still have been improving; the epoch cap was not raised.\n")


def selection_section(L, F):
    sel = F["selection"]
    L.append(f"**Status: {sel['status']}.**\n")
    if "amended_final" not in sel:
        return None
    d = sel["amended_detail"]
    rows = []
    for crit, info in d["criteria"].items():
        vs = [v for v in ("cos5", "cen")]
        rows.append([crit, info["better"] + " is better", f"{info[vs[0]]:.4f}", f"{info[vs[1]]:.4f}", info["winner"] or "tie"])
    L.append("**Amended rule (validation, scale-free criteria, 3-seed means, median over the 12 (task, channel) pairs):**\n")
    L.append(table(["criterion", "direction", "cos5", "cen", "better"], rows))
    L.append(f"\nCriteria won: {d['criteria_won']}. **Final headline variant under the amended rule: `{sel['amended_final']}`.**\n")
    ctx = sel["context"]
    L.append("Context only (NOT selection criteria under the amended rule): " + "; ".join(
        f"`{v}`: median dynamic range {ctx[v]['median_dynamic_range']:.3f}, median clean ceiling C {ctx[v]['median_C']:.3f}, strictly increasing on {ctx[v]['n_strictly_increasing_pairs']}/12 pairs (all 3 seeds)" for v in ctx) + ".\n")
    L.append(f"**Original pre-registered rule re-applied unchanged to the same 3-seed means** (strictly increasing pairs, then median dynamic range, then median AUC at 0 dB): picks `{sel['original_rule']}`. "
             "Its summary: " + "; ".join(f"`{v}`: {s['n_strictly_increasing']}/{s['n_pairs']} increasing, median range {s['median_dynamic_range']:.3f}, median AUC at 0 dB {s['median_auc_lowest_snr']:.3f}"
                                           for v, s in sel["original_rule_summary"].items()) + ". It is shown for transparency; it is not the selection rule.\n")
    return sel["amended_final"]


def f_tables(L, F, final):
    for split in ("test", "val"):
        for method in ("A_cos5", "A_cen", "B"):
            if method == "B" and split == "val":
                continue
            rows, rows2, rows3, rows4 = [], [], [], []
            for t in TASKS:
                for ch, chl in CH:
                    e = F["tasks"].get(t, {}).get(ch, {}).get(split, {}).get(method)
                    if not e:
                        continue
                    fa, fr = e["cands"]["F_abs"], e["cands"]["F_ret"]
                    st = fa["stats"]
                    rows.append([NAMES[t], chl] + [c3(c) for c in st["means"]] + [c3(st["clean"])])
                    sp = fa.get("spread", {})
                    rows2.append([NAMES[t], chl, "yes" if st["strictly_increasing"] else "no", f"{st['frac_resamples_strictly_increasing']:.2f}", c3(st["dynamic_range"]),
                                  c3(st["spearman_pooled"]), snr90(st["snr_at_90pct_clean"]), c3(st["auc"]["0.0"]), c3(st["auc"]["10.0"]),
                                  f3(np.mean([r["C"] for r in e["reference_per_seed"]])) , f"{st['item_sd_over_range_at_10dB']:.2f}",
                                  pm(sp.get("auc_at_0dB")), pm(sp.get("dynamic_range")), pm(sp.get("clean"))])
                    rows3.append([NAMES[t], chl] + [c3(c) for c in fr["stats"]["means"]] + [f3(np.mean([r["C"] for r in e["reference_per_seed"]]))])
                    for cand in ("centered cosine", "raw cosine"):
                        if cand in e["cands"]:
                            b = e["cands"][cand]["stats"]
                            rows4.append([NAMES[t], chl, cand, "yes" if b["strictly_increasing"] else "no", f"{b['frac_resamples_strictly_increasing']:.2f}", c3(b["dynamic_range"]),
                                          c3(b["spearman_pooled"]), snr90(b["snr_at_90pct_clean"]), c3(b["auc"]["0.0"]), c3(b["auc"]["10.0"])])
            if not rows:
                continue
            tag = f"`{method}`" + (" (final headline variant)" if method == f"A_{final}" else "")
            n = next(iter(F["tasks"][t][ch][split][method]["n_items"] for t in TASKS for ch, _ in CH if F["tasks"].get(t, {}).get(ch, {}).get(split, {}).get(method)), "?")
            L.append(f"#### {tag}, {split} split\n")
            L.append(f"**F_abs per SNR, mean [95% bootstrap CI over items]** (items per domain: see section 8; 5 noise draws per item" + ("; mean over 3 head seeds per item" if method != "B" else "; one deterministic decoder, no head seeds") + ").\n")
            L.append(table(["domain", "channel"] + [f"{int(s)} dB" for s in SNRS] + ["clean"], rows) + "\n")
            L.append("**F_abs summary statistics.** 'increasing' is on the point estimates of the mean curve; 'resamples increasing' is the share of bootstrap resamples in which the mean curve is strictly increasing; "
                     "Spearman is pooled over items, draws and SNR points, reported as measured (no threshold); 'SNR at 90%' is the lowest SNR at which mean F_abs reaches 90% of its clean value "
                     "(interval over bootstrap resamples; ≤ lowest grid point means already reached at 0 dB). AUC is matched-vs-mismatched pairs. The last three columns are head-seed spread (mean ± sd over 3 seeds), separate from the CIs.\n")
            L.append(table(["domain", "channel", "increasing", "resamples increasing", "range 25-0 dB", "pooled Spearman", "SNR at 90% of clean", "AUC 0 dB", "AUC 10 dB", "C", "item sd / range at 10 dB",
                            "seed spread AUC 0 dB", "seed spread range", "seed spread clean F_abs"], rows2) + "\n")
            L.append("**F_ret per SNR (retained fraction of the clean margin; channel loss only) with C (clean ceiling = clean F_abs, so F_abs = F_ret × C).**\n")
            L.append(table(["domain", "channel"] + [f"{int(s)} dB" for s in SNRS] + ["C"], rows3) + "\n")
            L.append("**Baselines: centered cosine and raw cosine** (raw = cosine to the anchor in the space the head was trained in; for `cen` heads it equals the centered cosine; approach B has no raw baseline).\n")
            L.append(table(["domain", "channel", "baseline", "increasing", "resamples increasing", "range 25-0 dB", "pooled Spearman", "SNR at 90% of clean", "AUC 0 dB", "AUC 10 dB"], rows4) + "\n")


def f_plot(F, final):
    for split in ("test",):
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
        for ax, t in zip(axes.flat, TASKS):
            for ch, chl in CH:
                for m in ("A_cos5", "A_cen", "B"):
                    e = F["tasks"].get(t, {}).get(ch, {}).get(split, {}).get(m)
                    if e:
                        band(ax, SNRS, e["cands"]["F_abs"]["stats"]["means"], COL[m], "o-" if ch == "awgn" else "s--", LAB[m] + (" [AWGN]" if ch == "awgn" else " [Rayleigh]"))
            style_ax(ax, NAMES[t], "F_abs")
        fig.suptitle(f"F_abs vs SNR, {split} split (mean over items, 95% bootstrap CI band; AWGN solid, Rayleigh dashed)", fontsize=11)
        legend_below(fig, 3)
        fig.tight_layout(rect=(0, 0.1, 1, 0.96))
        FIGS.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGS / f"f_abs_{split}.png", dpi=130)
        plt.close(fig)


# ===========================================================================================================================
def method_rows(e, key, snrs, with_static=True):
    rows = []
    cl = e["ceiling"].get(key)
    if cl:
        rows.append(["clean-source ceiling (frozen anchor)"] + [c3(cl["value"])] * len(snrs))
    for m in ("A_cos5", "A_cen", "B"):
        d = e["methods"].get(m)
        if not d:
            continue
        idxs = [SNRS.index(s) for s in snrs]
        rows.append([LAB[m]] + [c3(d[key]["curve"][i]) for i in idxs])
        if with_static:
            rows.append([f"  {m}, noise-free transceiver"] + [c3(d[key + "_noise_free"]["value"])] * len(snrs))
    return rows


def diff_rows(e, key, snrs):
    rows = []
    for m in ("A_cos5", "A_cen"):
        d = e["methods"].get(m)
        if d and f"{key}_minus_B" in d:
            rows.append([LAB[m] + " − B"] + [c3(d[f"{key}_minus_B"][SNRS.index(s)]) for s in snrs])
    return rows


def spread_rows(e, key, snrs):
    rows = []
    for m in ("A_cos5", "A_cen"):
        d = e["methods"].get(m)
        if d and "spread" in d[key]:
            rows.append([LAB[m]] + [pm(d[key]["spread"][SNRS.index(s)]) for s in snrs])
    return rows


CROSS_PAIRS = (("image_to_text_cifar10", "image → text, CIFAR-10 (10 class prompts)"), ("image_to_text_mnist", "image → text, MNIST (10 digit prompts)"),
               ("image_to_audio_cifar10", "image → audio, CIFAR-10 (CLASS-LEVEL, 5 shared classes, balanced accuracy)"),
               ("audio_to_text_esc50", "audio → text, ESC-50 (50 class prompts)"),
               ("audio_to_image_esc50", "audio → image, ESC-50 (CLASS-LEVEL, 5 shared classes, balanced accuracy, test fold only)"))


def cross_section(L, C):
    L.append("Top-1 is class-level retrieval: the transmitted item's embedding in S against clean class prompts or clean class prototypes. Every cell is mean [95% bootstrap CI over items]; "
             "A rows average the per-item score over 3 head seeds, and the seed spread is in the separate tables below. Chance levels are in the table captions. The Week 4 / Stage 1 transceivers are single-seed, "
             "so CIs exclude transceiver-training variance. 'Noise-free transceiver' is the same system with an identity channel (the SNR → ∞ limit); the gap between it and the clean-source ceiling is the encoder's own loss.\n")
    for key, title in CROSS_PAIRS:
        ex = C["experiments"].get(key)
        if not ex:
            continue
        for ch, chl in CH:
            e = ex.get(ch)
            if not e:
                continue
            ks = ["top1"] + (["top5"] if "top5" in e["methods"].get(next(iter(e["methods"])), {}) else [])
            for k in ks:
                chance = e["chance"].get(k)
                L.append(f"**{title}: {k}, {chl}.** {e['label']}; chance {chance:.2f}.\n")
                L.append(table(["method"] + [f"{int(s)} dB" for s in CROSS], method_rows(e, k, CROSS)) + "\n")
            diffs = diff_rows(e, "top1", CROSS)
            if diffs:
                L.append(f"Paired difference A − B in top-1, {chl} (same items and the same channel noise; mean [95% bootstrap CI over items]; positive = the head beats the pretrained encoder on the decoded output):\n")
                L.append(table(["method"] + [f"{int(s)} dB" for s in CROSS], diffs) + "\n")
            sp = spread_rows(e, "top1", CROSS)
            if sp:
                L.append(f"Seed-to-seed spread of top-1, {chl} (mean ± sd over 3 head seeds):\n")
                L.append(table(["method"] + [f"{int(s)} dB" for s in CROSS], sp) + "\n")
        if key == "audio_to_text_esc50":
            ts = ex["awgn"].get("template_sensitivity", {})
            if ts:
                L.append("**Prompt-wording sensitivity of audio → text (AWGN, top-1 at 25 dB; point estimates).** The first three templates are the pre-registered family, the last three are the held-out paraphrases:\n")
                rows = [[repr(t), f3(v["ceiling_top1"])] + [f3(v[m][SNRS.index(25.0)]) for m in ("A_cos5", "A_cen", "B") if m in v] for t, v in ts.items()]
                hdr = ["template", "clean ceiling"] + [m for m in ("A_cos5", "A_cen", "B") if any(m in v for v in ts.values())]
                L.append(table(hdr, rows) + "\n")


def cross_plot(C):
    keys = [k for k, _ in CROSS_PAIRS if k in C["experiments"]]
    if not keys:
        return
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        ex = C["experiments"][key]
        chance = None
        for ch, chl in CH:
            e = ex.get(ch)
            if not e:
                continue
            chance = e["chance"]["top1"]
            for m in ("A_cos5", "A_cen", "B"):
                if m in e["methods"]:
                    band(ax, SNRS, e["methods"][m]["top1"]["curve"], COL[m], "o-" if ch == "awgn" else "s--", LAB[m] + (" [AWGN]" if ch == "awgn" else " [Rayleigh]"))
            ax.axhline(e["ceiling"]["top1"]["value"]["point"], color="#6b7280", ls=":", lw=1.3, label="clean-source ceiling")
        if chance:
            ax.axhline(chance, color="#c8312b", ls="-.", lw=0.9, alpha=0.6, label="chance")
        style_ax(ax, dict(CROSS_PAIRS)[key].split("(")[0].strip(), "top-1" if "CLASS-LEVEL" not in dict(CROSS_PAIRS)[key] else "balanced top-1")
    fig.suptitle("Cross-modal retrieval after transmission vs SNR (95% bootstrap CI bands; AWGN solid, Rayleigh dashed)", fontsize=11)
    legend_below(fig, 4)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "cross_modal.png", dpi=130)
    plt.close(fig)


# ===========================================================================================================================
PROMPT_PAIRS = (("text_to_image_cifar10", "text → image, CIFAR-10 (10 classes)"), ("text_to_image_mnist", "text → image, MNIST (10 digits)"),
                ("text_to_audio_esc50", "text → audio, ESC-50 (50 classes)"))


def prompt_section(L, C):
    L.append("The caption-domain text transceiver was trained on the 280 class-prompt sentences (×10) as well as on captions, so results on those *training-time* templates (a) are optimistic. "
             "Held-out paraphrase templates (b), written once before any Stage 2 seed was trained (hash in the protocol record), with zero out-of-vocabulary words and no verbatim occurrence in the training data, "
             "are the **headline** for every claim involving transmitted text. The queries are the transmitted prompts; the gallery is the clean image/audio class prototypes. Intervals are bootstrap over prompts "
             "(10 to 150 prompts per set, so some are wide).\n")
    for key, title in PROMPT_PAIRS:
        ex = C["experiments"].get(key)
        if not ex:
            continue
        for ch, chl in CH:
            e = ex.get(ch)
            if not e:
                continue
            for sname in ("heldout", "train"):
                s = e["sets"][sname]
                L.append(f"**{title}, {chl}, set ({'b: held-out paraphrases (HEADLINE)' if sname == 'heldout' else 'a: training-time templates'}), {s['n_prompts']} prompts, top-1** (chance {s['chance']['top1']:.2f}).\n")
                L.append(table(["method"] + [f"{int(x)} dB" for x in CROSS], method_rows(s, "top1", CROSS, with_static=False)) + "\n")
            rows = []
            for m in ("A_cos5", "A_cen", "B"):
                if m in e["extra_loss_heldout_vs_train"]:
                    rows.append([LAB[m]] + [c3(e["extra_loss_heldout_vs_train"][m]["top1"][SNRS.index(x)], 2) for x in CROSS])
            raw = [[LAB[m]] + [c3(e["gap_train_minus_heldout"][m]["top1"][SNRS.index(x)], 2) for x in CROSS] for m in ("A_cos5", "A_cen", "B") if m in e["gap_train_minus_heldout"]]
            L.append(f"**Optimism check, {title}, {chl}.** Raw gap = top-1(training-time) − top-1(held-out). Part of it is CLIP's own template-wording effect, visible in the clean ceiling "
                     f"(ceiling gap top-1 = {e['ceiling_gap_train_minus_heldout']['top1']:+.3f}). The transceiver's optimism is the held-out set's EXTRA loss relative to its own ceiling: "
                     "(ceiling_b − top1_b) − (ceiling_a − top1_a); positive means the transceiver handles the paraphrases worse than the templates it was trained on.\n")
            L.append("Raw gap:\n")
            L.append(table(["method"] + [f"{int(x)} dB" for x in CROSS], raw) + "\n")
            L.append("Extra loss on held-out vs training-time templates (transceiver optimism):\n")
            L.append(table(["method"] + [f"{int(x)} dB" for x in CROSS], rows) + "\n")


def prompt_plot(C):
    keys = [k for k, _ in PROMPT_PAIRS if k in C["experiments"]]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(4.8 * len(keys), 4.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        for ch, chl in CH:
            e = C["experiments"][key].get(ch)
            if not e:
                continue
            for m in ("A_cos5", "A_cen", "B"):
                if m in e["sets"]["heldout"]["methods"]:
                    band(ax, SNRS, e["sets"]["heldout"]["methods"][m]["top1"]["curve"], COL[m], "o-" if ch == "awgn" else "s--", LAB[m] + (" [AWGN]" if ch == "awgn" else " [Rayleigh]"))
            ax.axhline(e["sets"]["heldout"]["ceiling"]["top1"]["value"]["point"], color="#6b7280", ls=":", lw=1.3, label="clean-source ceiling (held-out set)")
            ax.axhline(e["sets"]["heldout"]["chance"]["top1"], color="#c8312b", ls="-.", lw=0.9, alpha=0.6, label="chance")
        style_ax(ax, dict(PROMPT_PAIRS)[key] + ", held-out prompts", "top-1")
    fig.suptitle("Transmitted text prompts, HELD-OUT paraphrase templates (headline): retrieval vs SNR", fontsize=11)
    legend_below(fig, 4)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    fig.savefig(FIGS / "text_prompts_heldout.png", dpi=130)
    plt.close(fig)


# ===========================================================================================================================
def within_section(L, W):
    L.append("Self-retrieval: the received item's embedding retrieves its own clean anchor among all N test items of the domain (recall@k; chance k/N). The clean-source ceiling of self-retrieval is trivially 1.0, "
             "so the informative reference is the noise-free transceiver (identity channel). Class retrieval (images, ESC-50): against clean same-modality class prototypes. **Gallery sizes differ from Week 5 "
             "for image tasks and the noise protocol differs (Week 5: one seed, one noise draw, raw space), so Week 5 numbers are shown alongside and are comparable only up to those differences.**\n")
    for t in TASKS:
        for ch, chl in CH:
            e = W["tasks"].get(t, {}).get(ch)
            if not e:
                continue
            rows = []
            for m in ("A_cos5", "A_cen", "B"):
                d = e["metrics"].get(m)
                if not d:
                    continue
                for key in ("recall@1", "recall@5", "cos_to_anchor", "margin"):
                    if key in d:
                        rows.append([LAB[m], key] + [c3(d[key]["curve"][SNRS.index(s)]) for s in CROSS] + [c3(d[key + "_noise_free"]["value"])])
                    if f"{key}_minus_B" in d and key in ("recall@1", "recall@5"):
                        rows.append([LAB[m] + " − B (paired)", key] + [c3(d[f"{key}_minus_B"][SNRS.index(s)]) for s in CROSS] + ["-"])
            L.append(f"**{NAMES[t]}, {chl} (N = {e['n_items']}, chance recall@1 = {1 / e['n_items']:.4f}).**\n")
            L.append(table(["method", "metric"] + [f"{int(s)} dB" for s in CROSS] + ["noise-free"], rows) + "\n")
            if "class_retrieval" in e:
                cr = e["class_retrieval"]
                L.append(f"Class-prototype retrieval, top-1, {NAMES[t]}, {chl}:\n")
                L.append(table(["method"] + [f"{int(s)} dB" for s in CROSS], method_rows(cr, "top1", CROSS)) + "\n")
    w5 = []
    for t in ("cifar10", "mnist", "text", "audio"):
        for ch, chl in CH:
            p = Path("outputs/week5") / t / ch / "results.json"
            if p.exists():
                r = json.loads(p.read_text())
                key = "top1" if t in ("cifar10", "mnist") else "recall@1"
                for m, lab in (("A_cos5", "A cos5 (Week 5)"), ("B", "B (Week 5)")):
                    if m in r["results"]:
                        w5.append([NAMES[t], chl, lab, key] + [f3(r["results"][m][str(float(s))][key]) for s in (0, 10, 25)] + [str(r["n_eval"])])
    if w5:
        L.append("**Week 5 reference (single seed, one noise draw, raw anchor space, Week 5 gallery size):**\n")
        L.append(table(["domain", "channel", "method", "metric", "0 dB", "10 dB", "25 dB", "N"], w5) + "\n")


def within_plot(W):
    keys = [t for t in TASKS if t in W["tasks"] and W["tasks"][t]]
    if not keys:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
    for ax, t in zip(axes.flat, TASKS):
        for ch, chl in CH:
            e = W["tasks"].get(t, {}).get(ch)
            if not e:
                continue
            for m in ("A_cos5", "A_cen", "B"):
                if m in e["metrics"]:
                    band(ax, SNRS, e["metrics"][m]["recall@1"]["curve"], COL[m], "o-" if ch == "awgn" else "s--", LAB[m] + (" [AWGN]" if ch == "awgn" else " [Rayleigh]"))
        style_ax(ax, f"{NAMES[t]}: self-retrieval recall@1", "recall@1")
    fig.suptitle("Within-modality self-retrieval vs SNR (95% bootstrap CI bands; AWGN solid, Rayleigh dashed)", fontsize=11)
    legend_below(fig, 3)
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    fig.savefig(FIGS / "within_modality.png", dpi=130)
    plt.close(fig)


# ===========================================================================================================================
def main() -> None:
    F, C, W = ld("F.json"), ld("cross.json"), ld("within.json")
    proto = json.loads((ROOT / "protocol_declared.json").read_text())
    L = []
    findings = Path("docs/week5b_stage2_findings.md")
    L.append("# Week 5b, Stage 2: co-primary variants, three seeds, cross-modal channel experiments\n")
    if findings.exists():
        L.append(findings.read_text() + "\n")
    L.append("## 0. Protocol declaration (written before any Stage 2 seed was run)\n")
    am = proto["amendment"]
    L.append(f"Record `outputs/week5b/stage2/protocol_declared.json`, declared at **{proto['declared_at_utc']}** (UTC), hashing the Stage 1 variant choice (`{proto['stage1_variant_choice']}`, sha256 `{proto['stage1_variant_choice_sha256'][:16]}…`) "
             f"and the held-out prompt templates (sha256 `{proto['evaluation']['text_prompt_optimism']['heldout_templates_sha256'][:16]}…`).\n")
    L.append(f"**{am['kind']}.** {am['reason']}\n")
    r = am["final_variant_rule"]
    L.append("Final-variant rule (applied exactly as written once the 3-seed numbers existed): " + r["aggregation"] + " Criteria: " +
             "; ".join(f"**{k}** ({v})" for k, v in r["criteria"].items()) + ". Context only, not selection: " + ", ".join(r["context_only_not_selection"]) + ".\n")
    L.append("Held-out prompt templates (written once, never used for vocabulary, transceiver training or head training; zero `<unk>` and no verbatim overlap asserted in `semcom.data.prompts`):\n")
    L.append(table(["task", "training-time templates (a)", "held-out paraphrase templates (b)"],
                   [[k, "; ".join(repr(t) for t in v), "; ".join(repr(t) for t in proto["evaluation"]["text_prompt_optimism"]["heldout_templates"][k])]
                    for k, v in proto["evaluation"]["text_prompt_optimism"]["training_templates"].items()]) + "\n")
    final = None
    L.append("## 1. Heads (3 seeds × {cos5, cen} × 6 tasks × 2 channels)\n")
    head_tables(L)
    L.append("## 2. Final variant under the amended rule\n")
    if F:
        final = selection_section(L, F)
    else:
        L.append("F metrics not computed yet.\n")
    L.append("## 3. Fidelity: F_abs (headline), F_ret, C and the cosine baselines\n")
    L.append("Equal F across modalities does **not** yet mean equal semantic damage. F_abs is normalised within each domain's own anchor space (CLIP for image and text, CLAP-through-the-bridge for audio), and the clean ceiling C "
             "partly reflects instrument weakness (CLIP on MNIST digits, CLAP on speech and on 1,200 training clips). Cross-system comparisons are valid within a domain only; calibrating F against human judgements "
             "across modalities is what the human study is for.\n")
    L.append("**Every interval in this document excludes transceiver-training variance:** the Week 4 transceivers and the two Stage 1 transceivers (caption text, ESC-50 event audio) are single-seed. "
             "Item-level 95% CIs are percentile bootstraps over test items; the head-seed spread (mean ± sd over 3 seeds) is reported in separate columns and tables, never mixed into a CI. "
             "Each item's score is first averaged over 5 independent noise draws (and over the 3 head seeds for approach A).\n")
    if F:
        f_tables(L, F, final)
        f_plot(F, final)
        L.append("![F_abs vs SNR](figures_week5b_stage2/f_abs_test.png)\n")
    L.append("## 4. Cross-modal channel experiments (test split)\n")
    if C:
        cross_section(L, C)
        cross_plot(C)
        L.append("![cross-modal](figures_week5b_stage2/cross_modal.png)\n")
        L.append("## 5. Transmitted text: held-out paraphrases (headline) vs training-time templates\n")
        prompt_section(L, C)
        prompt_plot(C)
        L.append("![text prompts](figures_week5b_stage2/text_prompts_heldout.png)\n")
    L.append("## 6. Within-modality retrieval (test split; Week 5 comparable up to the stated differences)\n")
    if W:
        within_section(L, W)
        within_plot(W)
        L.append("![within-modality](figures_week5b_stage2/within_modality.png)\n")
    caveats = Path("docs/week5b_stage2_caveats.md")
    if caveats.exists():
        L.append("## 7. Deviations, caveats and what was not run\n")
        L.append(caveats.read_text() + "\n")
    OUT.write_text("\n".join(L))
    print(f"wrote {OUT} ({len(L)} blocks); final variant: {final}")


if __name__ == "__main__":
    main()
