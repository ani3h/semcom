"""Stage 2: compute every Stage 2 statistic from the cached embeddings and write JSON (the only source of the report's tables and plots).

    python experiments/week5b_s2_metrics.py section=F              # Fidelity candidates + the variant selection (val) per task x channel
    python experiments/week5b_s2_metrics.py section=cross          # cross-modal channel experiments (test) + the prompt-optimism check
    python experiments/week5b_s2_metrics.py section=within         # within-modality retrieval (test), Week 5 comparable

Per-item scores are averaged over the 5 noise draws and over head seeds; the item is the bootstrap unit (percentile 95%, 1000 resamples).
Head-seed spread (mean and sd over the 3 seeds) is stored separately. The Week 4 / Stage 1 transceivers are single-seed, so no interval
here includes transceiver-training variance.
"""
import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torchvision import datasets

from semcom.common_space import stage2_eval as ev
from semcom.common_space.anchors import PretrainedAnchors, anchor_path, load_anchors
from semcom.common_space.domains import get_domain
from semcom.common_space.fidelity import choose_variant
from semcom.common_space.gate import ESC50_TO_CIFAR10
from semcom.common_space.stage2 import SNR_GRID
from semcom.common_space.stage2_stats import (AMENDED_CRITERIA, boot_idx, choose_final_variant, diff_ci, f_curve_stats, seed_spread)
from semcom.data.audio_esc50 import ESC50Dataset
from semcom.data.prompts import ESC50_TEMPLATES, HELDOUT_TEMPLATES, TRAINING_TEMPLATES, class_names, esc50_classes

CROSS_SNRS = [0.0, 5.0, 10.0, 15.0, 25.0]
FIVE = ["airplane", "bird", "cat", "dog", "frog"]
KS = (1, 5)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return o


def emb_file(cfg, kind, task, ch, split, variant=None, seed=None) -> Path:
    name = f"{kind}_{task}_{ch}_{split}" + (f"_{variant}_s{seed}" if variant else "") + ".pt"
    return Path(cfg.out_root) / "emb" / name


def load_blob(cfg, kind, task, ch, split, variant=None, seed=None):
    p = emb_file(cfg, kind, task, ch, split, variant, seed)
    return torch.load(p) if p.exists() else None


def collect_methods(cfg, task, ch, split, kind_prefix="") -> dict[str, list]:
    """{"A_cos5": [blob per seed], "A_cen": [...], "B": [blob]} for the files that exist."""
    out = {}
    for variant in cfg.variants:
        blobs = [b for s in cfg.seeds if (b := load_blob(cfg, f"{kind_prefix}A", task, ch, split, variant, s)) is not None]
        if blobs:
            out[f"A_{variant}"] = blobs
    b = load_blob(cfg, f"{kind_prefix}B", task, ch, split)
    if b is not None:
        out["B"] = [b]
    return out


# ===========================================================================================================================
def run_F(cfg: DictConfig) -> None:
    out_dir = Path(cfg.out_root) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"snrs": SNR_GRID, "tasks": {}}
    crit = {v: {} for v in cfg.variants}  # amended rule inputs: variant -> pair -> [seed criteria dicts]   (validation, F_abs)
    cref = {v: {} for v in cfg.variants}  # mean clean ceiling C per variant and pair (context only)
    orig = {v: {} for v in cfg.variants}  # original rule inputs: variant -> pair -> point stats of the seed-mean F_abs (validation)
    for task in cfg.tasks:
        domain = get_domain(task, cfg.anchor_root, cfg.space_root, cfg.bridge_dir)
        results["tasks"][task] = {}
        for ch in cfg.channels:
            results["tasks"][task][ch] = {}
            for split in cfg.splits:
                methods = collect_methods(cfg, task, ch, split)
                if not methods:
                    continue
                a_S_full = domain.to_S(load_anchors(anchor_path(cfg.anchor_root, task, split))["anchors"])
                res_split = {}
                for name, blobs in methods.items():
                    if name == "B" and split != "test":
                        continue
                    t0 = time.time()
                    N = blobs[0]["z"].shape[2]
                    a_S, idx = a_S_full[:N], boot_idx(N, 1000, seed=N)
                    per_seed = []
                    for blob in blobs:
                        raw = None
                        if "head" in blob and blob["head"]["target"] == "raw":
                            raw = {"cos": blob["raw_cos"], "cos_mis": blob["raw_cos_mis"], "cos_clean": blob["raw_cos_clean"]}
                        arr = ev.f_arrays(blob["z"], blob["z_clean"], a_S, raw)
                        if "head" in blob and blob["head"]["target"] == "centered":
                            arr["raw cosine"] = arr["centered cosine"]  # a head trained in S has no separate raw space
                        per_seed.append(arr)
                    entry = {"n_items": int(N), "n_seeds": len(per_seed), "reference_per_seed": [a["ref"] for a in per_seed], "cands": {}}
                    for cand in ev.FCANDS:
                        if cand not in per_seed[0]:
                            continue
                        Fb = np.mean([a[cand]["F"] for a in per_seed], axis=0)
                        Mb = np.mean([a[cand]["F_mis"] for a in per_seed], axis=0)
                        Cb = np.mean([a[cand]["clean"] for a in per_seed], axis=0)
                        c = {"stats": f_curve_stats(Fb, Mb, Cb, SNR_GRID, idx)}
                        seeds_sum = [ev.seed_summary(a[cand]["F"], a[cand]["F_mis"], a[cand]["clean"], SNR_GRID) for a in per_seed]
                        c["per_seed"] = seeds_sum
                        if len(seeds_sum) > 1:
                            c["spread"] = {k: seed_spread([s[k] for s in seeds_sum if s[k] is not None]) for k in
                                           ("clean", "dynamic_range", "auc_at_0dB", "pooled_spearman", "item_noise_over_range", "snr_at_90pct_clean")}
                            c["spread"]["means"] = [seed_spread([s["means"][i] for s in seeds_sum]) for i in range(len(SNR_GRID))]
                        entry["cands"][cand] = c
                        if cand == "F_abs" and name.startswith("A_") and split == "val":
                            variant = name[2:]
                            crit[variant][f"{task}/{ch}"] = seeds_sum
                            cref[variant][f"{task}/{ch}"] = float(np.mean([r["C"] for r in entry["reference_per_seed"]]))
                            st = c["stats"]
                            orig[variant][f"{task}/{ch}"] = {"strictly_increasing": st["strictly_increasing"], "dynamic_range": st["dynamic_range"]["point"],
                                                             "auc": {k: v["point"] for k, v in st["auc"].items()}}
                    res_split[name] = entry
                    print(f"[F] {task}/{ch}/{split}/{name}: {len(per_seed)} seed(s), N={N} [{time.time() - t0:.0f}s]", flush=True)
                results["tasks"][task][ch][split] = res_split
    results["selection"] = select_variant(crit, orig, cref, cfg)
    (out_dir / "F.json").write_text(json.dumps(jsonable(results)))
    print(f"[F] wrote {out_dir / 'F.json'}; selection: {json.dumps(jsonable({k: results['selection'].get(k) for k in ('status', 'amended_final', 'original_rule')}))}", flush=True)


def select_variant(crit, orig, cref, cfg) -> dict:
    """The amended rule (validation, scale-free criteria) and, unchanged, the original Stage 1 rule, both on the 3-seed means."""
    pairs = [f"{t}/{ch}" for t in cfg.tasks for ch in cfg.channels]
    complete = all(len(crit[v].get(p, [])) == len(cfg.seeds) for v in cfg.variants for p in pairs)
    out = {"status": "complete" if complete else "INCOMPLETE (not all seeds / pairs available)", "pairs": pairs, "n_seeds": len(cfg.seeds)}
    if not complete:
        return out
    chosen, detail = choose_final_variant(crit)
    out["amended_final"] = chosen
    out["amended_detail"] = detail
    # context only, NOT selection criteria
    out["context"] = {v: {"median_dynamic_range": float(np.median([np.mean([s["dynamic_range"] for s in crit[v][p]]) for p in pairs])),
                          "median_C": float(np.median([cref[v][p] for p in pairs])), "n_strictly_increasing_pairs": int(sum(all(s["strictly_increasing"] for s in crit[v][p]) for p in pairs))}
                      for v in cfg.variants}
    # the original rule, re-applied unchanged to the seed-mean statistics of the two co-primary variants
    chosen_o, summary_o = choose_variant({v: {p: {**orig[v][p], "auc": orig[v][p]["auc"]} for p in pairs} for v in cfg.variants})
    out["original_rule"] = chosen_o
    out["original_rule_summary"] = summary_o
    return out


# ===========================================================================================================================
def build_refs(cfg, pre: PretrainedAnchors) -> dict:
    """Clean galleries and prototypes in S, built once: text prompts (caption domain), image class prototypes, audio class prototypes."""
    dom = {n: get_domain(n, cfg.anchor_root, cfg.space_root, cfg.bridge_dir) for n in ("cifar10", "mnist", "caption_text", "esc50_audio")}
    refs = {"dom": dom}

    def text_S(texts):
        return dom["caption_text"].to_S(pre.encode_text(texts).cpu())

    refs["text_S"] = text_S
    c10 = datasets.CIFAR10(cfg.data_root, train=False)
    mn = datasets.MNIST(cfg.data_root, train=False)
    refs["c10_names"], refs["c10_y"] = c10.classes, torch.tensor(c10.targets)
    refs["mn_y"] = mn.targets.clone()
    for t, y in (("cifar10", refs["c10_y"]), ("mnist", refs["mn_y"])):
        a = dom[t].to_S(load_anchors(anchor_path(cfg.anchor_root, t, "test"))["anchors"])
        refs[f"{t}_S"] = a
        refs[f"{t}_protos"] = ev.class_prototypes(a, y, 10)
    blob = load_anchors(anchor_path(cfg.anchor_root, "esc50", "all", "_clap_16k"))
    aud_S = dom["esc50_audio"].to_S(blob["anchors"])
    folds, labels, cats = blob["folds"], blob["labels"], blob["categories"]
    train = folds <= 3
    refs["esc_names"] = esc50_classes(cfg.data_root)
    refs["esc_protos"] = ev.class_prototypes(aud_S[train], labels[train], 50)  # clean audio prototypes from the TRAIN folds only
    m = torch.tensor([FIVE.index(ESC50_TO_CIFAR10[cats[int(l)]]) if cats[int(l)] in ESC50_TO_CIFAR10 else -1 for l in labels])
    refs["esc_to5"] = torch.tensor([FIVE.index(ESC50_TO_CIFAR10[c]) if c in ESC50_TO_CIFAR10 else -1 for c in cats])
    refs["aud5_protos"] = ev.class_prototypes(aud_S[train & (m >= 0)], m[train & (m >= 0)], 5)
    c10_5 = torch.tensor([FIVE.index(c10.classes[int(t)]) if c10.classes[int(t)] in FIVE else -1 for t in refs["c10_y"]])
    refs["c10_to5"] = c10_5
    refs["img5_protos"] = ev.class_prototypes(refs["cifar10_S"][c10_5 >= 0], c10_5[c10_5 >= 0], 5)
    ds = ESC50Dataset(cfg.data_root, split="test", download=False)
    refs["esc_test_y"] = torch.tensor(ds.meta["target"].values)
    refs["esc_test_S"] = dom["esc50_audio"].to_S(load_anchors(anchor_path(cfg.anchor_root, "esc50_audio", "test"))["anchors"])
    return refs


def per_method_class_scores(methods, protos, labels, sel=None, ks=KS):
    """{method: {k: [per-seed (S,N) arrays]}} and {method: {k: noise-free (N,) averaged over seeds}} for class-prototype retrieval."""
    curves, static = {}, {}
    for name, blobs in methods.items():
        per = [ev.class_correct(b["z"], protos, labels, ks, sel) for b in blobs]
        sta = [ev.class_correct_static(b["z_clean"], protos, labels, ks, sel) for b in blobs]
        curves[name] = {k: [p[k] for p in per] for k in per[0]}
        static[name] = {k: np.mean([s[k] for s in sta], axis=0) for k in sta[0]}
    return curves, static


def summarize_experiment(curves, static, ceiling, labels_np, balanced, idx) -> dict:
    out = {"ceiling": {f"top{k}": ev.summarize_static(v, idx, labels_np, balanced) for k, v in ceiling.items()}, "methods": {}}
    for name in curves:
        out["methods"][name] = {"snr": SNR_GRID}
        for k in curves[name]:
            out["methods"][name][f"top{k}"] = ev.summarize(curves[name][k], idx, labels_np, balanced)
            out["methods"][name][f"top{k}_noise_free"] = ev.summarize_static(static[name][k], idx, labels_np, balanced)
            if "B" in curves and name != "B":  # paired A - B: same items, same channel noise
                out["methods"][name][f"top{k}_minus_B"] = ev.paired_difference(curves[name][k], curves["B"][k], idx, labels_np, balanced)
    return out


def run_cross(cfg: DictConfig) -> None:
    out_dir = Path(cfg.out_root) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pre = PretrainedAnchors(device=device)
    refs = build_refs(cfg, pre)
    results = {"snrs": SNR_GRID, "cross_snrs": CROSS_SNRS, "experiments": {}}
    c10_names, esc = refs["c10_names"], refs["esc_names"]

    # ---- clean class-prompt galleries (text), the pre-registered ESC-50 template is the headline
    g_c10 = refs["text_S"]([f"a photo of a {c}" for c in c10_names])
    g_mn = refs["text_S"]([f"a handwritten digit {n}" for n in range(10)])
    esc_templates = list(ESC50_TEMPLATES)
    g_esc = {t: refs["text_S"]([t.format(c) if "{}" in t else t for c in esc]) for t in esc_templates}
    g_esc_held = {t: refs["text_S"]([t.format(c=c) for c in esc]) for t in HELDOUT_TEMPLATES["esc50"]}
    headline_t = esc_templates[0]

    for ch in cfg.channels:
        # ---------------------------------------------------------------- image -> text, image -> audio
        for task, g_text, y, nm in (("cifar10", g_c10, refs["c10_y"], "image_to_text_cifar10"), ("mnist", g_mn, refs["mn_y"], "image_to_text_mnist")):
            methods = collect_methods(cfg, task, ch, "test")
            if not methods:
                continue
            N = next(iter(methods.values()))[0]["z"].shape[2]
            labels, idx = y[:N], boot_idx(N, 1000, seed=N)
            curves, static = per_method_class_scores(methods, g_text, labels)
            ceil = ev.class_correct_static(refs[f"{task}_S"][:N], g_text, labels, KS)
            results["experiments"].setdefault(nm, {})[ch] = {**summarize_experiment(curves, static, ceil, labels.numpy(), False, idx), "n_classes": 10,
                                                              "chance": {"top1": 0.1, "top5": 0.5}, "label": "class-level, 10 class prompts"}
            if task == "cifar10":  # image -> audio over the 5 shared classes (balanced accuracy)
                sel = (refs["c10_to5"][:N] >= 0).nonzero().squeeze(1)
                l5 = refs["c10_to5"][:N][sel]
                idx5 = boot_idx(len(sel), 1000, seed=7)
                curves, static = per_method_class_scores(methods, refs["aud5_protos"], l5, sel, ks=(1,))
                ceil = ev.class_correct_static(refs["cifar10_S"][:N], refs["aud5_protos"], l5, (1,), sel)
                results["experiments"].setdefault("image_to_audio_cifar10", {})[ch] = {
                    **summarize_experiment(curves, static, ceil, l5.numpy(), True, idx5), "n_classes": 5, "chance": {"top1": 0.2},
                    "label": "CLASS-LEVEL over the 5 shared ESC-50/CIFAR-10 classes, balanced accuracy; audio prototypes = clean ESC-50 train folds 1-3"}
        # ---------------------------------------------------------------- audio -> text, audio -> image (ESC-50 test fold 5)
        methods = collect_methods(cfg, "esc50_audio", ch, "test")
        if methods:
            labels = refs["esc_test_y"]
            N = len(labels)
            idx = boot_idx(N, 1000, seed=N)
            curves, static = per_method_class_scores(methods, g_esc[headline_t], labels)
            ceil = ev.class_correct_static(refs["esc_test_S"], g_esc[headline_t], labels, KS)
            res = {**summarize_experiment(curves, static, ceil, labels.numpy(), False, idx), "n_classes": 50, "chance": {"top1": 0.02, "top5": 0.1},
                   "label": f"50 ESC-50 class prompts, template {headline_t!r}", "template_sensitivity": {}}
            for t, g in {**g_esc, **g_esc_held}.items():  # prompt wording: ceiling and each method's top-1 at 25 dB, point estimates
                c2, s2 = per_method_class_scores(methods, g, labels, ks=(1,))
                res["template_sensitivity"][t] = {"ceiling_top1": float(ev.class_correct_static(refs["esc_test_S"], g, labels, (1,))[1].mean()),
                                                  **{m: [float(np.mean([x.mean(1) for x in c2[m][1]], axis=0)[i]) for i in range(len(SNR_GRID))] for m in c2}}
            results["experiments"].setdefault("audio_to_text_esc50", {})[ch] = res
            sel = (refs["esc_to5"][labels] >= 0).nonzero().squeeze(1)
            l5 = refs["esc_to5"][labels][sel]
            curves, static = per_method_class_scores(methods, refs["img5_protos"], l5, sel, ks=(1,))
            ceil = ev.class_correct_static(refs["esc_test_S"], refs["img5_protos"], l5, (1,), sel)
            results["experiments"].setdefault("audio_to_image_esc50", {})[ch] = {
                **summarize_experiment(curves, static, ceil, l5.numpy(), True, boot_idx(len(sel), 1000, seed=11)), "n_classes": 5, "chance": {"top1": 0.2},
                "label": "CLASS-LEVEL over the 5 shared classes, balanced accuracy, ESC-50 test fold only; image prototypes = CIFAR-10 test images"}
        # ---------------------------------------------------------------- text -> image / audio: the transmitted prompts (a) vs (b)
        pm = collect_methods(cfg, "caption_text", ch, "prompts", kind_prefix="prompts")
        meta = (next(iter(pm.values()))[0]["meta"]) if pm else []
        if pm:
            for task, protos, nm, anchor_texts in (("cifar10", refs["cifar10_protos"], "text_to_image_cifar10", None), ("mnist", refs["mnist_protos"], "text_to_image_mnist", None),
                                                   ("esc50", refs["esc_protos"], "text_to_audio_esc50", None)):
                entry = {"sets": {}}
                for sname, label in (("train", "a: training-time templates"), ("heldout", "b: held-out paraphrase templates (HEADLINE)")):
                    sel = torch.tensor([i for i, m in enumerate(meta) if m["task"] == task and m["set"] == sname])
                    labels = torch.tensor([meta[i]["class_idx"] for i in sel.tolist()])
                    texts_clean = [meta[i]["input"] for i in sel.tolist()]
                    idx = boot_idx(len(sel), 1000, seed=len(sel))
                    sub = {name: blobs for name, blobs in pm.items()}
                    curves, static = per_method_class_scores(sub, protos, labels, sel)
                    clean_emb = refs["text_S"](texts_clean)
                    ceil = ev.class_correct_static(clean_emb, protos, labels, KS)
                    entry["sets"][sname] = {**summarize_experiment(curves, static, ceil, labels.numpy(), False, idx), "n_prompts": int(len(sel)), "label": label,
                                            "n_classes": int(protos.size(0)), "chance": {"top1": 1 / protos.size(0), "top5": 5 / protos.size(0)},
                                            "_per_item": {m: {k: np.mean(curves[m][k], axis=0) for k in curves[m]} for m in curves},
                                            "_ceil_item": {k: v for k, v in ceil.items()}}
                # the optimism gap: training-template minus held-out accuracy, per method and SNR (independent prompt sets)
                entry["gap_train_minus_heldout"] = {}
                for m in pm:
                    entry["gap_train_minus_heldout"][m] = {f"top{k}": [diff_ci(entry["sets"]["train"]["_per_item"][m][k][s], entry["sets"]["heldout"]["_per_item"][m][k][s])
                                                                       for s in range(len(SNR_GRID))] for k in entry["sets"]["train"]["_per_item"][m]}
                # The raw gap mixes two things: how well CLIP separates the classes under each template wording (visible in the clean ceiling) and how
                # well the transceiver transmits each set. The transceiver's optimism is the held-out set's EXTRA loss relative to its own ceiling:
                # (ceiling_b - acc_b) - (ceiling_a - acc_a), per prompt, bootstrapped over prompts.
                entry["extra_loss_heldout_vs_train"] = {}
                ca, cb = entry["sets"]["train"]["_ceil_item"], entry["sets"]["heldout"]["_ceil_item"]
                for m in pm:
                    entry["extra_loss_heldout_vs_train"][m] = {
                        f"top{k}": [diff_ci(cb[k][:, None].repeat(len(SNR_GRID), 1)[:, s] - entry["sets"]["heldout"]["_per_item"][m][k][s],
                                            ca[k][:, None].repeat(len(SNR_GRID), 1)[:, s] - entry["sets"]["train"]["_per_item"][m][k][s]) for s in range(len(SNR_GRID))]
                        for k in entry["sets"]["train"]["_per_item"][m]}
                entry["ceiling_gap_train_minus_heldout"] = {f"top{k}": float(ca[k].mean() - cb[k].mean()) for k in ca}
                for s in entry["sets"].values():
                    s.pop("_per_item")
                    s.pop("_ceil_item")
                entry["label"] = "class-level; query = transmitted class prompt, gallery = clean image/audio class prototypes"
                results["experiments"].setdefault(nm, {})[ch] = entry
        print(f"[cross] {ch} done", flush=True)
    (out_dir / "cross.json").write_text(json.dumps(jsonable(results)))
    print(f"[cross] wrote {out_dir / 'cross.json'}", flush=True)


# ===========================================================================================================================
def run_within(cfg: DictConfig) -> None:
    out_dir = Path(cfg.out_root) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"snrs": SNR_GRID, "tasks": {}}
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pre = PretrainedAnchors(device=device)
    refs = build_refs(cfg, pre)
    for task in cfg.tasks:
        domain = get_domain(task, cfg.anchor_root, cfg.space_root, cfg.bridge_dir)
        a_full = domain.to_S(load_anchors(anchor_path(cfg.anchor_root, task, "test"))["anchors"])
        results["tasks"][task] = {}
        for ch in cfg.channels:
            methods = collect_methods(cfg, task, ch, "test")
            if not methods:
                continue
            N = next(iter(methods.values()))[0]["z"].shape[2]
            a_S, idx = a_full[:N], boot_idx(N, 1000, seed=N)
            entry = {"n_items": int(N), "metrics": {}}
            for name, blobs in methods.items():
                per = [ev.self_retrieval_arrays(b["z"], a_S) for b in blobs]
                sta = [ev.self_retrieval_static(b["z_clean"], a_S) for b in blobs]
                m = {}
                for key in per[0]:
                    m[key] = ev.summarize([p[key] for p in per], idx)
                    m[key + "_noise_free"] = ev.summarize_static(np.mean([s[key] for s in sta], axis=0), idx)
                    if "B" in methods and name != "B":
                        b_per = ev.self_retrieval_arrays(methods["B"][0]["z"], a_S)
                        m[key + "_minus_B"] = ev.paired_difference([p[key] for p in per], [b_per[key]], idx)
                entry["metrics"][name] = m
            if task in ("cifar10", "mnist", "esc50_audio"):  # within-modality class retrieval against clean same-modality prototypes
                if task == "esc50_audio":
                    labels, protos = refs["esc_test_y"], refs["esc_protos"]
                    ceil_emb = refs["esc_test_S"]
                else:
                    y_all = refs["c10_y"] if task == "cifar10" else refs["mn_y"]
                    labels, ceil_emb = y_all[:N], a_S
                    protos = ev.class_prototypes(a_full[N:], y_all[N:], 10)  # prototypes from test images that are NOT queries (no self-inclusion)
                curves, static = per_method_class_scores(methods, protos, labels)
                ceil = ev.class_correct_static(ceil_emb, protos, labels, KS)
                entry["class_retrieval"] = {**summarize_experiment(curves, static, ceil, labels.numpy(), False, idx),
                                            "label": "within-modality class-prototype retrieval (clean same-modality prototypes)"}
            results["tasks"][task][ch] = entry
            print(f"[within] {task}/{ch} N={N}", flush=True)
    (out_dir / "within.json").write_text(json.dumps(jsonable(results)))
    print(f"[within] wrote {out_dir / 'within.json'}", flush=True)


@hydra.main(config_path="../configs", config_name="week5b_s2_eval", version_base=None)
def main(cfg: DictConfig) -> None:
    {"F": run_F, "cross": run_cross, "within": run_within}[cfg.section](cfg)


if __name__ == "__main__":
    main()
