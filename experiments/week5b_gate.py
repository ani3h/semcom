"""Stage 0, step 3: the clean (noise-free) cross-modal gate for the shared space S, with the failure breakdown.

    python experiments/week5b_gate.py

S = centered CLIP space. Image and text are native to it (per-domain mean-centering, fit on train anchors); audio enters
through CLAP -> per-modality centering (ESC-50 train folds 1-3) -> text-trained bridge. Nothing here involves a channel.

Pre-registered protocol (fixed before any audio number was computed, see configs/week5b.yaml):
  G1  audio->text top-1 in S on the ESC-50 test fold (fold 5, 400 clips) >= gate.g1_abs AND >= gate.g1_rel x native CLAP
      zero-shot top-1 on the same clips with the same class set and prompt template.
  G2  audio->image BALANCED top-1 (5 classes shared by ESC-50 and CIFAR-10, chance 0.2) on all mapped ESC-50 clips:
      point estimate >= gate.g2_min_ratio x chance AND lower 95% bootstrap bound > chance.
Decision seed = cfg.seed; the other seeds only report spread. Writes outputs/week5b/stage0/results.json and
docs/week5b_stage0.md (every number in the report is generated from results.json).
"""
import json
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torchvision import datasets

from semcom.common_space.anchors import PretrainedAnchors, anchor_path, load_anchors
from semcom.common_space.bridge import TextBridge
from semcom.common_space.correction import Centerer
from semcom.common_space.gate import (ESC50_TO_CIFAR10, balanced_accuracy, bootstrap_ci, class_prototypes, r_precision,
                                      topk_correct)
from semcom.tracking.run_naming import make_run_name
from semcom.tracking.wandb_logger import init_run


def pick_device(name: str) -> torch.device:
    return torch.device(name) if name != "auto" else torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def ci_dict(correct, labels=None, balanced=False, n_boot=2000, seed=0) -> dict:
    p, lo, hi = bootstrap_ci(correct, labels, balanced, n_boot, seed=seed)
    return {"point": p, "lo": lo, "hi": hi, "n": int(len(correct))}


def zero_shot(q: torch.Tensor, protos: torch.Tensor, labels: torch.Tensor, n_boot: int) -> dict:
    scores = q @ protos.T
    out = {"top1": ci_dict(topk_correct(scores, labels, 1), n_boot=n_boot)}
    if protos.size(0) > 5:
        out["top5"] = ci_dict(topk_correct(scores, labels, 5), n_boot=n_boot)
    out["pred"] = scores.argmax(1).tolist()
    return out


def load_bridge(path: Path) -> TextBridge:
    ck = torch.load(path)
    m = TextBridge(ck["kind"], hidden=ck["hidden"])
    m.load_state_dict(ck["state"])
    return m.eval()


@torch.no_grad()
def image_text_check(name, train_anchors, test_anchors, labels, prompts_clip, cy, n_boot) -> dict:
    ci_ = Centerer().fit(train_anchors)
    img_s, txt_s = ci_.transform(test_anchors), cy.transform(prompts_clip)
    out = {"n_images": len(labels), "n_classes": prompts_clip.size(0)}
    for tag, im, tx in (("raw CLIP", test_anchors, prompts_clip), ("S (centered)", img_s, txt_s)):
        zs = zero_shot(im, tx, labels, n_boot)
        rp = r_precision(tx, im, torch.arange(tx.size(0)), labels)
        out[tag] = {"i2t_top1": zs["top1"], "i2t_top5": zs.get("top5"), "t2i_p@1": rp["p@1"], "t2i_r_precision": rp["r_precision"]}
    return out


@hydra.main(config_path="../configs", config_name="week5b", version_base=None)
def main(cfg: DictConfig) -> None:
    device = pick_device(cfg.device)
    out_dir, nb, g = Path(cfg.out_root) / "stage0", cfg.gate.n_boot, cfg.gate
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = PretrainedAnchors(device=device)
    bridge_dir = Path(cfg.out_root) / "bridge"
    bridge_res = json.loads((bridge_dir / "results.json").read_text())
    cx, cy = Centerer.load(bridge_dir / "centerer_clap_text.pt"), Centerer.load(bridge_dir / "centerer_clip_text.pt")

    # ------------------------------------------------------------------ ESC-50 anchors, prompts
    blob = load_anchors(anchor_path(cfg.anchor_root, "esc50", "all", "_clap_16k"))
    full = load_anchors(anchor_path(cfg.anchor_root, "esc50", "all", "_clap_fullband"))["anchors"]
    A, folds, y = blob["anchors"], blob["folds"].numpy(), blob["labels"]
    names, cats = blob["class_names"], blob["categories"]
    ca = Centerer().fit(A[torch.from_numpy(folds <= 3)])  # audio mean from TRAIN folds only
    test = torch.from_numpy(folds == 5)
    template = cfg.esc50.template
    P_clap = pre.encode_text_clap([template.format(n) for n in names]).cpu()
    P_clip = pre.encode_text([template.format(n) for n in names]).cpu()
    ytest = y[test]
    print(f"[gate] ESC-50: {len(A)} clips, test fold {int(test.sum())}; template {template!r}", flush=True)

    native = {  # native CLAP zero-shot, same clips / classes / template
        "L0 native CLAP (16 kHz source)": zero_shot(A[test], P_clap, ytest, nb),
        "L0b native CLAP (original 44.1 kHz audio)": zero_shot(full[test], P_clap, ytest, nb),
        "L0c native CLAP + per-modality centering": zero_shot(ca.transform(A[test]), cx.transform(P_clap), ytest, nb),
    }
    native_all = zero_shot(A, P_clap, y, nb)
    variants = {}
    for v in cfg.esc50.prompt_variants:
        pc, pl = pre.encode_text_clap([v.format(n) for n in names]).cpu(), pre.encode_text([v.format(n) for n in names]).cpu()
        variants[v] = {"native_top1": zero_shot(A[test], pc, ytest, nb)["top1"]["point"], "_pl": pl}

    # ------------------------------------------------------------------ audio->image setup (CIFAR-10 x ESC-50 overlap)
    c10_ds = datasets.CIFAR10(cfg.data_root, train=False)
    c10_names = c10_ds.classes
    five = ["airplane", "bird", "cat", "dog", "frog"]
    idx5 = {n: i for i, n in enumerate(five)}
    a_lab5 = torch.tensor([idx5[ESC50_TO_CIFAR10[cats[int(t)]]] if cats[int(t)] in ESC50_TO_CIFAR10 else -1 for t in y])
    mapped_all, mapped_test = a_lab5 >= 0, (a_lab5 >= 0) & test
    c10_train = load_anchors(anchor_path(cfg.anchor_root, "cifar10", "train"))["anchors"]
    c10_test = load_anchors(anchor_path(cfg.anchor_root, "cifar10", "test"))["anchors"]
    c10_y = torch.tensor(c10_ds.targets)
    keep = torch.tensor([c10_names[int(t)] in idx5 for t in c10_y])
    img_lab5 = torch.tensor([idx5[c10_names[int(t)]] for t in c10_y[keep]])
    img_s = Centerer().fit(c10_train).transform(c10_test[keep])
    protos_img = class_prototypes(img_s, img_lab5, 5)
    P5_photo = cy.transform(pre.encode_text([f"a photo of a {c}" for c in five]).cpu())
    P5_clap_sound = pre.encode_text_clap([template.format(c) for c in five]).cpu()
    print(f"[gate] audio->image: {int(mapped_all.sum())} mapped clips ({int(mapped_test.sum())} in test fold), "
          f"{int(keep.sum())} CIFAR-10 test images of the 5 classes", flush=True)

    # ------------------------------------------------------------------ per-seed bridge ladder + G1 / G2
    seeds_out, decision, kept_S = {}, str(cfg.seed), {}
    for seed in cfg.seeds:
        kind = bridge_res["seeds"][str(seed)]["chosen"]
        bridge = load_bridge(bridge_dir / f"{kind}_s{seed}.pt")
        with torch.no_grad():
            S_a = bridge(ca.transform(A))
            S_a_textmean = bridge(cx.transform(A))
            S_a_nocenter = bridge(F.normalize(A, dim=-1))
            S_p_bridge = bridge(cx.transform(P_clap))
        S_p_clip = cy.transform(P_clip)
        kept_S[str(seed)] = S_a
        rungs = {
            "L1 audio->bridge vs CLIP-text prompts (the gate)": zero_shot(S_a[test], S_p_clip, ytest, nb),
            "L2 audio->bridge vs CLAP-text prompts->bridge": zero_shot(S_a[test], S_p_bridge, ytest, nb),
            "L3 audio centered with the TEXT mean (ablation)": zero_shot(S_a_textmean[test], S_p_clip, ytest, nb),
            "L4 audio not centered (ablation)": zero_shot(S_a_nocenter[test], S_p_clip, ytest, nb),
        }
        l1_all = zero_shot(S_a, S_p_clip, y, nb)
        l1_var = {v: zero_shot(S_a[test], cy.transform(d["_pl"]), ytest, nb)["top1"]["point"] for v, d in variants.items()}
        l0 = native["L0 native CLAP (16 kHz source)"]["top1"]["point"]
        l1 = rungs["L1 audio->bridge vs CLIP-text prompts (the gate)"]["top1"]
        g1 = {"top1": l1, "native_top1": l0, "ratio_to_native": l1["point"] / max(l0, 1e-9), "abs_ok": l1["point"] >= g.g1_abs,
              "rel_ok": l1["point"] >= g.g1_rel * l0}
        g1["pass"] = g1["abs_ok"] and g1["rel_ok"]

        def g2_run(sel):
            s = S_a[sel]
            pred = (s @ protos_img.T).argmax(1)
            correct, lab = (pred == a_lab5[sel]).float().numpy(), a_lab5[sel].numpy()
            return {"balanced_top1": ci_dict(correct, lab, True, nb), "majority_class_share": float(np.bincount(lab).max() / len(lab)),
                    "pred_hist": np.bincount(pred.numpy(), minlength=5).tolist(), "n": int(sel.sum())}

        g2_all, g2_test = g2_run(mapped_all), g2_run(mapped_test)
        pt, lo = g2_all["balanced_top1"]["point"], g2_all["balanced_top1"]["lo"]
        g2 = {"all_folds": g2_all, "test_fold": g2_test, "chance": 0.2, "point_ok": pt >= g.g2_min_ratio * 0.2, "ci_lo_ok": lo > 0.2}
        g2["pass"] = g2["point_ok"] and g2["ci_lo_ok"]
        sel = mapped_all
        refs = {  # which side is the weak link: audio->text-of-photo-prompts, native CLAP on the same 5 classes, image->text, image->audio
            "audio(S)->CLIP-text 'a photo of a {c}'": ci_dict(((S_a[sel] @ P5_photo.T).argmax(1) == a_lab5[sel]).float().numpy(), a_lab5[sel].numpy(), True, nb),
            "native CLAP audio->CLAP-text sound prompts": ci_dict(((A[sel] @ P5_clap_sound.T).argmax(1) == a_lab5[sel]).float().numpy(), a_lab5[sel].numpy(), True, nb),
            "image(S)->CLIP-text 'a photo of a {c}'": ci_dict(((img_s @ P5_photo.T).argmax(1) == img_lab5).float().numpy(), img_lab5.numpy(), True, nb),
        }
        protos_aud = class_prototypes(S_a[sel], a_lab5[sel], 5)
        refs["image(S)->audio prototypes (reverse direction)"] = ci_dict(((img_s @ protos_aud.T).argmax(1) == img_lab5).float().numpy(), img_lab5.numpy(), True, nb)

        pred1 = torch.tensor(rungs["L1 audio->bridge vs CLIP-text prompts (the gate)"]["pred"])
        pred0 = torch.tensor(native["L0 native CLAP (16 kHz source)"]["pred"])
        per_class = []
        for c in range(len(names)):
            m = ytest == c
            per_class.append({"class": names[c], "n": int(m.sum()), "native": float((pred0[m] == c).float().mean()), "S": float((pred1[m] == c).float().mean())})
        pairs = {}
        for t, p in zip(ytest.tolist(), pred1.tolist()):
            if t != p:
                pairs[(names[t], names[p])] = pairs.get((names[t], names[p]), 0) + 1
        hist = np.bincount(pred1.numpy(), minlength=len(names))
        breakdown = {"largest_drops": sorted(per_class, key=lambda r: r["native"] - r["S"], reverse=True)[:10],
                     "classes_below_60pct_of_native": int(sum(1 for r in per_class if r["S"] < g.g1_rel * r["native"])),
                     "classes_with_native_acc_gt_0": int(sum(1 for r in per_class if r["native"] > 0)),
                     "top_confusions": [{"true": a, "pred": b, "count": n} for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:10]],
                     "max_prediction_share": float(hist.max() / hist.sum()), "classes_never_predicted": int((hist == 0).sum()),
                     "bridge_kind": kind}
        seeds_out[str(seed)] = {"bridge_kind": kind, "rungs": {k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in rungs.items()},
                                "L1_all_folds": {k: v for k, v in l1_all.items() if k != "pred"}, "prompt_variants_L1_top1": l1_var,
                                "G1": g1, "G2": g2, "G2_references": refs, "breakdown": breakdown}
        print(f"[gate] seed {seed} ({kind}): G1 top-1 {l1['point']:.3f} [{l1['lo']:.3f}, {l1['hi']:.3f}] vs native {l0:.3f} -> "
              f"{'PASS' if g1['pass'] else 'FAIL'} | G2 balanced {pt:.3f} [{lo:.3f}, {g2_all['balanced_top1']['hi']:.3f}] -> "
              f"{'PASS' if g2['pass'] else 'FAIL'}", flush=True)


    # ------------------------------------------------------------------ controls: is the gate measuring anything?
    S_a, S_p_clip = kept_S[decision], cy.transform(P_clip)
    torch.manual_seed(0)
    untrained = TextBridge("linear").eval()  # same architecture and centering, weights never trained
    with torch.no_grad():
        S_rand = untrained(ca.transform(A))
    sel = mapped_all
    pred_g2 = (S_a[sel] @ protos_img.T).argmax(1)
    rng, perm_g1, perm_g2 = np.random.default_rng(0), [], []
    for _ in range(200):
        pl = ytest[torch.from_numpy(rng.permutation(len(ytest)))]
        perm_g1.append(float(topk_correct(S_a[test] @ S_p_clip.T, pl, 1).mean()))
        pl2 = a_lab5[sel][torch.from_numpy(rng.permutation(int(sel.sum())))]
        perm_g2.append(balanced_accuracy((pred_g2 == pl2).float().numpy(), pl2.numpy()))
    controls = {
        "G1_untrained_bridge_top1": ci_dict(topk_correct(S_rand[test] @ S_p_clip.T, ytest, 1), n_boot=nb),
        "G2_untrained_bridge_balanced_top1": ci_dict(((S_rand[sel] @ protos_img.T).argmax(1) == a_lab5[sel]).float().numpy(),
                                                     a_lab5[sel].numpy(), True, nb),
        "G1_permuted_labels_top1_mean_max": [float(np.mean(perm_g1)), float(np.max(perm_g1))],
        "G2_permuted_labels_balanced_mean_max": [float(np.mean(perm_g2)), float(np.max(perm_g2))],
        "chance": {"G1": 1 / len(names), "G2": 0.2},
    }
    print(f"[gate] controls: untrained bridge G1 {controls['G1_untrained_bridge_top1']['point']:.3f}, G2 "
          f"{controls['G2_untrained_bridge_balanced_top1']['point']:.3f}; permuted-label G2 mean {controls['G2_permuted_labels_balanced_mean_max'][0]:.3f}", flush=True)

    # ------------------------------------------------------------------ clean image/text checks in S
    clip_prompts = lambda tmpl, ns: pre.encode_text([tmpl.format(n) for n in ns]).cpu()  # noqa: E731
    checks = {}
    c100 = {s: load_anchors(anchor_path(cfg.anchor_root, "cifar100", s)) for s in ("train", "test")} if \
        anchor_path(cfg.anchor_root, "cifar100", "test").exists() else None
    checks["CIFAR-10"] = image_text_check("cifar10", c10_train, c10_test, c10_y, clip_prompts("a photo of a {}", c10_names), cy, nb)
    if c100 is not None:
        checks["CIFAR-100"] = image_text_check("cifar100", c100["train"]["anchors"], c100["test"]["anchors"], c100["test"]["labels"],
                                               clip_prompts("a photo of a {}", [c.replace("_", " ") for c in datasets.CIFAR100(cfg.data_root, train=False, download=False).classes]), cy, nb)
    mn_train = load_anchors(anchor_path(cfg.anchor_root, "mnist", "train"))["anchors"]
    mn_test = load_anchors(anchor_path(cfg.anchor_root, "mnist", "test"))["anchors"]
    mn_y = datasets.MNIST(cfg.data_root, train=False).targets.clone()
    checks["MNIST"] = image_text_check("mnist", mn_train, mn_test, mn_y, clip_prompts("a handwritten digit {}", range(10)), cy, nb)

    result = {
        "protocol": {"template": template, "decision_seed": decision, "seeds": [int(s) for s in cfg.seeds], "n_boot": nb,
                     "gate": OmegaConf.to_container(cfg.gate), "esc50_to_cifar10": ESC50_TO_CIFAR10,
                     "audio_centerer_fit_clips": int(ca.n_fit), "text_centerers_fit_captions": [int(cx.n_fit)]},
        "native": {k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in native.items()},
        "native_all_folds_top1": native_all["top1"], "prompt_variants_native_top1": {v: d["native_top1"] for v, d in variants.items()},
        "seeds": seeds_out, "controls": controls, "clean_image_text_checks": checks,
        "bridge_agreement": {s: {"chosen": d["chosen"], **d["kinds"][d["chosen"]]} for s, d in bridge_res["seeds"].items()},
        "bridge_ridge_reference": bridge_res["ridge_reference"],
        "gate_pass": bool(seeds_out[decision]["G1"]["pass"] and seeds_out[decision]["G2"]["pass"]),
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    run = init_run(cfg, make_run_name("audio", "esc50", "clean", "none", cfg.seed), project=cfg.tracking.project, mode=cfg.tracking.mode,
                   group="audio", job_type="stage0-gate")
    run.summary.update({"gate_pass": result["gate_pass"], "G1_top1": seeds_out[decision]["G1"]["top1"]["point"],
                        "G2_balanced_top1": seeds_out[decision]["G2"]["all_folds"]["balanced_top1"]["point"]})
    run.finish()
    print(f"[gate] results -> {out_dir / 'results.json'}; OVERALL GATE: {'PASS' if result['gate_pass'] else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
