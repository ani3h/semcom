"""Stage 1, step 2: reconstruction-quality sanity gate for the two NEW transceivers (caption-domain text, ESC-50 event audio),
per SNR and channel, before any head is trained.

    python experiments/week5b_sanity.py

Text: token accuracy, BLEU-1, exact match, and the CLIP-text embedding of the decoded sentence against the original's anchor in S
(centered cosine + recall within a gallery), plus exact-recovery of every class-prompt set. Audio: SI-SDR, log-spectral distance and the
CLAP embedding of the decoded clip against the original's anchor in S. Decisions use the VALIDATION split; test is reported only.
Both transceivers are single-seed, so intervals elsewhere exclude transceiver-training variance.
"""
import json
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from semcom.common_space.adapters import make_adapter
from semcom.common_space.anchors import PretrainedAnchors, anchor_path, load_anchors
from semcom.common_space.domains import get_domain
from semcom.common_space.loader import load_week4, make_dataset
from semcom.common_space.retrieval import alignment, recall_at_k
from semcom.data.prompts import prompt_sets
from semcom.encoders.audio.metrics import log_spectral_distance, si_sdr
from semcom.encoders.text.task import bleu, token_accuracy
from semcom.encoders.text.tokenizer import PAD_ID


def pick_device(name: str) -> torch.device:
    return torch.device(name) if name != "auto" else torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def text_point(w4, adapter, domain, items, a_S, channel, snr, seed, device, gallery=1000) -> dict:
    torch.manual_seed(seed)
    batch = w4.spec.to_device(w4.spec.collate(items), device)
    src = batch["src"]
    ids = w4.model(src, channel, snr, src_mask=src != PAD_ID).cpu()
    tok = w4.spec.tokenizer
    hyps = [" ".join(tok.ids_to_tokens(p)) for p in ids]
    refs = [" ".join(tok.ids_to_tokens(t)) for t in batch["tgt_out"].cpu()]
    emb = F.normalize(adapter.embed_decoded(ids), dim=-1).cpu()  # CLIP of the decoded sentence
    z = domain.to_S(emb)
    by_source = {}
    for src_name in sorted({it["raw"] for it in items}):
        idx = [i for i, it in enumerate(items) if it["raw"] == src_name]
        by_source[src_name] = token_accuracy([hyps[i] for i in idx], [refs[i] for i in idx])
    return {"token_acc": token_accuracy(hyps, refs), "bleu1": bleu(hyps, refs, 1), "exact_match": float(np.mean([h == r for h, r in zip(hyps, refs)])),
            "token_acc_by_source": by_source, **alignment(z, a_S), **recall_at_k(z, a_S, ks=(1, 5)), "n": len(items)}


@torch.no_grad()
def prompt_recovery(w4, channel, snr, seed, device, draws, root) -> dict:
    out = {}
    tok = w4.spec.tokenizer
    for name, prompts in prompt_sets(root).items():
        batch = w4.spec.to_device(w4.spec.collate([{"input": p} for p in prompts]), device)
        ok = []
        for d in range(draws):
            torch.manual_seed(seed + 97 * d)
            ids = w4.model(batch["src"], channel, snr, src_mask=batch["src"] != PAD_ID).cpu()
            hyps = [" ".join(tok.ids_to_tokens(p)) for p in ids]
            ok.append(np.mean([h == " ".join(tok.ids_to_tokens(t)) for h, t in zip(hyps, batch["tgt_out"].cpu())]))
        out[name] = float(np.mean(ok))
    return out


@torch.no_grad()
def audio_point(w4, adapter, domain, x, a_S, a_raw, channel, snr, seed, device) -> dict:
    torch.manual_seed(seed)
    est = torch.cat([w4.model(x[s:s + 50].to(device), channel, snr).cpu() for s in range(0, len(x), 50)])
    emb = adapter.anchors.encode_audio(est, 16000).cpu()  # CLAP of the decoded clip
    z = domain.to_S(emb)
    n = len(x)
    return {"si_sdr": si_sdr(est, x).mean().item(), "lsd": log_spectral_distance(est, x).mean().item(),
            "clap_raw_cosine": (emb * a_raw).sum(-1).mean().item(), **alignment(z, a_S), **recall_at_k(z, a_S, ks=(1, 5, 10)), "n": n}


@hydra.main(config_path="../configs", config_name="week5b", version_base=None)
def main(cfg: DictConfig) -> None:
    device = pick_device(cfg.device)
    space, bridge_dir = Path(cfg.out_root) / "space", Path(cfg.out_root) / "bridge"
    out_dir = Path(cfg.out_root) / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = PretrainedAnchors(device=device)
    grid, draws = [float(s) for s in (0, 5, 10, 15, 20, 25)], int(cfg.stage1.noise_draws)
    for task in list(cfg.get("sanity_tasks", ["caption_text", "esc50_audio"])):
        results = {"tasks": {task: {}}}
        domain = get_domain(task, cfg.anchor_root, space, bridge_dir)
        for ch_name in ("awgn", "rayleigh"):
            w4 = load_week4(task, ch_name, cfg.w4_root, device)
            channel = hydra.utils.instantiate(OmegaConf.load(f"configs/channel/{ch_name}.yaml"))
            adapter = make_adapter(task, pre, w4.spec)
            per_split = {}
            for split in ("val", "test"):
                ds = make_dataset(w4, split, cfg.data_root, {"prompt_repeat": 0} if task == "caption_text" else None)
                n = min(len(ds), int(cfg.stage1.f_items))
                items = [ds[i] for i in range(n)]
                a_raw = load_anchors(anchor_path(cfg.anchor_root, task, split))["anchors"][:n]
                a_S = domain.to_S(a_raw)
                rows = {}
                for snr in grid:
                    seed = int(cfg.seed) * 100 + int(snr)
                    if task == "caption_text":
                        rows[str(snr)] = text_point(w4, adapter, domain, items, a_S, channel, snr, seed, device)
                    else:
                        x = torch.stack([it["input"] for it in items])
                        rows[str(snr)] = audio_point(w4, adapter, domain, x, a_S, a_raw, channel, snr, seed, device)
                per_split[split] = rows
                print(f"[sanity] {task}/{ch_name}/{split}: " + " | ".join(
                    f"{int(float(s))} dB " + (f"tokacc {r['token_acc']:.2f} bleu1 {r['bleu1']:.2f}" if task == "caption_text" else f"sisdr {r['si_sdr']:.1f} lsd {r['lsd']:.1f}")
                    + f" cos {r['cos_to_anchor']:.2f} R@1 {r['recall@1']:.2f}" for s, r in list(rows.items())[::5]), flush=True)
            entry = {"splits": per_split}
            if task == "caption_text":
                entry["prompt_exact_recovery"] = {str(snr): prompt_recovery(w4, channel, snr, int(cfg.seed) * 100 + int(snr), device, draws, cfg.data_root) for snr in grid}
                entry["vocab_report"] = w4.spec.vocab_report
            results["tasks"][task][ch_name] = entry
        (out_dir / f"sanity_{task}.json").write_text(json.dumps(results))

    # ---- verdict (needs both tasks): pre-registered thresholds, validation split, 25 dB AWGN
    files = [out_dir / f"sanity_{t}.json" for t in ("caption_text", "esc50_audio")]
    if not all(f.exists() for f in files):
        print("[sanity] one task still missing; no verdict yet", flush=True)
        return
    results = {"thresholds": OmegaConf.to_container(cfg.stage1.sanity), "tasks": {}}
    for f in files:
        results["tasks"].update(json.loads(f.read_text())["tasks"])
    th, verdict = cfg.stage1.sanity, {}
    t = results["tasks"]["caption_text"]["awgn"]["splits"]["val"]["25.0"]
    a = results["tasks"]["esc50_audio"]["awgn"]["splits"]["val"]["25.0"]
    verdict["caption_text"] = {"bleu1": t["bleu1"], "token_acc": t["token_acc"],
                               "broken": not (t["bleu1"] >= th.text.min_bleu1 and t["token_acc"] >= th.text.min_token_acc)}
    verdict["esc50_audio"] = {"si_sdr": a["si_sdr"], "s_cosine": a["cos_to_anchor"], "recall@1": a["recall@1"],
                              "broken": not (a["si_sdr"] >= th.audio.min_si_sdr_db and a["cos_to_anchor"] >= th.audio.min_s_cosine
                                             and a["recall@1"] >= th.audio.min_recall_at_1)}
    results["verdict"] = verdict
    results["proceed"] = not any(v["broken"] for v in verdict.values())
    (out_dir / "sanity.json").write_text(json.dumps(results, indent=2))
    print(f"[sanity] verdict: {json.dumps(verdict)} -> {'PROCEED' if results['proceed'] else 'STOP'}", flush=True)


if __name__ == "__main__":
    main()
