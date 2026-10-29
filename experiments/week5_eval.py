"""Retrieval sanity checks for the common space: clean-input ceiling vs approach A (learned head, both loss
variants) vs approach B (pretrained model on the decoded output), per SNR, for one task and channel.

    python experiments/week5_eval.py common=cifar10 channel=rayleigh

Run on MPS (default) so the channel noise stream matches the Week 4 evaluations.
  image tasks : class-level image->text retrieval over the class prompts (top-1/top-5), MNIST has no approach B
  text        : self-retrieval of the original sentence among eval.max_items sentences (recall@k)
  audio       : clean<->transmitted retrieval among all test clips (recall@k); no audio<->transcript test
"""
import json
import time
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from semcom.common_space.adapters import make_adapter
from semcom.common_space.anchors import PretrainedAnchors, anchor_path, load_anchors
from semcom.common_space.direct import DirectPretrainedSpace
from semcom.common_space.heads import build_head
from semcom.common_space.learned import LearnedHeadSpace
from semcom.common_space.loader import load_week4, make_dataset
from semcom.common_space.retrieval import alignment, class_topk, random_pair_cosine, recall_at_k
from semcom.tracking.run_naming import make_run_name
from semcom.tracking.seed import set_seed
from semcom.tracking.wandb_logger import init_run

IMAGE_TASKS = ("cifar10", "mnist")


def pick_device(name: str) -> torch.device:
    return torch.device(name) if name != "auto" else torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def embed_all(fn, items, w4, device, batch_size) -> torch.Tensor:
    outs = []
    for s in range(0, len(items), batch_size):
        batch = w4.spec.to_device(w4.spec.collate(items[s:s + batch_size]), device)
        outs.append(fn(batch).detach())
    return torch.cat(outs)


def load_head_space(path: Path, w4, adapter, device) -> LearnedHeadSpace:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    hc = {k: v for k, v in ck["head_cfg"].items() if k != "kind"}
    head = build_head(ck["head_cfg"]["kind"], sigma=ck["sigma"], **hc)
    head.load_state_dict(ck["head"])
    return LearnedHeadSpace(w4.model, head.to(device).eval(), adapter)


@hydra.main(config_path="../configs", config_name="week5", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    if device.type != "mps":
        print(f"WARNING: evaluating on {device}; Week 4/5 numbers are comparable only on MPS (device-dependent noise stream)")
    task, channel_name = cfg.task, HydraConfig.get().runtime.choices["channel"]
    channel = hydra.utils.instantiate(cfg.channel)
    out_dir = Path(cfg.out_root) / task / channel_name

    w4 = load_week4(task, channel_name, cfg.w4_root, device)
    pretrained = PretrainedAnchors(cfg.anchors.clip, cfg.anchors.clap, device)
    adapter = make_adapter(task, pretrained, w4.spec)
    test_ds = make_dataset(w4, "test", cfg.data_root)
    n = len(test_ds) if cfg.eval.max_items is None else min(cfg.eval.max_items, len(test_ds))
    items = [test_ds[i] for i in range(n)]
    blob = load_anchors(anchor_path(cfg.anchor_root, task, "test"))
    anchors = blob["anchors"][:n].to(device)

    if task in IMAGE_TASKS:
        labels = torch.stack([it["label"] for it in items]).to(device)
        prompt_emb = pretrained.encode_text(list(cfg.eval.prompts)).to(device)
        score = lambda emb: {**class_topk(emb, prompt_emb, labels), **alignment(emb, anchors)}  # noqa: E731
        ceiling_emb = anchors
    else:
        score = lambda emb: {**recall_at_k(emb, anchors, ks=(1, 5, 10)), **alignment(emb, anchors)}  # noqa: E731
        # text: the best the transceiver could do is the tokenizer round trip of the raw sentence; audio: identity
        ceiling_emb = F.normalize(blob["anchors_roundtrip"][:n].float(), dim=-1).to(device) if task == "text" else anchors

    spaces: dict[str, object] = {}
    for path in sorted(out_dir.glob("head_*.pt"), key=lambda p: (p.stem != "head_cos", p.stem)):  # main run first
        spaces[f"A_{path.stem.removeprefix('head_')}"] = load_head_space(path, w4, adapter, device)
    notes = {}
    if adapter.supports_direct:
        spaces["B"] = DirectPretrainedSpace(w4.model, adapter)
    else:
        notes["B"] = "not applicable: MNIST decode() returns class logits, not a reconstruction"

    results = {"ceiling": {}, **{name: {} for name in spaces}}
    rpc = {"anchors": random_pair_cosine(anchors.cpu()), "ceiling": random_pair_cosine(ceiling_emb.cpu())}
    rpc.update({name: {} for name in spaces})
    ceiling_scores = score(ceiling_emb)

    run_name = make_run_name(adapter.modality, cfg.dataset_name, channel_name, "U0-25", cfg.seed)
    run = init_run(cfg, run_name, project=cfg.tracking.project, mode=cfg.tracking.mode, group=adapter.modality,
                   job_type="common-space-eval", tags=["eval", task])
    run.define_metric("eval/*", step_metric="snr_db")
    print(f"[{run_name}] methods: ceiling + {list(spaces)} on {device}; n={n}; "
          f"random-pair cos of the anchor space: {rpc['anchors']:.3f}", flush=True)

    t0 = time.time()
    for snr in cfg.eval.snr_grid_db:
        snr = float(snr)
        log = {"snr_db": snr}
        results["ceiling"][snr] = ceiling_scores
        log.update({f"eval/ceiling/{k}": v for k, v in ceiling_scores.items()})
        for name, space in spaces.items():
            torch.manual_seed(cfg.eval.seed + int(round(snr * 10)))
            emb = embed_all(lambda b: space.embed_received(b, channel, snr), items, w4, device, cfg.eval.batch_size)
            results[name][snr] = score(emb)
            rpc[name][snr] = random_pair_cosine(emb.cpu())
            log.update({f"eval/{name}/{k}": v for k, v in results[name][snr].items()})
        run.log(log)
        print(f"  {snr:>4.0f} dB: " + " | ".join(
            f"{name} " + " ".join(f"{k}={v:.3f}" for k, v in (ceiling_scores if name == 'ceiling' else results[name][snr]).items()
                                    if k in ("top1", "recall@1", "cos_to_anchor"))
            for name in ["ceiling", *spaces]) + f"  [{time.time() - t0:.0f}s]", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(
        {"task": task, "channel": channel_name, "n_eval": n, "device": str(device), "notes": notes,
         "random_pair_cosine": rpc,
         "results": {m: {str(s): v for s, v in by_snr.items()} for m, by_snr in results.items()}}, indent=2))
    run.summary["random_pair_cosine_anchors"] = rpc["anchors"]
    run.finish()
    print(f"Done. Results in {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
