# semcom

Develop generalised modality-agnostic semantic performance metrics to characterise the efficacy of an end-to-end semantic communication set-up.

Core metric under development: `M = g(F, U, E; channel_conditions)` — Fidelity, Task Utility, Efficiency, computed in a shared embedding space. This repo currently covers the infrastructure layer (dataset loaders, channel simulation, experiment tracking), the per-modality transceivers (Week 4) and the common semantic space (Week 5). The F/U/E metric logic lands in Weeks 6-7.

## Setup

Use the repo-local virtualenv (don't install into a shared conda base):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
```

`torchaudio` and the Edinburgh DataShare loader need network access on first use to download and cache datasets under `data_cache/` (gitignored).

## Running the smoke test

```bash
python experiments/week3_smoke_test.py
```

Swap dataset/channel via Hydra overrides, e.g.:

```bash
python experiments/week3_smoke_test.py dataset=mnist channel=rayleigh
```

## Per-Modality Transceivers

Each modality has an encoder/decoder pair behind the `Transceiver` interface in `src/semcom/encoders/base.py` (`encode()`, `decode()`, `forward(x, channel, snr_db)`), plus a `TaskSpec` that the generic driver uses for batching, the training loss and the task metric.

| task | model | dataset | task metric |
|---|---|---|---|
| `text` | DeepSC-style Transformer | Europarl (English side of `de-en`, 100k sentences) | BLEU-1/4, exact match |
| `cifar10` | Deep JSCC CNN (1/6 bandwidth ratio) | CIFAR-10 | PSNR, SSIM |
| `mnist` | task-oriented JSCC CNN + classifier head | MNIST | accuracy |
| `audio` | SE-ResNet (DeepSC-S-style) | Edinburgh/Valentini clean speech, 1 s crops @ 16 kHz | SI-SDR, STOI, PESQ |

Training is channel-in-the-loop with SNR drawn uniformly from 0–25 dB per batch, one model per (task, channel). The test metric is then evaluated on a grid of fixed SNRs.

```bash
python experiments/week4_train.py task=cifar10 channel=awgn      # or: text | mnist | audio, channel=rayleigh
python experiments/week4_plot.py                                  # AWGN vs Rayleigh curves per task
```

Outputs land in `outputs/week4/<task>/<channel>/` (`results.json`, `snr_curve.png`, `best.pt`). W&B runs are grouped by modality and default to `tracking.mode=offline`; after `wandb login`, set `tracking.mode=online` or `wandb sync wandb/offline-run-*`. Smoke-test any task quickly with `tracking.mode=disabled train.epochs=1 train.max_steps_per_epoch=40`.

## Week 5: common semantic space

`src/semcom/common_space/` puts every modality's source and received signal into a 512-d, L2-normalised space behind `embed_source(batch)` / `embed_received(batch, channel, snr_db)`. Image and text use frozen **CLIP** (`openai/clip-vit-base-patch32`) and audio uses frozen **CLAP** (`laion/clap-htsat-unfused`). Both are 512-d but they are *different, unaligned spaces*: cross-modal comparison is only valid between image and text, and cosine values are only comparable within a modality (see `random_pair_cosine` in the results).

- **A. Learned heads** (`LearnedHeadSpace`): a small head maps the frozen Week 4 transceiver's *received symbols* into the anchor space, trained with InfoNCE (+ an optional cosine term) against cached anchors, with the channel in the loop and SNR ~ U[0, 25] dB per batch. The transceivers, channel and anchors are frozen.
- **B. Pretrained on the decoded output** (`DirectPretrainedSpace`, no training): decoded text / reconstructed image / reconstructed waveform through CLIP/CLAP. Not applicable to MNIST (its `decode()` returns class logits).

```bash
python experiments/week5_cache_anchors.py common=cifar10             # once per task: cifar10 | mnist | text | audio
python experiments/week5_train_heads.py common=cifar10 channel=awgn  # loss.cos_weight=0 for the InfoNCE-only ablation
python experiments/week5_eval.py common=cifar10 channel=awgn         # ceiling vs A vs B per SNR (run on MPS)
python experiments/week5_plot.py                                      # retrieval-vs-SNR curves per task
```

Methods notes, including the input soft-clip used for Rayleigh and what the data can and cannot support: [docs/week5_common_space.md](docs/week5_common_space.md).

## Week 5b: a shared space S for text, audio and image (Stage 0 done)

S is the per-domain mean-centered CLIP space. Image and text are native to it; audio enters through CLAP, per-modality centering and a CLAP-to-CLIP bridge trained only on caption text. Stage 0 checks that this works on clean data before any channel experiment. Report (generated from `results.json`): [docs/week5b_stage0.md](docs/week5b_stage0.md).

```bash
python experiments/week5b_prepare.py   # caption corpus, text/ESC-50/CIFAR-100 embeddings (downloads ESC-50, caption text, CIFAR-100)
python experiments/week5b_bridge.py    # train the CLAP-text -> CLIP-text bridge (linear and MLP, 3 seeds)
python experiments/week5b_gate.py      # clean cross-modal gate (audio->text, audio->image) with controls
python experiments/week5b_report.py    # regenerate docs/week5b_stage0.md
```

## Week 5b Stage 1: new transceivers, S anchors, heads, Fidelity selection

Adds a caption-domain text transceiver and an ESC-50 event-audio transceiver (folds 1-3 train, 4 validation, 5 test only), caches per-domain centering statistics fitted on train data only, trains three head variants per task and channel, and compares Fidelity candidates (F_abs headline, F_ret, clean ceiling C, raw and centered cosine). Report (generated from the results files): [docs/week5b_stage1.md](docs/week5b_stage1.md).

```bash
python experiments/week4_train.py task=caption_text channel=awgn      # also: task=esc50_audio, channel=rayleigh
python experiments/week5b_stage1_space.py                             # domain centering + anchors in S (train data only)
python experiments/week5b_sanity.py                                   # transceiver reconstruction gate (writes outputs/week5b/stage1/sanity.json)
python experiments/week5_train_heads.py common=caption_text channel=awgn target=centered   # target=raw|centered, loss.cos_weight=0.5|5
python experiments/week5b_fselect.py                                  # F statistics per task x channel x variant x split
python experiments/week5b_report_stage1.py                            # regenerates docs/week5b_stage1.md and the variant choice
```

## Week 5b Stage 2: co-primary variants, 3 seeds, cross-modal channel experiments

Trains `cos5` and `cen` heads for 3 seeds on every task and channel, builds embedding caches (items x 5 noise draws x SNR grid, in S) for the heads (approach A) and for the pretrained encoders on the decoded output (approach B), and computes every statistic from those caches with percentile bootstrap CIs over items and a separate head-seed spread. The amended variant rule and the held-out paraphrase prompts were written to `outputs/week5b/stage2/protocol_declared.json` before any Stage 2 seed was run. Report: [docs/week5b_stage2.md](docs/week5b_stage2.md).

```bash
python experiments/week5b_s2_declare.py                       # once, before any Stage 2 run (refuses to overwrite)
python experiments/week5b_s2_build_caches.py common_s2=cifar10  # clean-symbol caches for every task (any common_s2 value satisfies the config group)
python experiments/week5b_s2_scheduler.py --workers 4         # resumable head training; applies the ESC-50 regularisation rule itself
python experiments/week5b_s2_embed.py kind=A                  # head embeddings (also kind=promptsA)
python experiments/week5b_s2_embed.py kind=B 'tasks=[cifar10]' 'splits=[test]'   # pretrained encoder on decoded output (also kind=promptsB)
python experiments/week5b_s2_metrics.py section=F             # also section=cross, section=within
python experiments/week5b_s2_report.py                        # regenerates docs/week5b_stage2.md and docs/figures_week5b_stage2/
```

Everything long-running is resumable: a head is done iff its `train_*.json` exists, and embedding files that exist are skipped. The machine has 18 GB of memory and each head process holds 4-5 GB, so the scheduler admits a new job only while at least 25% of memory is free.

## Tests

```bash
pytest                 # fast tests only (no downloads)
pytest --run-slow       # include dataset-loader tests that download data
```
