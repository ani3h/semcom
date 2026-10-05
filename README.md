# semcom

Develop generalised modality-agnostic semantic performance metrics to characterise the efficacy of an end-to-end semantic communication set-up.

Core metric under development: `M = g(F, U, E; channel_conditions)` — Fidelity, Task Utility, Efficiency, computed in a shared embedding space. This repo currently covers infrastructure layer: dataset loaders, channel simulation, and experiment tracking. Encoders and metric logic land in later weeks.

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

## Tests

```bash
pytest                 # fast tests only (no downloads)
pytest --run-slow       # include dataset-loader tests that download data
```
