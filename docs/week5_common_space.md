# Week 5: common semantic space (methods notes)

Code: `src/semcom/common_space/`. Drivers: `experiments/week5_*.py`. Configs: `configs/week5.yaml`, `configs/common/`.
Results: `outputs/week5/<task>/<channel>/results.json` and `outputs/week5/<task>/retrieval_curves.png` (gitignored).

## What was built

A `CommonSpace` interface with `embed_source(batch)` and `embed_received(batch, channel, snr_db)`, returning 512-d L2-normalised
vectors, with two implementations behind it:

- **A, `LearnedHeadSpace` (primary)**: a small head maps the *received symbols* of the frozen Week 4 transceiver into the anchor space.
- **B, `DirectPretrainedSpace` (baseline, no training)**: the transceiver's decoded output (text, reconstructed image, reconstructed
  waveform) goes through the frozen pretrained CLIP/CLAP. Not applicable to MNIST: its `decode()` returns class logits.

`embed_source` is the same in both: the frozen anchor of the clean original (for text, the *raw* sentence, not its tokenizer
round trip). F therefore measures encoder loss plus channel loss. The Week 4 transceivers and the Week 3 channel are untouched.

## Spaces, anchors, and the limit this implies

Image and text use CLIP `openai/clip-vit-base-patch32`; audio uses CLAP `laion/clap-htsat-unfused`. Both are 512-d, but they are
**different, unaligned spaces**. Cross-modal retrieval is only meaningful between image and text. Cosine values are only comparable within
a modality, so F values from different modalities are not directly comparable without normalisation. The baseline similarity of each
space (mean cosine between random unrelated pairs) is logged for that purpose:

| task | anchor space | A main | A cos5 | A InfoNCE-only | B |
|---|---|---|---|---|---|
| cifar10 | 0.774 | 0.055 | 0.663 | 0.021 | 0.853 |
| mnist | 0.950 | 0.041 | 0.717 | 0.031 | n/a |
| text | 0.800 | 0.028 | 0.631 | 0.021 | 0.818 |
| audio | 0.801 | 0.340 | 0.772 | 0.242 | 0.812 |

(Anchor column: test split. Other columns: embeddings of the received signal at 10 dB, AWGN.)

Anchors are computed once and cached by dataset index (`data_cache/anchors/`). Audio training uses 3 fixed random 1 s crops per
utterance because the loader's random crops cannot be cached. CLIP preprocessing: grey to RGB, bicubic to 224 px, CLIP mean/std.
CLAP: 16 kHz resampled to 48 kHz, CLAP pads/repeats a 1 s clip to 10 s. CLIP and CLAP run on MPS; their outputs matched CPU
(min cosine 1.00000) with real weights. Fresh vs cached anchors also match (cosine 1.00000).

## Head input conditioning: the Rayleigh soft-clip

Every head first applies `x -> c * tanh(x / sigma / c)` with `c = 4` and `sigma` = the std of the transceiver's clean (pre-channel)
symbols, a per-model constant measured on 2000 training items. Reason: the Rayleigh channel zero-forces the fading gain, so deep fades
produce enormous outliers in the received symbols (up to about 530 sigma for audio at 0 dB, still about 166 sigma at 25 dB). The
channel returns only equalized symbols, so a head cannot use the fading gain to down-weight them. The clip is near-identity for
|x| << 4 sigma and bounds everything else at 4. **This is a design choice that shapes the Rayleigh results**; it is applied identically
to AWGN heads.

## Heads, loss, training

| task | head | params |
|---|---|---|
| CIFAR-10 | MLP 1024-1024-512 on the flat symbols | ~1.6M |
| MNIST | MLP 256-512-512 | ~0.4M |
| text | per-token Linear(16,256) + learned positions + 1 self-attention layer + masked mean pool + Linear(256,512) | ~0.7M |
| audio | Conv1d(64,256,k5) over frames + mean/std pooling + Linear(512,512) | ~0.3M |

Loss: symmetric InfoNCE against the cached anchors with a learnable temperature (init 0.07, clamped to [0.01, 0.5]), plus
`cos_weight * (1 - cos)`. Three variants were trained per task and channel (24 heads): `cos_weight = 0.5` (the approved main run),
`0` (InfoNCE only) and `5` (ablation). The two loss terms are logged separately. Training: AdamW, lr 1e-3, weight decay 1e-4,
100 or 200 warmup steps, cosine decay, batch 512 (audio 256), SNR ~ U[0, 25] dB per batch, per channel (each head sits on the
matching Week 4 transceiver). Best checkpoint by validation loss of its own objective at 10 dB; the learned temperature ends at 0.010-0.013 (the clamp floor is 0.01). Epochs: 40 (CIFAR-10, MNIST), 20
(text), 30 (audio). The transceiver, channel and anchors are frozen; a test checks that gradients reach only the head.

## Evaluation

All on MPS, one noise draw per item, seeded per SNR. Image tasks: class-level image-to-text retrieval over the 10 prompts
("a photo of a {class}", "a handwritten digit {n}"). Text: self-retrieval of the original sentence among the first 1000 test
sentences (ceiling = CLIP embedding of the tokenizer round trip of the raw sentence, mean cosine 0.99 to the raw one). Audio: clean vs
transmitted retrieval among all 824 test clips (centre 1 s crops). Single seed, one noise realisation, no confidence intervals.

## Results (test set)

**CIFAR-10 class retrieval top-1 (chance 0.10)**; margin = cosine to own anchor minus cosine to unrelated anchors

| method | channel | 0 dB | 10 dB | 25 dB | margin at 0 / 10 / 25 dB |
|---|---|---|---|---|---|
| clean-input ceiling | any | 0.883 | 0.883 | 0.883 | 0.227 / 0.227 / 0.227 |
| A: InfoNCE + 0.5 cos (main) | AWGN | 0.376 | 0.459 | 0.467 | 0.119 / 0.145 / 0.149 |
| A: InfoNCE + 0.5 cos (main) | Rayleigh | 0.318 | 0.403 | 0.428 | 0.094 / 0.124 / 0.132 |
| A: InfoNCE only | AWGN | 0.322 | 0.397 | 0.407 | 0.106 / 0.130 / 0.134 |
| A: InfoNCE only | Rayleigh | 0.294 | 0.375 | 0.397 | 0.092 / 0.121 / 0.128 |
| A: InfoNCE + 5 cos (ablation) | AWGN | 0.400 | 0.480 | 0.495 | 0.084 / 0.108 / 0.112 |
| A: InfoNCE + 5 cos (ablation) | Rayleigh | 0.352 | 0.451 | 0.480 | 0.072 / 0.099 / 0.107 |
| B: pretrained on decoded output | AWGN | 0.415 | 0.628 | 0.658 | 0.046 / 0.099 / 0.114 |
| B: pretrained on decoded output | Rayleigh | 0.312 | 0.478 | 0.542 | 0.030 / 0.061 / 0.078 |

**MNIST class retrieval top-1 (chance 0.10)**; margin = cosine to own anchor minus cosine to unrelated anchors

| method | channel | 0 dB | 10 dB | 25 dB | margin at 0 / 10 / 25 dB |
|---|---|---|---|---|---|
| clean-input ceiling | any | 0.357 | 0.357 | 0.357 | 0.050 / 0.050 / 0.050 |
| A: InfoNCE + 0.5 cos (main) | AWGN | 0.361 | 0.376 | 0.373 | 0.091 / 0.114 / 0.117 |
| A: InfoNCE + 0.5 cos (main) | Rayleigh | 0.373 | 0.406 | 0.409 | 0.075 / 0.103 / 0.111 |
| A: InfoNCE only | AWGN | 0.352 | 0.361 | 0.363 | 0.088 / 0.111 / 0.114 |
| A: InfoNCE only | Rayleigh | 0.372 | 0.396 | 0.396 | 0.074 / 0.101 / 0.109 |
| A: InfoNCE + 5 cos (ablation) | AWGN | 0.409 | 0.470 | 0.472 | 0.063 / 0.080 / 0.082 |
| A: InfoNCE + 5 cos (ablation) | Rayleigh | 0.417 | 0.492 | 0.509 | 0.052 / 0.071 / 0.077 |
| B | - | n/a: MNIST `decode()` returns class logits | | | |

**Text self-retrieval recall@1 (N=1000, chance 0.001)**; margin = cosine to own anchor minus cosine to unrelated anchors

| method | channel | 0 dB | 10 dB | 25 dB | margin at 0 / 10 / 25 dB |
|---|---|---|---|---|---|
| clean-input ceiling | any | 0.975 | 0.975 | 0.975 | 0.186 / 0.186 / 0.186 |
| A: InfoNCE + 0.5 cos (main) | AWGN | 0.174 | 0.659 | 0.760 | 0.088 / 0.139 / 0.149 |
| A: InfoNCE + 0.5 cos (main) | Rayleigh | 0.134 | 0.542 | 0.707 | 0.082 / 0.127 / 0.141 |
| A: InfoNCE only | AWGN | 0.183 | 0.656 | 0.762 | 0.088 / 0.139 / 0.149 |
| A: InfoNCE only | Rayleigh | 0.121 | 0.538 | 0.713 | 0.082 / 0.127 / 0.141 |
| A: InfoNCE + 5 cos (ablation) | AWGN | 0.117 | 0.455 | 0.525 | 0.067 / 0.108 / 0.117 |
| A: InfoNCE + 5 cos (ablation) | Rayleigh | 0.082 | 0.343 | 0.447 | 0.060 / 0.096 / 0.108 |
| B: pretrained on decoded output | AWGN | 0.042 | 0.844 | 0.913 | 0.030 / 0.147 / 0.166 |
| B: pretrained on decoded output | Rayleigh | 0.012 | 0.287 | 0.655 | 0.026 / 0.076 / 0.120 |

**Audio clean-vs-transmitted recall@1 (N=824, chance 0.001)**; margin = cosine to own anchor minus cosine to unrelated anchors

| method | channel | 0 dB | 10 dB | 25 dB | margin at 0 / 10 / 25 dB |
|---|---|---|---|---|---|
| clean-input ceiling | any | 1.000 | 1.000 | 1.000 | 0.199 / 0.199 / 0.199 |
| A: InfoNCE + 0.5 cos (main) | AWGN | 0.087 | 0.154 | 0.186 | 0.103 / 0.144 / 0.149 |
| A: InfoNCE + 0.5 cos (main) | Rayleigh | 0.061 | 0.108 | 0.121 | 0.079 / 0.121 / 0.130 |
| A: InfoNCE only | AWGN | 0.096 | 0.161 | 0.189 | 0.100 / 0.130 / 0.134 |
| A: InfoNCE only | Rayleigh | 0.068 | 0.112 | 0.125 | 0.076 / 0.110 / 0.116 |
| A: InfoNCE + 5 cos (ablation) | AWGN | 0.057 | 0.120 | 0.133 | 0.098 / 0.140 / 0.147 |
| A: InfoNCE + 5 cos (ablation) | Rayleigh | 0.032 | 0.062 | 0.084 | 0.084 / 0.129 / 0.141 |
| B: pretrained on decoded output | AWGN | 0.018 | 0.027 | 0.036 | 0.080 / 0.104 / 0.112 |
| B: pretrained on decoded output | Rayleigh | 0.012 | 0.013 | 0.019 | 0.051 / 0.070 / 0.076 |

Full curves and the extra metrics (top-5, recall@5/10, cosine to anchor, unrelated cosine) are in the plots and `results.json`.

## What the data cannot support, and known caveats

- **Cross-modal retrieval beyond image-text.** There is no paired audio-text/image data, and CLAP is a different space from CLIP.
- **Image-text retrieval is class-level.** Top-5 over 10 prompts has a 0.5 chance level, so top-1 is the informative number. CLIP's
  own zero-shot ceiling is 0.883 on CIFAR-10 but only 0.357 on MNIST, so the MNIST ceiling is weak. A can exceed it (likely because the head
  regresses towards class-level structure and so averages away per-image CLIP noise; not verified).
- **Audio-transcript retrieval was deliberately not run.** A 1 s crop covers only about 40% of a 2.5 s utterance and CLAP is not a
  speech-content model, so a chance-level result could not show either way. Documented as a known CLAP limitation. Audio retrieval is
  weak throughout (CLAP embeddings of 1 s speech clips are tightly clustered; the test set is 2 unseen speakers).
- **CIFAR-10 A is bounded by the Week 4 encoder**, which was trained for pixel MSE: an MLP probe on its clean symbols reaches 54%
  class accuracy. The Week 4 transceivers were not fine-tuned, to keep the Week 4 baselines comparable.
- **Absolute cosine versus ranking.** With the approved 0.5 weight, the head outputs sit far from the anchor manifold (cosine to own
  anchor 0.2-0.3 while the anchor space's own baseline is 0.77-0.95); retrieval is competitive but the raw cosine is not a usable
  Fidelity distance. Weight 5 restores the absolute alignment (0.8-0.9) but is nearly flat in SNR and costs text/audio retrieval. The
  margin (own minus unrelated cosine) is the quantity that varies with SNR in every setting.
- **Deviations from the written plan:** 40 epochs for the two image tasks (a convergence probe showed InfoNCE still improving at 20);
  the weight-5 ablation and the cosine-to-unrelated / margin metrics were added after the first MNIST evaluation.
