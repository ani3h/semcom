# Week 5b, Stage 1: new transceivers, shared-space anchors, heads and Fidelity selection

**Transceiver sanity gate: PASS (proceed).** caption_text: bleu1 0.996, token_acc 0.995 (ok); esc50_audio: si_sdr 11.200, s_cosine 0.613, recall@1 0.572 (ok). **Variant chosen for Stage 2: `cos5` (cos 5 (ablation, raw targets))**, chosen on a single head seed. All numbers below are generated from `outputs/week5b/{stage1,space}/*.json`.

Both new transceivers are single-seed, so every interval in this document and in Stage 2 excludes transceiver-training variance.

## 1. Caption-domain text transceiver: vocabulary

| item | value |
|---|---|
| vocabulary size (incl. 4 special tokens) | 8848 |
| training sentences (caption train split minus a 5,000-caption monitoring hold-out, plus prompt sentences) | 193338 |
| distinct class-prompt words forced into the vocabulary | 193 |
| of those, absent from the training sentences | 0 |
| `<unk>` tokens in prompt set `cifar10_photo` (10 prompts) | 0 |
| `<unk>` tokens in prompt set `cifar100_photo` (100 prompts) | 0 |
| `<unk>` tokens in prompt set `mnist_digit` (10 prompts) | 0 |
| `<unk>` tokens in prompt set `cifar10_sound` (10 prompts) | 0 |
| `<unk>` tokens in prompt set `esc50 This is a sound of {}.` (50 prompts) | 0 |
| `<unk>` tokens in prompt set `esc50 {}` (50 prompts) | 0 |
| `<unk>` tokens in prompt set `esc50 the sound of {}` (50 prompts) | 0 |

The vocabulary is the union of the words seen at least 3 times in the caption train split and every word of every class-prompt set (CIFAR-10, CIFAR-100, MNIST, ESC-50 with all three templates); training asserts zero `<unk>` on every prompt set. The held-out caption splits (corpus val and test) were not used for the vocabulary or for any training or checkpoint selection of the transceiver. **Deviation:** the 280 distinct prompt sentences were also added to the transceiver's training split (×10, train split only), because many prompt words occur fewer than 3 times, or never, in the captions, and a vocabulary entry that is never trained cannot be transmitted. Heads and anchors never see the prompts. Consequence: prompt sets are *in the training distribution* of the text transceiver, so prompt recovery below is optimistic for unseen text.

## 2. Transceiver sanity gate (reconstruction quality per SNR and channel)

Pre-registered stop thresholds (judgement calls, fixed before any result; validation split, 25 dB AWGN): text BLEU-1 ≥ 0.5 and token accuracy ≥ 0.5; audio SI-SDR ≥ 0.0 dB, cosine to the anchor in S ≥ 0.2 and decoded→original recall@1 ≥ 0.1. Decisions use the validation split; the test split is reported only (ESC-50 fold 5 is touched by no training or selection).

**Caption text (new), val split (n = 1000)**; cos/margin/recall are the decoded signal's CLIP/CLAP embedding, mapped into S, against the original's S anchor (recall within a gallery of 1000).

| channel | SNR (dB) | token_acc | bleu1 | exact_match | cos_to_anchor | margin | recall@1 | recall@5 |
|---|---|---|---|---|---|---|---|---|
| awgn | 0 | 0.462 | 0.595 | 0.029 | 0.462 | 0.457 | 0.396 | 0.538 |
| awgn | 5 | 0.925 | 0.944 | 0.637 | 0.919 | 0.916 | 0.968 | 0.983 |
| awgn | 10 | 0.990 | 0.992 | 0.947 | 0.982 | 0.979 | 1.000 | 1.000 |
| awgn | 15 | 0.994 | 0.996 | 0.975 | 0.986 | 0.984 | 0.999 | 1.000 |
| awgn | 20 | 0.995 | 0.996 | 0.976 | 0.987 | 0.984 | 1.000 | 1.000 |
| awgn | 25 | 0.995 | 0.996 | 0.979 | 0.987 | 0.985 | 1.000 | 1.000 |
| rayleigh | 0 | 0.240 | 0.405 | 0.001 | 0.256 | 0.253 | 0.110 | 0.214 |
| rayleigh | 5 | 0.538 | 0.673 | 0.090 | 0.571 | 0.569 | 0.527 | 0.676 |
| rayleigh | 10 | 0.832 | 0.880 | 0.397 | 0.839 | 0.836 | 0.890 | 0.946 |
| rayleigh | 15 | 0.945 | 0.960 | 0.727 | 0.937 | 0.935 | 0.980 | 0.995 |
| rayleigh | 20 | 0.973 | 0.980 | 0.860 | 0.966 | 0.964 | 0.991 | 0.997 |
| rayleigh | 25 | 0.980 | 0.984 | 0.891 | 0.971 | 0.968 | 0.993 | 0.999 |

**Caption text (new), test split (n = 1000)**; cos/margin/recall are the decoded signal's CLIP/CLAP embedding, mapped into S, against the original's S anchor (recall within a gallery of 1000).

| channel | SNR (dB) | token_acc | bleu1 | exact_match | cos_to_anchor | margin | recall@1 | recall@5 |
|---|---|---|---|---|---|---|---|---|
| awgn | 0 | 0.444 | 0.577 | 0.029 | 0.489 | 0.458 | 0.369 | 0.512 |
| awgn | 5 | 0.915 | 0.938 | 0.621 | 0.916 | 0.888 | 0.957 | 0.984 |
| awgn | 10 | 0.989 | 0.992 | 0.939 | 0.984 | 0.957 | 0.996 | 1.000 |
| awgn | 15 | 0.993 | 0.996 | 0.966 | 0.989 | 0.962 | 0.997 | 1.000 |
| awgn | 20 | 0.994 | 0.997 | 0.970 | 0.990 | 0.963 | 0.997 | 1.000 |
| awgn | 25 | 0.996 | 0.997 | 0.977 | 0.990 | 0.963 | 0.997 | 1.000 |
| rayleigh | 0 | 0.226 | 0.406 | 0.001 | 0.289 | 0.267 | 0.118 | 0.207 |
| rayleigh | 5 | 0.528 | 0.659 | 0.079 | 0.591 | 0.563 | 0.510 | 0.665 |
| rayleigh | 10 | 0.802 | 0.866 | 0.377 | 0.836 | 0.808 | 0.872 | 0.934 |
| rayleigh | 15 | 0.930 | 0.953 | 0.707 | 0.937 | 0.909 | 0.972 | 0.985 |
| rayleigh | 20 | 0.961 | 0.975 | 0.827 | 0.965 | 0.937 | 0.990 | 0.995 |
| rayleigh | 25 | 0.970 | 0.983 | 0.880 | 0.974 | 0.946 | 0.994 | 0.999 |

**ESC-50 event audio (new), val split (n = 400)**; cos/margin/recall are the decoded signal's CLIP/CLAP embedding, mapped into S, against the original's S anchor (recall within a gallery of 400).

| channel | SNR (dB) | si_sdr | lsd | cos_to_anchor | margin | recall@1 | recall@5 | recall@10 | clap_raw_cosine |
|---|---|---|---|---|---|---|---|---|---|
| awgn | 0 | 2.483 | 16.779 | 0.335 | 0.326 | 0.130 | 0.412 | 0.580 | 0.520 |
| awgn | 5 | 6.133 | 14.759 | 0.503 | 0.491 | 0.382 | 0.730 | 0.845 | 0.658 |
| awgn | 10 | 8.688 | 13.964 | 0.580 | 0.566 | 0.510 | 0.832 | 0.890 | 0.721 |
| awgn | 15 | 10.200 | 13.638 | 0.606 | 0.590 | 0.560 | 0.855 | 0.908 | 0.739 |
| awgn | 20 | 10.918 | 13.517 | 0.612 | 0.596 | 0.577 | 0.868 | 0.918 | 0.742 |
| awgn | 25 | 11.200 | 13.488 | 0.613 | 0.597 | 0.572 | 0.865 | 0.918 | 0.743 |
| rayleigh | 0 | -3.426 | 16.267 | 0.115 | 0.100 | 0.018 | 0.078 | 0.125 | 0.310 |
| rayleigh | 5 | 0.406 | 15.781 | 0.205 | 0.190 | 0.055 | 0.203 | 0.280 | 0.407 |
| rayleigh | 10 | 3.139 | 15.641 | 0.321 | 0.307 | 0.162 | 0.420 | 0.545 | 0.515 |
| rayleigh | 15 | 5.144 | 15.619 | 0.414 | 0.398 | 0.270 | 0.587 | 0.712 | 0.593 |
| rayleigh | 20 | 6.366 | 15.630 | 0.464 | 0.447 | 0.352 | 0.652 | 0.772 | 0.636 |
| rayleigh | 25 | 7.076 | 15.659 | 0.485 | 0.467 | 0.382 | 0.690 | 0.793 | 0.653 |

**ESC-50 event audio (new), test split (n = 400)**; cos/margin/recall are the decoded signal's CLIP/CLAP embedding, mapped into S, against the original's S anchor (recall within a gallery of 400).

| channel | SNR (dB) | si_sdr | lsd | cos_to_anchor | margin | recall@1 | recall@5 | recall@10 | clap_raw_cosine |
|---|---|---|---|---|---|---|---|---|---|
| awgn | 0 | 2.039 | 16.233 | 0.324 | 0.309 | 0.150 | 0.420 | 0.543 | 0.520 |
| awgn | 5 | 5.573 | 14.343 | 0.478 | 0.461 | 0.350 | 0.688 | 0.793 | 0.652 |
| awgn | 10 | 8.024 | 13.612 | 0.559 | 0.540 | 0.490 | 0.785 | 0.877 | 0.716 |
| awgn | 15 | 9.458 | 13.320 | 0.580 | 0.559 | 0.550 | 0.815 | 0.880 | 0.732 |
| awgn | 20 | 10.145 | 13.218 | 0.579 | 0.558 | 0.558 | 0.822 | 0.890 | 0.732 |
| awgn | 25 | 10.414 | 13.208 | 0.578 | 0.557 | 0.575 | 0.817 | 0.897 | 0.732 |
| rayleigh | 0 | -3.847 | 15.936 | 0.125 | 0.108 | 0.020 | 0.093 | 0.147 | 0.322 |
| rayleigh | 5 | -0.169 | 15.531 | 0.206 | 0.187 | 0.072 | 0.192 | 0.260 | 0.411 |
| rayleigh | 10 | 2.551 | 15.437 | 0.305 | 0.284 | 0.140 | 0.352 | 0.488 | 0.509 |
| rayleigh | 15 | 4.514 | 15.476 | 0.385 | 0.363 | 0.248 | 0.517 | 0.652 | 0.583 |
| rayleigh | 20 | 5.758 | 15.518 | 0.430 | 0.406 | 0.282 | 0.618 | 0.715 | 0.622 |
| rayleigh | 25 | 6.427 | 15.569 | 0.448 | 0.424 | 0.303 | 0.627 | 0.720 | 0.638 |

**Class-prompt exact recovery by the text transceiver** (fraction of prompts decoded exactly; averaged over noise draws; prompts were in its training data):

*awgn*

| prompt set | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB |
|---|---|---|---|---|---|---|
| cifar10_photo | 0.34 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| cifar100_photo | 0.26 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| mnist_digit | 0.32 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 |
| cifar10_sound | 0.10 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 |
| esc50 This is a sound of {}. | 0.08 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| esc50 {} | 0.26 | 0.91 | 1.00 | 1.00 | 1.00 | 1.00 |
| esc50 the sound of {} | 0.19 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 |

*rayleigh*

| prompt set | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB |
|---|---|---|---|---|---|---|
| cifar10_photo | 0.18 | 0.78 | 0.94 | 1.00 | 1.00 | 1.00 |
| cifar100_photo | 0.15 | 0.56 | 0.88 | 0.98 | 1.00 | 1.00 |
| mnist_digit | 0.30 | 0.62 | 0.90 | 0.98 | 1.00 | 1.00 |
| cifar10_sound | 0.04 | 0.58 | 0.82 | 0.96 | 1.00 | 1.00 |
| esc50 This is a sound of {}. | 0.07 | 0.44 | 0.84 | 0.95 | 0.99 | 0.99 |
| esc50 {} | 0.07 | 0.38 | 0.71 | 0.94 | 0.97 | 1.00 |
| esc50 the sound of {} | 0.09 | 0.44 | 0.84 | 0.96 | 0.98 | 1.00 |

## 3. Anchors in S and centering statistics

Image and text anchors are centered CLIP; audio anchors are bridged, centered CLAP (seed-42 linear bridge). Each domain's centering mean is fitted on its TRAIN anchors only and cached (`outputs/week5b/space/centerer_<task>.pt`, provenance in `manifest.json`); ESC-50 uses folds 1-3 (asserted), and the new tasks' anchor caches assert that fold 5 shares no file with train or validation.

| domain | centering fitted on | n | bridge | random-pair cosine, raw test | random-pair cosine, S test |
|---|---|---|---|---|---|
| CIFAR-10 image | train anchors | 45000 | - | 0.774 | 0.001 |
| MNIST image | train anchors | 54000 | - | 0.950 | 0.002 |
| Europarl text | train anchors | 80000 | - | 0.802 | 0.001 |
| Valentini speech | train anchors | 31245 | seed 42 | 0.801 | 0.141 |
| Caption text (new) | train captions (Stage 0 centerer) | 195538 | - | 0.604 | 0.033 |
| ESC-50 event audio (new) | ESC-50 folds 1-3 | 1200 | seed 42 | 0.217 | 0.012 |

## 4. Heads (1 seed, 3 variants, 6 tasks × 2 channels)

| task | channel | variant | target space | head params | best val loss | best epoch / epochs | train time |
|---|---|---|---|---|---|---|---|
| CIFAR-10 image | awgn | cos | raw | 1.57M | 2.746 | 21 / 40 | 3.4 min |
| CIFAR-10 image | awgn | cos5 | raw | 1.57M | 3.931 | 28 / 40 | 4.0 min |
| CIFAR-10 image | awgn | cen | centered | 1.57M | 2.258 | 17 / 40 | 3.7 min |
| CIFAR-10 image | rayleigh | cos | raw | 1.57M | 2.261 | 39 / 40 | 5.1 min |
| CIFAR-10 image | rayleigh | cos5 | raw | 1.57M | 3.671 | 39 / 40 | 3.5 min |
| CIFAR-10 image | rayleigh | cen | centered | 1.57M | 1.961 | 40 / 40 | 3.7 min |
| MNIST image | awgn | cos | raw | 0.39M | 1.206 | 40 / 40 | 2.7 min |
| MNIST image | awgn | cos5 | raw | 0.39M | 2.235 | 37 / 40 | 2.7 min |
| MNIST image | awgn | cen | centered | 0.39M | 0.581 | 40 / 40 | 3.2 min |
| MNIST image | rayleigh | cos | raw | 0.39M | 1.675 | 40 / 40 | 3.2 min |
| MNIST image | rayleigh | cos5 | raw | 0.39M | 2.692 | 39 / 40 | 3.0 min |
| MNIST image | rayleigh | cen | centered | 0.39M | 0.977 | 40 / 40 | 3.3 min |
| Europarl text | awgn | cos | raw | 0.68M | 1.523 | 20 / 20 | 6.6 min |
| Europarl text | awgn | cos5 | raw | 0.68M | 3.118 | 20 / 20 | 7.5 min |
| Europarl text | awgn | cen | centered | 0.68M | 1.205 | 20 / 20 | 7.8 min |
| Europarl text | rayleigh | cos | raw | 0.68M | 2.003 | 20 / 20 | 8.8 min |
| Europarl text | rayleigh | cos5 | raw | 0.68M | 3.689 | 20 / 20 | 5.6 min |
| Europarl text | rayleigh | cen | centered | 0.68M | 1.626 | 20 / 20 | 13.7 min |
| Valentini speech | awgn | cos | raw | 0.34M | 2.038 | 30 / 30 | 4.4 min |
| Valentini speech | awgn | cos5 | raw | 0.34M | 2.847 | 30 / 30 | 4.6 min |
| Valentini speech | awgn | cen | centered | 0.34M | 1.673 | 30 / 30 | 6.2 min |
| Valentini speech | rayleigh | cos | raw | 0.34M | 2.467 | 30 / 30 | 5.6 min |
| Valentini speech | rayleigh | cos5 | raw | 0.34M | 3.390 | 30 / 30 | 5.7 min |
| Valentini speech | rayleigh | cen | centered | 0.34M | 2.167 | 30 / 30 | 5.4 min |
| Caption text (new) | awgn | cos | raw | 0.68M | 0.351 | 20 / 20 | 18.3 min |
| Caption text (new) | awgn | cos5 | raw | 0.68M | 1.300 | 20 / 20 | 17.2 min |
| Caption text (new) | awgn | cen | centered | 0.68M | 0.418 | 20 / 20 | 26.8 min |
| Caption text (new) | rayleigh | cos | raw | 0.68M | 0.589 | 20 / 20 | 26.9 min |
| Caption text (new) | rayleigh | cos5 | raw | 0.68M | 1.985 | 20 / 20 | 9.4 min |
| Caption text (new) | rayleigh | cen | centered | 0.68M | 0.610 | 20 / 20 | 26.9 min |
| ESC-50 event audio (new) | awgn | cos | raw | 0.34M | 2.343 | 22 / 100 | 3.0 min |
| ESC-50 event audio (new) | awgn | cos5 | raw | 0.34M | 4.501 | 21 / 100 | 3.0 min |
| ESC-50 event audio (new) | awgn | cen | centered | 0.34M | 2.636 | 25 / 100 | 3.1 min |
| ESC-50 event audio (new) | rayleigh | cos | raw | 0.34M | 2.223 | 43 / 100 | 3.2 min |
| ESC-50 event audio (new) | rayleigh | cos5 | raw | 0.34M | 4.321 | 34 / 100 | 3.5 min |
| ESC-50 event audio (new) | rayleigh | cen | centered | 0.34M | 2.455 | 37 / 100 | 3.9 min |

A best epoch far below the total means the head overfit its training set and the checkpoint with the lowest validation loss was kept.

**Loss-curve check** (parsed from the training logs): relative improvement of the validation loss over the last 5 epochs. Because the learning rate follows a cosine decay, the best epoch is usually the last one; a large improvement over the last epochs would suggest the head was cut short, a value near zero that it had converged. Image tasks used the 40-epoch setting.

| task | channel | variant | epochs logged | final val loss | improvement over last 5 epochs |
|---|---|---|---|---|---|
| CIFAR-10 image | awgn | cos | 40 | 2.775 | -0.7% |
| CIFAR-10 image | awgn | cos5 | 40 | 3.956 | -0.3% |
| CIFAR-10 image | awgn | cen | 40 | 2.352 | -0.6% |
| CIFAR-10 image | rayleigh | cos | 40 | 2.262 | 0.9% |
| CIFAR-10 image | rayleigh | cos5 | 40 | 3.676 | 0.5% |
| CIFAR-10 image | rayleigh | cen | 40 | 1.961 | 0.9% |
| MNIST image | awgn | cos | 40 | 1.206 | 0.2% |
| MNIST image | awgn | cos5 | 40 | 2.238 | 0.2% |
| MNIST image | awgn | cen | 40 | 0.581 | 1.0% |
| MNIST image | rayleigh | cos | 40 | 1.675 | 0.5% |
| MNIST image | rayleigh | cos5 | 40 | 2.692 | 0.2% |
| MNIST image | rayleigh | cen | 40 | 0.977 | 1.0% |
| Europarl text | awgn | cos | 20 | 1.523 | 7.2% |
| Europarl text | awgn | cos5 | 20 | 3.118 | 4.2% |
| Europarl text | awgn | cen | 20 | 1.205 | 5.0% |
| Europarl text | rayleigh | cos | 20 | 2.003 | 6.6% |
| Europarl text | rayleigh | cos5 | 20 | 3.689 | 3.6% |
| Europarl text | rayleigh | cen | 20 | 1.626 | 5.0% |
| Valentini speech | awgn | cos | 30 | 2.038 | 2.4% |
| Valentini speech | awgn | cos5 | 30 | 2.847 | 1.7% |
| Valentini speech | awgn | cen | 30 | 1.673 | 1.8% |
| Valentini speech | rayleigh | cos | 30 | 2.467 | 2.1% |
| Valentini speech | rayleigh | cos5 | 30 | 3.390 | 1.2% |
| Valentini speech | rayleigh | cen | 30 | 2.167 | 2.0% |
| Caption text (new) | awgn | cos | 20 | 0.351 | 5.1% |
| Caption text (new) | awgn | cos5 | 20 | 1.300 | 3.2% |
| Caption text (new) | awgn | cen | 20 | 0.418 | 2.8% |
| Caption text (new) | rayleigh | cos | 20 | 0.589 | 5.3% |
| Caption text (new) | rayleigh | cos5 | 20 | 1.985 | 2.7% |
| Caption text (new) | rayleigh | cen | 20 | 0.610 | 4.1% |
| ESC-50 event audio (new) | awgn | cos | 100 | 2.642 | -0.5% |
| ESC-50 event audio (new) | awgn | cos5 | 100 | 4.791 | -0.3% |
| ESC-50 event audio (new) | awgn | cen | 100 | 2.971 | -0.5% |
| ESC-50 event audio (new) | rayleigh | cos | 100 | 2.284 | -0.9% |
| ESC-50 event audio (new) | rayleigh | cos5 | 100 | 4.382 | -0.4% |
| ESC-50 event audio (new) | rayleigh | cen | 100 | 2.524 | 0.1% |

Heads still improving by more than 2% over their last 5 epochs and with their best epoch at the end: Europarl text awgn cos, Europarl text awgn cos5, Europarl text awgn cen, Europarl text rayleigh cos, Europarl text rayleigh cos5, Europarl text rayleigh cen, Valentini speech awgn cos, Valentini speech rayleigh cos, Valentini speech rayleigh cen, Caption text (new) awgn cos, Caption text (new) awgn cos5, Caption text (new) awgn cen, Caption text (new) rayleigh cos, Caption text (new) rayleigh cos5, Caption text (new) rayleigh cen.


Variant semantics: `cos` and `cos5` keep the Week 5 recipe (InfoNCE + 0.5 / 5 × cosine against the RAW CLIP/CLAP anchors) and are scored after mapping their output into S; `cen` trains the whole objective (InfoNCE + 0.5 × cosine) toward the centered S anchor. For the four tasks that already had Week 5 heads, `cos` and `cos5` are those heads (same code, same seed 42, same epochs: 40 image / 20 Europarl text / 30 speech) and were **reused rather than retrained**; only `cen` was trained for them. All three variants were trained for the two new tasks (20 caption-text epochs, 100 ESC-50 epochs because it has only 1,200 training clips). The loss curves did not call for a change from the 40-epoch image setting.

## 5. Fidelity selection

Definitions (all in S, per item): `margin = cos(z, a_own) − mean cos(z, a_other)`; **F_abs = margin / (1 − b_aa)** (headline; b_aa is the mean unrelated anchor-anchor cosine, so the divisor is the anchor's own margin and there is no clean-channel rescale); **F_ret = margin / m_clean** (retained fraction of the noise-free margin, channel loss only); **C = m_clean / (1 − b_aa)** (clean ceiling per domain), so F_abs = F_ret × C exactly; baselines: centered cosine (cosine to the anchor in S) and raw cosine (cosine to the anchor in the space the head was trained in; identical to centered cosine for the `cen` variant). Statistics use 5 independent noise draws per item over the grid 0-25 dB on the validation split (test is shown separately). 'Strictly increasing' is on the point estimates of the mean per SNR; 'sig. steps' counts adjacent SNR steps whose paired bootstrap 95% lower bound is above 0. Pooled Spearman is over every (item, draw, SNR) with no threshold. **C partly reflects instrument weakness** (for example CLIP on MNIST digits, CLAP on speech), so comparisons between systems are valid within a domain only.

### Variant choice (validation, F_abs, single head seed)

Pre-registered rule: the variant with the most (task, channel) pairs whose mean F_abs is strictly increasing over the SNR grid; ties broken by the larger median dynamic range, then by the larger median matched-vs-mismatched AUC at 0 dB.

| variant | strictly increasing / pairs | median dynamic range | median AUC at 0 dB |
|---|---|---|---|
| cos (cos 0.5 (main, raw targets)) | 12 / 12 | 0.100 | 0.907 |
| cos5 (cos 5 (ablation, raw targets)) **← chosen** | 12 / 12 | 0.125 | 0.921 |
| cen (cos 0.5 toward the centered S anchor) | 12 / 12 | 0.075 | 0.967 |

The choice of `cos5` was made on one head seed, with one noise realisation set, so near-ties should not be read as a real ranking; Stage 2 repeats the chosen variant over 3 seeds and reports seed-to-seed spread separately.

### How fragile is the choice? (the same statistics under other criteria, both splits)

All three variants are strictly increasing on every pair, so the pre-registered rule is decided by its first tie-breaker, the dynamic range of F_abs. That range depends on the scale of F_abs (the clean ceiling C), not only on how well F separates conditions, so the table also shows scale-free criteria. 'sd/range' is the item-level standard deviation of F_abs at 10 dB divided by the dynamic range (lower means a cleaner signal); 'won' counts the (task, channel) pairs, out of 12, on which that variant is best.

**val split** (pairs won: range / AUC 0 dB / Spearman / sd-range)

| variant | strictly increasing | median C | median range | median AUC 0 dB | median AUC 10 dB | median pooled Spearman | median sd/range | pairs won |
|---|---|---|---|---|---|---|---|---|
| cos | 12/12 | 0.268 | 0.100 | 0.907 | 0.971 | 0.371 | 0.75 | 0 / 0 / 1 / 1 |
| cos5 | 12/12 | 0.478 | 0.125 | 0.921 | 0.968 | 0.293 | 0.97 | 12 / 0 / 2 / 2 |
| cen | 12/12 | 0.263 | 0.075 | 0.967 | 0.992 | 0.420 | 0.61 | 0 / 12 / 9 / 9 |

**test split** (pairs won: range / AUC 0 dB / Spearman / sd-range)

| variant | strictly increasing | median C | median range | median AUC 0 dB | median AUC 10 dB | median pooled Spearman | median sd/range | pairs won |
|---|---|---|---|---|---|---|---|---|
| cos | 12/12 | 0.265 | 0.096 | 0.906 | 0.977 | 0.352 | 0.79 | 0 / 0 / 1 / 1 |
| cos5 | 12/12 | 0.488 | 0.125 | 0.921 | 0.973 | 0.299 | 0.95 | 12 / 0 / 2 / 2 |
| cen | 12/12 | 0.254 | 0.073 | 0.966 | 0.994 | 0.406 | 0.63 | 0 / 12 / 9 / 9 |

**Reading:** the rule picks `cos5`, and the validation and test splits agree on that. But `cen` (training toward the centered anchor) is best on every scale-free criterion; `cos5` wins only the scale-dependent range, because its heads sit closer to the anchors in absolute terms (higher C). Both are real advantages: a larger usable F_abs range versus cleaner separation of conditions. The selection was made on one head seed and the gaps in the scale-free criteria are large and consistent across tasks, while the range gap is a scale effect, so I would carry **both `cos5` and `cen`** into Stage 2 unless you prefer one; heads train in minutes, so this costs about twice the Stage 2 head compute, not twice the wall time.

### The four Fidelity candidates compared (chosen variant, validation)

| candidate | strictly increasing | mean sig. steps | median range | median pooled Spearman | median AUC 0 dB | median AUC 10 dB | median SNR at 90% of clean |
|---|---|---|---|---|---|---|---|
| F_abs | 12/12 | 5.00/5 | 0.125 | 0.293 | 0.921 | 0.968 | 6.1 dB (12 interpolated, 0 at/below 0 dB or unreached) |
| F_ret | 12/12 | 5.00/5 | 0.259 | 0.293 | 0.921 | 0.968 | 6.1 dB (12 interpolated, 0 at/below 0 dB or unreached) |
| centered cosine | 12/12 | 5.00/5 | 0.131 | 0.311 | 0.921 | 0.969 | 6.0 dB (12 interpolated, 0 at/below 0 dB or unreached) |
| raw cosine | 6/12 | 2.92/5 | 0.011 | 0.083 | 0.910 | 0.961 | 0.0 dB (2 interpolated, 10 at/below 0 dB or unreached) |

F_abs and F_ret are monotone transforms of the same margin, so their AUC and Spearman coincide; they differ in scale (F_ret is 1 by construction at the clean value, F_abs keeps the ceiling C). The two cosine baselines differ from the margin in what they subtract: the centered cosine keeps the item's own similarity to unrelated anchors in the score.

### Clean ceiling C (= mean F_abs of the noise-free signal) per domain and variant, validation

| task | channel | cos | cos5 | cen |
|---|---|---|---|---|
| CIFAR-10 image | awgn | 0.266 | 0.390 | 0.267 |
| CIFAR-10 image | rayleigh | 0.229 | 0.365 | 0.225 |
| MNIST image | awgn | 0.406 | 0.676 | 0.402 |
| MNIST image | rayleigh | 0.385 | 0.666 | 0.326 |
| Europarl text | awgn | 0.269 | 0.414 | 0.256 |
| Europarl text | rayleigh | 0.252 | 0.396 | 0.247 |
| Valentini speech | awgn | 0.297 | 0.572 | 0.267 |
| Valentini speech | rayleigh | 0.254 | 0.543 | 0.230 |
| Caption text (new) | awgn | 0.371 | 0.634 | 0.322 |
| Caption text (new) | rayleigh | 0.295 | 0.572 | 0.260 |
| ESC-50 event audio (new) | awgn | 0.230 | 0.289 | 0.261 |
| ESC-50 event audio (new) | rayleigh | 0.241 | 0.305 | 0.264 |

C partly reflects instrument weakness (CLIP on MNIST digits, CLAP on speech and on 1,200-clip ESC-50 training data) and head quality, so it is a property of the (transceiver, head, anchor space) triple: comparisons are valid within a domain only.

**F_abs, variant `cos5`, val split**

| task | channel | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | clean | increasing | sig. steps | range | Spearman | SNR at 90% of clean | AUC 0 dB | AUC 10 dB | C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 image | awgn | 0.310 | 0.362 | 0.380 | 0.387 | 0.389 | 0.389 | 0.390 | yes | 5/5 | 0.079 | 0.158 | 3.9 dB | 0.932 | 0.963 | 0.390 |
| CIFAR-10 image | rayleigh | 0.268 | 0.319 | 0.345 | 0.357 | 0.362 | 0.364 | 0.365 | yes | 5/5 | 0.096 | 0.232 | 6.9 dB | 0.927 | 0.964 | 0.365 |
| MNIST image | awgn | 0.548 | 0.628 | 0.659 | 0.671 | 0.674 | 0.675 | 0.676 | yes | 5/5 | 0.127 | 0.355 | 3.7 dB | 0.990 | 0.995 | 0.676 |
| MNIST image | rayleigh | 0.479 | 0.571 | 0.623 | 0.648 | 0.659 | 0.663 | 0.666 | yes | 5/5 | 0.184 | 0.469 | 7.7 dB | 0.978 | 0.993 | 0.666 |
| Europarl text | awgn | 0.259 | 0.343 | 0.389 | 0.405 | 0.411 | 0.413 | 0.414 | yes | 5/5 | 0.154 | 0.355 | 8.2 dB | 0.916 | 0.981 | 0.414 |
| Europarl text | rayleigh | 0.241 | 0.310 | 0.357 | 0.380 | 0.390 | 0.394 | 0.396 | yes | 5/5 | 0.153 | 0.380 | 9.9 dB | 0.900 | 0.971 | 0.396 |
| Valentini speech | awgn | 0.476 | 0.544 | 0.565 | 0.569 | 0.571 | 0.572 | 0.572 | yes | 5/5 | 0.095 | 0.174 | 2.8 dB | 0.919 | 0.950 | 0.572 |
| Valentini speech | rayleigh | 0.419 | 0.497 | 0.528 | 0.537 | 0.541 | 0.542 | 0.543 | yes | 5/5 | 0.123 | 0.222 | 4.4 dB | 0.887 | 0.930 | 0.543 |
| Caption text (new) | awgn | 0.352 | 0.520 | 0.597 | 0.622 | 0.630 | 0.633 | 0.634 | yes | 5/5 | 0.280 | 0.519 | 8.3 dB | 0.971 | 0.998 | 0.634 |
| Caption text (new) | rayleigh | 0.268 | 0.400 | 0.495 | 0.541 | 0.560 | 0.567 | 0.572 | yes | 5/5 | 0.299 | 0.583 | 12.1 dB | 0.923 | 0.995 | 0.572 |
| ESC-50 event audio (new) | awgn | 0.238 | 0.275 | 0.285 | 0.288 | 0.289 | 0.289 | 0.289 | yes | 5/5 | 0.050 | 0.112 | 2.9 dB | 0.892 | 0.920 | 0.289 |
| ESC-50 event audio (new) | rayleigh | 0.227 | 0.274 | 0.295 | 0.302 | 0.304 | 0.305 | 0.305 | yes | 5/5 | 0.078 | 0.182 | 5.3 dB | 0.878 | 0.921 | 0.305 |

**F_ret, variant `cos5`, val split**

| task | channel | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | clean | increasing | sig. steps | range | Spearman | SNR at 90% of clean | AUC 0 dB | AUC 10 dB | C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 image | awgn | 0.797 | 0.929 | 0.976 | 0.992 | 0.998 | 0.999 | 1.000 | yes | 5/5 | 0.203 | 0.158 | 3.9 dB | 0.932 | 0.963 | 0.390 |
| CIFAR-10 image | rayleigh | 0.734 | 0.872 | 0.945 | 0.978 | 0.992 | 0.997 | 1.000 | yes | 5/5 | 0.263 | 0.232 | 6.9 dB | 0.927 | 0.964 | 0.365 |
| MNIST image | awgn | 0.811 | 0.930 | 0.975 | 0.993 | 0.997 | 0.999 | 1.000 | yes | 5/5 | 0.188 | 0.355 | 3.7 dB | 0.990 | 0.995 | 0.676 |
| MNIST image | rayleigh | 0.720 | 0.858 | 0.935 | 0.974 | 0.990 | 0.997 | 1.000 | yes | 5/5 | 0.276 | 0.469 | 7.7 dB | 0.978 | 0.993 | 0.666 |
| Europarl text | awgn | 0.625 | 0.829 | 0.940 | 0.979 | 0.994 | 0.998 | 1.000 | yes | 5/5 | 0.373 | 0.355 | 8.2 dB | 0.916 | 0.981 | 0.414 |
| Europarl text | rayleigh | 0.608 | 0.784 | 0.902 | 0.959 | 0.985 | 0.995 | 1.000 | yes | 5/5 | 0.387 | 0.380 | 9.9 dB | 0.900 | 0.971 | 0.396 |
| Valentini speech | awgn | 0.833 | 0.951 | 0.988 | 0.995 | 0.998 | 0.999 | 1.000 | yes | 5/5 | 0.167 | 0.174 | 2.8 dB | 0.919 | 0.950 | 0.572 |
| Valentini speech | rayleigh | 0.772 | 0.916 | 0.972 | 0.990 | 0.996 | 0.999 | 1.000 | yes | 5/5 | 0.227 | 0.222 | 4.4 dB | 0.887 | 0.930 | 0.543 |
| Caption text (new) | awgn | 0.556 | 0.820 | 0.942 | 0.982 | 0.994 | 0.998 | 1.000 | yes | 5/5 | 0.443 | 0.519 | 8.3 dB | 0.971 | 0.998 | 0.634 |
| Caption text (new) | rayleigh | 0.469 | 0.699 | 0.866 | 0.946 | 0.979 | 0.992 | 1.000 | yes | 5/5 | 0.523 | 0.583 | 12.1 dB | 0.923 | 0.995 | 0.572 |
| ESC-50 event audio (new) | awgn | 0.825 | 0.953 | 0.988 | 0.997 | 0.999 | 1.000 | 1.000 | yes | 5/5 | 0.175 | 0.112 | 2.9 dB | 0.892 | 0.920 | 0.289 |
| ESC-50 event audio (new) | rayleigh | 0.744 | 0.896 | 0.967 | 0.988 | 0.996 | 0.999 | 1.000 | yes | 5/5 | 0.255 | 0.182 | 5.3 dB | 0.878 | 0.921 | 0.305 |

**centered cosine, variant `cos5`, val split**

| task | channel | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | clean | increasing | sig. steps | range | Spearman | SNR at 90% of clean | AUC 0 dB | AUC 10 dB | C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 image | awgn | 0.307 | 0.359 | 0.378 | 0.384 | 0.386 | 0.387 | 0.387 | yes | 5/5 | 0.080 | 0.161 | 4.0 dB | 0.933 | 0.963 | - |
| CIFAR-10 image | rayleigh | 0.263 | 0.315 | 0.342 | 0.354 | 0.359 | 0.361 | 0.363 | yes | 5/5 | 0.098 | 0.239 | 7.1 dB | 0.928 | 0.965 | - |
| MNIST image | awgn | 0.546 | 0.626 | 0.657 | 0.669 | 0.672 | 0.673 | 0.673 | yes | 5/5 | 0.126 | 0.359 | 3.7 dB | 0.990 | 0.995 | - |
| MNIST image | rayleigh | 0.477 | 0.569 | 0.620 | 0.646 | 0.656 | 0.661 | 0.663 | yes | 5/5 | 0.184 | 0.474 | 7.7 dB | 0.979 | 0.993 | - |
| Europarl text | awgn | 0.247 | 0.332 | 0.378 | 0.394 | 0.400 | 0.402 | 0.403 | yes | 5/5 | 0.155 | 0.360 | 8.3 dB | 0.917 | 0.982 | - |
| Europarl text | rayleigh | 0.228 | 0.298 | 0.346 | 0.369 | 0.379 | 0.383 | 0.385 | yes | 5/5 | 0.155 | 0.390 | 10.2 dB | 0.902 | 0.973 | - |
| Valentini speech | awgn | 0.493 | 0.566 | 0.588 | 0.594 | 0.596 | 0.596 | 0.597 | yes | 5/5 | 0.103 | 0.207 | 3.0 dB | 0.909 | 0.945 | - |
| Valentini speech | rayleigh | 0.419 | 0.500 | 0.535 | 0.547 | 0.552 | 0.554 | 0.555 | yes | 5/5 | 0.135 | 0.263 | 5.0 dB | 0.878 | 0.923 | - |
| Caption text (new) | awgn | 0.345 | 0.513 | 0.591 | 0.616 | 0.624 | 0.627 | 0.628 | yes | 5/5 | 0.282 | 0.530 | 8.3 dB | 0.971 | 0.998 | - |
| Caption text (new) | rayleigh | 0.258 | 0.391 | 0.487 | 0.533 | 0.552 | 0.560 | 0.564 | yes | 5/5 | 0.302 | 0.594 | 12.2 dB | 0.924 | 0.995 | - |
| ESC-50 event audio (new) | awgn | 0.258 | 0.294 | 0.304 | 0.306 | 0.307 | 0.307 | 0.307 | yes | 5/5 | 0.049 | 0.112 | 2.5 dB | 0.894 | 0.921 | - |
| ESC-50 event audio (new) | rayleigh | 0.241 | 0.289 | 0.310 | 0.317 | 0.319 | 0.320 | 0.320 | yes | 5/5 | 0.079 | 0.187 | 4.9 dB | 0.879 | 0.922 | - |

**raw cosine, variant `cos5`, val split**

| task | channel | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | clean | increasing | sig. steps | range | Spearman | SNR at 90% of clean | AUC 0 dB | AUC 10 dB | C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 image | awgn | 0.811 | 0.819 | 0.821 | 0.821 | 0.821 | 0.821 | 0.821 | yes | 4/5 | 0.011 | 0.085 | ≤ 0 dB | 0.922 | 0.958 | - |
| CIFAR-10 image | rayleigh | 0.802 | 0.809 | 0.811 | 0.811 | 0.811 | 0.810 | 0.810 | no | 2/5 | 0.008 | 0.082 | ≤ 0 dB | 0.923 | 0.964 | - |
| MNIST image | awgn | 0.901 | 0.904 | 0.904 | 0.905 | 0.905 | 0.905 | 0.905 | yes | 3/5 | 0.003 | 0.071 | ≤ 0 dB | 0.985 | 0.996 | - |
| MNIST image | rayleigh | 0.906 | 0.910 | 0.911 | 0.912 | 0.912 | 0.912 | 0.912 | yes | 4/5 | 0.006 | 0.132 | ≤ 0 dB | 0.970 | 0.995 | - |
| Europarl text | awgn | 0.788 | 0.797 | 0.801 | 0.801 | 0.802 | 0.802 | 0.802 | yes | 4/5 | 0.014 | 0.130 | ≤ 0 dB | 0.891 | 0.965 | - |
| Europarl text | rayleigh | 0.795 | 0.803 | 0.806 | 0.807 | 0.807 | 0.806 | 0.806 | no | 3/5 | 0.011 | 0.111 | ≤ 0 dB | 0.878 | 0.954 | - |
| Valentini speech | awgn | 0.909 | 0.915 | 0.914 | 0.913 | 0.912 | 0.912 | 0.912 | no | 1/5 | 0.003 | 0.000 | ≤ 0 dB | 0.915 | 0.958 | - |
| Valentini speech | rayleigh | 0.899 | 0.907 | 0.907 | 0.906 | 0.905 | 0.905 | 0.905 | no | 1/5 | 0.006 | 0.028 | ≤ 0 dB | 0.880 | 0.937 | - |
| Caption text (new) | awgn | 0.728 | 0.791 | 0.818 | 0.827 | 0.829 | 0.830 | 0.830 | yes | 5/5 | 0.102 | 0.487 | 1.6 dB | 0.924 | 0.994 | - |
| Caption text (new) | rayleigh | 0.691 | 0.738 | 0.770 | 0.785 | 0.790 | 0.792 | 0.793 | yes | 5/5 | 0.101 | 0.525 | 2.5 dB | 0.884 | 0.989 | - |
| ESC-50 event audio (new) | awgn | 0.538 | 0.552 | 0.552 | 0.550 | 0.549 | 0.549 | 0.549 | no | 1/5 | 0.011 | 0.032 | ≤ 0 dB | 0.905 | 0.928 | - |
| ESC-50 event audio (new) | rayleigh | 0.550 | 0.566 | 0.569 | 0.567 | 0.565 | 0.564 | 0.564 | no | 2/5 | 0.015 | 0.047 | ≤ 0 dB | 0.891 | 0.930 | - |

**F_abs, variant `cos5`, test split**

| task | channel | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | clean | increasing | sig. steps | range | Spearman | SNR at 90% of clean | AUC 0 dB | AUC 10 dB | C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 image | awgn | 0.307 | 0.360 | 0.379 | 0.386 | 0.388 | 0.388 | 0.389 | yes | 5/5 | 0.081 | 0.166 | 4.0 dB | 0.937 | 0.966 | 0.389 |
| CIFAR-10 image | rayleigh | 0.266 | 0.319 | 0.346 | 0.359 | 0.364 | 0.366 | 0.367 | yes | 5/5 | 0.100 | 0.246 | 7.1 dB | 0.936 | 0.972 | 0.367 |
| MNIST image | awgn | 0.542 | 0.624 | 0.655 | 0.665 | 0.668 | 0.669 | 0.670 | yes | 5/5 | 0.128 | 0.352 | 3.7 dB | 0.987 | 0.994 | 0.670 |
| MNIST image | rayleigh | 0.472 | 0.565 | 0.616 | 0.641 | 0.652 | 0.656 | 0.659 | yes | 5/5 | 0.184 | 0.469 | 7.7 dB | 0.975 | 0.990 | 0.659 |
| Europarl text | awgn | 0.255 | 0.340 | 0.386 | 0.402 | 0.407 | 0.408 | 0.409 | yes | 5/5 | 0.153 | 0.356 | 8.1 dB | 0.914 | 0.983 | 0.409 |
| Europarl text | rayleigh | 0.243 | 0.310 | 0.357 | 0.378 | 0.388 | 0.392 | 0.394 | yes | 5/5 | 0.149 | 0.376 | 9.8 dB | 0.903 | 0.974 | 0.394 |
| Valentini speech | awgn | 0.498 | 0.567 | 0.586 | 0.591 | 0.592 | 0.593 | 0.593 | yes | 5/5 | 0.095 | 0.165 | 2.6 dB | 0.861 | 0.893 | 0.593 |
| Valentini speech | rayleigh | 0.451 | 0.529 | 0.558 | 0.569 | 0.572 | 0.573 | 0.574 | yes | 5/5 | 0.122 | 0.211 | 4.2 dB | 0.820 | 0.863 | 0.574 |
| Caption text (new) | awgn | 0.362 | 0.519 | 0.593 | 0.617 | 0.625 | 0.627 | 0.628 | yes | 5/5 | 0.266 | 0.469 | 8.1 dB | 0.971 | 0.998 | 0.628 |
| Caption text (new) | rayleigh | 0.279 | 0.410 | 0.497 | 0.539 | 0.557 | 0.563 | 0.568 | yes | 5/5 | 0.285 | 0.533 | 11.7 dB | 0.927 | 0.994 | 0.568 |
| ESC-50 event audio (new) | awgn | 0.225 | 0.256 | 0.265 | 0.266 | 0.267 | 0.267 | 0.267 | yes | 4/5 | 0.042 | 0.097 | 2.4 dB | 0.875 | 0.901 | 0.267 |
| ESC-50 event audio (new) | rayleigh | 0.206 | 0.247 | 0.266 | 0.271 | 0.273 | 0.274 | 0.274 | yes | 5/5 | 0.068 | 0.157 | 4.9 dB | 0.852 | 0.897 | 0.274 |

**F_ret, variant `cos5`, test split**

| task | channel | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | clean | increasing | sig. steps | range | Spearman | SNR at 90% of clean | AUC 0 dB | AUC 10 dB | C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 image | awgn | 0.790 | 0.927 | 0.975 | 0.993 | 0.997 | 0.999 | 1.000 | yes | 5/5 | 0.209 | 0.166 | 4.0 dB | 0.937 | 0.966 | 0.389 |
| CIFAR-10 image | rayleigh | 0.724 | 0.868 | 0.943 | 0.978 | 0.991 | 0.997 | 1.000 | yes | 5/5 | 0.273 | 0.246 | 7.1 dB | 0.936 | 0.972 | 0.367 |
| MNIST image | awgn | 0.809 | 0.932 | 0.978 | 0.992 | 0.998 | 0.999 | 1.000 | yes | 5/5 | 0.191 | 0.352 | 3.7 dB | 0.987 | 0.994 | 0.670 |
| MNIST image | rayleigh | 0.716 | 0.857 | 0.935 | 0.973 | 0.990 | 0.996 | 1.000 | yes | 5/5 | 0.280 | 0.469 | 7.7 dB | 0.975 | 0.990 | 0.659 |
| Europarl text | awgn | 0.624 | 0.831 | 0.942 | 0.981 | 0.994 | 0.998 | 1.000 | yes | 5/5 | 0.374 | 0.356 | 8.1 dB | 0.914 | 0.983 | 0.409 |
| Europarl text | rayleigh | 0.616 | 0.788 | 0.905 | 0.959 | 0.984 | 0.994 | 1.000 | yes | 5/5 | 0.378 | 0.376 | 9.8 dB | 0.903 | 0.974 | 0.394 |
| Valentini speech | awgn | 0.840 | 0.957 | 0.989 | 0.997 | 0.999 | 1.000 | 1.000 | yes | 5/5 | 0.160 | 0.165 | 2.6 dB | 0.861 | 0.893 | 0.593 |
| Valentini speech | rayleigh | 0.786 | 0.922 | 0.972 | 0.991 | 0.997 | 0.999 | 1.000 | yes | 5/5 | 0.213 | 0.211 | 4.2 dB | 0.820 | 0.863 | 0.574 |
| Caption text (new) | awgn | 0.576 | 0.826 | 0.944 | 0.982 | 0.995 | 0.998 | 1.000 | yes | 5/5 | 0.423 | 0.469 | 8.1 dB | 0.971 | 0.998 | 0.628 |
| Caption text (new) | rayleigh | 0.491 | 0.722 | 0.875 | 0.950 | 0.981 | 0.992 | 1.000 | yes | 5/5 | 0.501 | 0.533 | 11.7 dB | 0.927 | 0.994 | 0.568 |
| ESC-50 event audio (new) | awgn | 0.842 | 0.960 | 0.991 | 0.998 | 0.999 | 1.000 | 1.000 | yes | 4/5 | 0.158 | 0.097 | 2.4 dB | 0.875 | 0.901 | 0.267 |
| ESC-50 event audio (new) | rayleigh | 0.752 | 0.902 | 0.970 | 0.990 | 0.997 | 0.999 | 1.000 | yes | 5/5 | 0.247 | 0.157 | 4.9 dB | 0.852 | 0.897 | 0.274 |

**centered cosine, variant `cos5`, test split**

| task | channel | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | clean | increasing | sig. steps | range | Spearman | SNR at 90% of clean | AUC 0 dB | AUC 10 dB | C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 image | awgn | 0.304 | 0.357 | 0.377 | 0.384 | 0.385 | 0.386 | 0.386 | yes | 5/5 | 0.082 | 0.170 | 4.1 dB | 0.938 | 0.966 | - |
| CIFAR-10 image | rayleigh | 0.262 | 0.315 | 0.343 | 0.356 | 0.361 | 0.363 | 0.364 | yes | 5/5 | 0.101 | 0.253 | 7.2 dB | 0.937 | 0.973 | - |
| MNIST image | awgn | 0.542 | 0.627 | 0.658 | 0.668 | 0.672 | 0.673 | 0.673 | yes | 5/5 | 0.130 | 0.362 | 3.8 dB | 0.987 | 0.994 | - |
| MNIST image | rayleigh | 0.472 | 0.566 | 0.619 | 0.644 | 0.655 | 0.659 | 0.662 | yes | 5/5 | 0.188 | 0.476 | 7.8 dB | 0.974 | 0.990 | - |
| Europarl text | awgn | 0.247 | 0.332 | 0.378 | 0.394 | 0.399 | 0.401 | 0.402 | yes | 5/5 | 0.154 | 0.362 | 8.2 dB | 0.915 | 0.984 | - |
| Europarl text | rayleigh | 0.234 | 0.302 | 0.349 | 0.370 | 0.380 | 0.384 | 0.386 | yes | 5/5 | 0.150 | 0.384 | 9.9 dB | 0.905 | 0.975 | - |
| Valentini speech | awgn | 0.500 | 0.571 | 0.591 | 0.597 | 0.598 | 0.599 | 0.599 | yes | 5/5 | 0.098 | 0.196 | 2.7 dB | 0.842 | 0.879 | - |
| Valentini speech | rayleigh | 0.432 | 0.507 | 0.538 | 0.551 | 0.556 | 0.558 | 0.559 | yes | 5/5 | 0.126 | 0.246 | 4.8 dB | 0.803 | 0.846 | - |
| Caption text (new) | awgn | 0.346 | 0.507 | 0.582 | 0.606 | 0.614 | 0.616 | 0.617 | yes | 5/5 | 0.270 | 0.522 | 8.2 dB | 0.970 | 0.998 | - |
| Caption text (new) | rayleigh | 0.253 | 0.390 | 0.480 | 0.523 | 0.541 | 0.548 | 0.552 | yes | 5/5 | 0.295 | 0.593 | 12.0 dB | 0.927 | 0.994 | - |
| ESC-50 event audio (new) | awgn | 0.247 | 0.277 | 0.284 | 0.285 | 0.285 | 0.285 | 0.285 | yes | 3/5 | 0.039 | 0.092 | 1.7 dB | 0.874 | 0.901 | - |
| ESC-50 event audio (new) | rayleigh | 0.221 | 0.263 | 0.282 | 0.287 | 0.289 | 0.289 | 0.289 | yes | 4/5 | 0.068 | 0.161 | 4.6 dB | 0.853 | 0.898 | - |

**raw cosine, variant `cos5`, test split**

| task | channel | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | clean | increasing | sig. steps | range | Spearman | SNR at 90% of clean | AUC 0 dB | AUC 10 dB | C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 image | awgn | 0.811 | 0.820 | 0.821 | 0.822 | 0.822 | 0.822 | 0.822 | no | 4/5 | 0.010 | 0.086 | ≤ 0 dB | 0.928 | 0.963 | - |
| CIFAR-10 image | rayleigh | 0.803 | 0.810 | 0.812 | 0.812 | 0.812 | 0.812 | 0.811 | no | 2/5 | 0.009 | 0.081 | ≤ 0 dB | 0.928 | 0.972 | - |
| MNIST image | awgn | 0.901 | 0.904 | 0.905 | 0.905 | 0.905 | 0.905 | 0.905 | no | 3/5 | 0.004 | 0.082 | ≤ 0 dB | 0.980 | 0.992 | - |
| MNIST image | rayleigh | 0.906 | 0.910 | 0.912 | 0.912 | 0.912 | 0.913 | 0.913 | yes | 5/5 | 0.006 | 0.150 | ≤ 0 dB | 0.965 | 0.990 | - |
| Europarl text | awgn | 0.788 | 0.797 | 0.801 | 0.802 | 0.802 | 0.802 | 0.802 | yes | 3/5 | 0.014 | 0.132 | ≤ 0 dB | 0.899 | 0.975 | - |
| Europarl text | rayleigh | 0.796 | 0.803 | 0.807 | 0.807 | 0.807 | 0.807 | 0.807 | no | 2/5 | 0.011 | 0.115 | ≤ 0 dB | 0.887 | 0.960 | - |
| Valentini speech | awgn | 0.912 | 0.917 | 0.915 | 0.913 | 0.912 | 0.912 | 0.912 | no | 1/5 | 0.000 | -0.055 | ≤ 0 dB | 0.868 | 0.918 | - |
| Valentini speech | rayleigh | 0.903 | 0.909 | 0.908 | 0.906 | 0.905 | 0.905 | 0.904 | no | 1/5 | 0.001 | -0.042 | ≤ 0 dB | 0.830 | 0.892 | - |
| Caption text (new) | awgn | 0.741 | 0.798 | 0.823 | 0.830 | 0.833 | 0.833 | 0.834 | yes | 5/5 | 0.092 | 0.471 | 0.8 dB | 0.910 | 0.990 | - |
| Caption text (new) | rayleigh | 0.703 | 0.748 | 0.775 | 0.788 | 0.792 | 0.794 | 0.795 | yes | 5/5 | 0.091 | 0.513 | 1.4 dB | 0.878 | 0.984 | - |
| ESC-50 event audio (new) | awgn | 0.533 | 0.541 | 0.538 | 0.535 | 0.534 | 0.533 | 0.533 | no | 1/5 | 0.000 | -0.003 | ≤ 0 dB | 0.874 | 0.903 | - |
| ESC-50 event audio (new) | rayleigh | 0.541 | 0.554 | 0.553 | 0.550 | 0.547 | 0.546 | 0.546 | no | 1/5 | 0.005 | 0.014 | ≤ 0 dB | 0.863 | 0.906 | - |

**All three variants side by side: F_abs on validation** (increasing / dynamic range / AUC at 0 dB)

| task | channel | cos | cos5 | cen |
|---|---|---|---|---|
| CIFAR-10 image | awgn | inc / 0.053 / 0.940 | inc / 0.079 / 0.932 | inc / 0.049 / 0.967 |
| CIFAR-10 image | rayleigh | inc / 0.061 / 0.936 | inc / 0.096 / 0.927 | inc / 0.058 / 0.969 |
| MNIST image | awgn | inc / 0.087 / 0.995 | inc / 0.127 / 0.990 | inc / 0.083 / 0.997 |
| MNIST image | rayleigh | inc / 0.124 / 0.984 | inc / 0.184 / 0.978 | inc / 0.101 / 0.990 |
| Europarl text | awgn | inc / 0.111 / 0.915 | inc / 0.154 / 0.916 | inc / 0.107 / 0.964 |
| Europarl text | rayleigh | inc / 0.105 / 0.900 | inc / 0.153 / 0.900 | inc / 0.105 / 0.953 |
| Valentini speech | awgn | inc / 0.094 / 0.888 | inc / 0.095 / 0.919 | inc / 0.053 / 0.966 |
| Valentini speech | rayleigh | inc / 0.105 / 0.833 | inc / 0.123 / 0.887 | inc / 0.067 / 0.948 |
| Caption text (new) | awgn | inc / 0.155 / 0.924 | inc / 0.280 / 0.971 | inc / 0.127 / 0.995 |
| Caption text (new) | rayleigh | inc / 0.145 / 0.839 | inc / 0.299 / 0.923 | inc / 0.119 / 0.977 |
| ESC-50 event audio (new) | awgn | inc / 0.032 / 0.880 | inc / 0.050 / 0.892 | inc / 0.033 / 0.916 |
| ESC-50 event audio (new) | rayleigh | inc / 0.050 / 0.871 | inc / 0.078 / 0.878 | inc / 0.051 / 0.923 |

## 6. Caveats and deviations from the approved plan

- **Single seed everywhere.** Heads are single-seed (variant chosen on one), and both new transceivers are single-seed, so intervals exclude transceiver-training variance.
- **Deviation (prompts in the text transceiver's training set)** as described in section 1; held-out caption splits were not used. The prompt-recovery numbers are therefore optimistic for unseen text.
- **Deviation (Week 5 heads reused)** for `cos` and `cos5` on the four older tasks (section 4); `cen` is new for all six tasks.
- **Interpretation of the third variant.** I read 'cosine toward the centered anchor' as training the whole objective in S (InfoNCE and cosine both against the centered S anchor), keeping the two Week 5 variants against raw anchors so they remain comparable with Week 5. A cosine-weight-5 variant toward the centered anchor (the missing cell of a 2×2) was not trained.
- **Selection and test.** F statistics were computed on both splits, but the choice used validation only. For ESC-50 the validation split is fold 4 (used also to select head and transceiver checkpoints) and test is fold 5, which no training, centering statistic or selection touched.
- **Speech audio stays in the comparison with a weak instrument:** its S anchors have a residual random-pair cosine (table in section 3), so its F_abs scale is not comparable to the others.
- **ESC-50 heads see only 1,200 training clips** and overfit (best epochs in section 4); event audio's clean ceiling C is the lowest of the six domains (section 5). Its F_abs saturates early in SNR, so it separates low SNRs but not high ones.
- **LSD** barely improves above 10 dB on ESC-50 and stays high; it is dominated by low-energy spectral bins, and I did not investigate its composition further, so treat it as a relative measure only (silent reference frames are excluded).
- **Not started:** Stage 2 (3 seeds for the chosen variant, cross-modal channel experiments, ImageBind ablation).
