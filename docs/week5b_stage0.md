# Week 5b, Stage 0: clean cross-modal gate for the shared space S

**Overall gate (decision seed 42): PASS.** Generated from `outputs/week5b/stage0/results.json`; nothing here involves a channel. Per-seed pass flags (G1, G2): seed 42: True/True, seed 43: True/True, seed 44: True/True.

## 1. Verdict against the pre-registered criteria

| gate | criterion | result (decision seed) | pass | mean ± sd over seeds [42, 43, 44] |
|---|---|---|---|---|
| G1 audio→text, ESC-50 test fold (400 clips, 50 classes, chance 0.02) | top-1 ≥ 0.3 and ≥ 0.6 × native CLAP | S top-1 0.702 [0.658, 0.745]; native 0.887; ratio 0.79 | yes | S top-1 0.711 ± 0.007 |
| G2 audio→image, balanced top-1 over 5 shared classes (320 clips, chance 0.2) | point ≥ 2.0 × chance and lower 95% bound > chance | 0.982 [0.966, 0.995] | yes | point 0.980 ± 0.003; lower bounds 0.961 ± 0.005 |

The native figure uses the same 400 clips, the same 50 class names and the same template (`This is a sound of {}.`) as the S-space test. The template was fixed before any audio number was computed; the secondary prompt variants below show how much it matters. G1 is judged on the test fold (the only audio-fitted quantity, the audio mean, is fit on folds 1-3); G2 uses all five folds because the test fold holds only 64 mapped clips. The test-fold-only G2 number is reported too and, being perfect, has a degenerate bootstrap interval, so it is not used for the decision.

## 2. How S is built

S is the per-domain mean-centered CLIP space (512-d, unit norm). Image and text are native to it. Audio enters through CLAP, per-modality mean-centering (ESC-50 train folds 1-3), and a linear CLAP→CLIP bridge trained only on caption text (centered CLAP-text embedding → centered CLIP-text embedding; symmetric InfoNCE + 0.5·cosine, learnable temperature). Statistics fit on train data only: audio mean on 1200 clips, text means on 195538 train captions.

## 3. Text-to-text agreement of the bridge (held-out captions)

| seed | bridge | params | val R@1 | test cos (centered) | test R@1 | test R@5 | R@1 COCO | R@1 AudioCaps |
|---|---|---|---|---|---|---|---|---|
| 42 | linear (chosen) | 263k | 0.896 | 0.483 | 0.892 | 0.993 | 0.892 | 0.661 |
| 42 | mlp | 1050k | 0.905 | 0.490 | 0.901 | 0.994 | 0.901 | 0.688 |
| 43 | linear (chosen) | 263k | 0.897 | 0.482 | 0.893 | 0.993 | 0.893 | 0.658 |
| 43 | mlp | 1050k | 0.907 | 0.488 | 0.902 | 0.995 | 0.902 | 0.686 |
| 44 | linear (chosen) | 263k | 0.899 | 0.482 | 0.895 | 0.993 | 0.895 | 0.665 |
| 44 | mlp | 1050k | 0.909 | 0.487 | 0.904 | 0.994 | 0.904 | 0.691 |
| - | ridge regression (reference) | - | - | 0.682 | 0.560 | 0.866 | - | - |

Captions train/val/test = 195538/7166/9357 (COCO sub-sampled, all AudioCaps kept, captions duplicated across splits removed; split by source item). Retrieval is within disjoint galleries of 1000 captions (chance 0.001). The linear-vs-MLP choice follows the pre-registered rule (MLP only if it beats linear's validation R@1 by 0.05) and was made on validation captions only. Centered cosine is moderate because training optimises ranking (InfoNCE) with a 0.5-weight cosine term.

## 4. ESC-50 audio→text: ladder and failure breakdown

| method (test fold, decision seed) | top-1 [95% CI] | top-5 [95% CI] |
|---|---|---|
| L0 native CLAP (16 kHz source) | 0.887 [0.855, 0.918] | 0.983 [0.967, 0.995] |
| L0b native CLAP (original 44.1 kHz audio) | 0.905 [0.875, 0.930] | 0.988 [0.975, 0.998] |
| L0c native CLAP + per-modality centering | 0.895 [0.865, 0.923] | 0.983 [0.967, 0.995] |
| L1 audio->bridge vs CLIP-text prompts (the gate) | 0.702 [0.658, 0.745] | 0.945 [0.920, 0.965] |
| L2 audio->bridge vs CLAP-text prompts->bridge | 0.853 [0.817, 0.887] | 0.957 [0.938, 0.975] |
| L3 audio centered with the TEXT mean (ablation) | 0.685 [0.637, 0.730] | 0.925 [0.897, 0.950] |
| L4 audio not centered (ablation) | 0.683 [0.637, 0.728] | 0.935 [0.910, 0.957] |

Native zero-shot on all 2000 clips: 0.883 [0.868, 0.897]. L1 on all 2000 clips: 0.680 [0.659, 0.700]. L2 over seeds: 0.852 ± 0.004.

Reading the ladder: L0 is native CLAP. L1 is the gate. L2 sends the CLAP-text prompts through the same bridge, so the audio side and the prompt side are treated identically; the gap L0→L2 is the cost of the audio→bridge path, and the gap L2→L1 is the cost of using CLIP's own text encoder for the prompts. L3/L4 ablate the per-modality centering of audio.

Prompt sensitivity (top-1, test fold), native → S: `{}`: 0.863 → 0.652; `the sound of {}`: 0.832 → 0.757; `This is a sound of {}.`: 0.887 → 0.702.

Failure breakdown (decision seed): 11 of 49 classes with non-zero native accuracy fall below 60% of their native per-class accuracy; 4 classes are never predicted; the most-predicted class takes 4.5% of predictions (no collapse onto a few prompts).

| class | n | native acc | S acc |
|---|---|---|---|
| door wood creaks | 8 | 1.00 | 0.00 |
| pouring water | 8 | 1.00 | 0.12 |
| breathing | 8 | 0.88 | 0.00 |
| mouse click | 8 | 0.88 | 0.00 |
| coughing | 8 | 1.00 | 0.25 |
| chainsaw | 8 | 1.00 | 0.25 |
| clock tick | 8 | 1.00 | 0.38 |
| engine | 8 | 0.75 | 0.25 |
| cow | 8 | 1.00 | 0.62 |
| water drops | 8 | 1.00 | 0.62 |

Most frequent confusions in S: door wood creaks → door wood knock (8); hen → rooster (8); mouse click → keyboard typing (7); coughing → sneezing (6); pouring water → drinking sipping (5); breathing → snoring (5); clock tick → clock alarm (4); chainsaw → hand saw (3).

## 5. Audio→image (class level) and controls

ESC-50 classes mapped to CIFAR-10 before any result was computed: `{"airplane": "airplane", "cat": "cat", "dog": "dog", "frog": "frog", "chirping_birds": "bird", "crow": "bird", "hen": "bird", "rooster": "bird"}`. Balanced top-1 is used because the 'bird' class holds 50% of the clips, so a constant 'bird' predictor already scores that in plain accuracy. Audio queries are classified by cosine to per-class CIFAR-10 test-image prototypes in S.

| setting | balanced top-1 / top-1 [95% CI] |
|---|---|
| audio(S) → image prototypes, all folds (the gate) | 0.982 [0.966, 0.995] |
| audio(S) → image prototypes, test fold only | 1.000 [1.000, 1.000] (n=64) |
| audio(S)->CLIP-text 'a photo of a {c}' | 0.980 [0.970, 0.989] |
| native CLAP audio->CLAP-text sound prompts | 0.918 [0.899, 0.935] |
| image(S)->CLIP-text 'a photo of a {c}' | 0.899 [0.891, 0.907] |
| image(S)->audio prototypes (reverse direction) | 0.838 [0.828, 0.849] |
| control: untrained bridge, audio→image | 0.090 [0.061, 0.119] |
| control: untrained bridge, audio→text 50-way (G1 setting) | 0.027 [0.013, 0.045] |
| control: permuted labels, audio→image (mean / max of 200) | 0.201 / 0.270 |
| control: permuted labels, audio→text (mean / max of 200) | 0.020 / 0.038 |

The reference rows are not like-for-like: the S route is scored against CLIP-text *photo* prompts ('a photo of a {c}'), the native row against CLAP-text *sound* prompts, and the two prompt styles separate these five classes differently, so a higher S number than native here says nothing about the bridge being better than CLAP. An untrained bridge can fall below chance on a balanced metric because a random map is systematically, not randomly, wrong; the point of that control is that it is not above chance.


Chance: G1 0.02, G2 0.20. Prediction histogram over the 5 classes (airplane, bird, cat, dog, frog), decision seed: [40, 154, 39, 48, 39].

## 6. Clean image↔text checks in S (no audio, no channel)

| dataset | space | classes | image→text top-1 [CI] | top-5 [CI] | text→image P@1 | text→image R-precision |
|---|---|---|---|---|---|---|
| CIFAR-10 | raw CLIP | 10 | 0.883 [0.877, 0.890] | 0.993 [0.991, 0.994] | 1.000 | 0.818 |
| CIFAR-10 | S (centered) | 10 | 0.900 [0.894, 0.906] | 0.995 [0.994, 0.997] | 1.000 | 0.821 |
| CIFAR-100 | raw CLIP | 100 | 0.632 [0.622, 0.641] | 0.882 [0.876, 0.888] | 0.900 | 0.540 |
| CIFAR-100 | S (centered) | 100 | 0.645 [0.635, 0.654] | 0.880 [0.874, 0.887] | 0.890 | 0.541 |
| MNIST | raw CLIP | 10 | 0.357 [0.347, 0.367] | 0.766 [0.758, 0.775] | 0.300 | 0.177 |
| MNIST | S (centered) | 10 | 0.501 [0.492, 0.511] | 0.869 [0.862, 0.876] | 0.300 | 0.163 |

Centering helps image→text most where CLIP is weakest (MNIST top-1 up, CIFAR-10 slightly up, CIFAR-100 within noise), but it does not uniformly improve text→image: MNIST's R-precision does not improve. With ten queries, text→image P@1 is a coarse number; R-precision is the informative one.


Text→image is class-level: each class prompt queries the test images; R-precision is precision at k = images per class (1.0 is perfect). Centering is per domain: each image dataset has its own mean (fit on its train anchors), text uses the caption-domain CLIP-text mean.

## 7. Caveats, deviations and what was not run

- **Deviations from the approved plan:** (1) the bridge and the gate are repeated over three seeds because they cost minutes, with seed 42 as the pre-declared decision seed; (2) G1 is judged on the ESC-50 test fold and G2 on all folds (stated above); (3) the untrained-bridge and permuted-label controls were added; (4) COCO captions were sub-sampled to 150k for training and duplicates across splits were removed.
- **Audio→image is class-level and goes through a text-trained bridge.** No paired audio-image data exists here, so this shows that audio and image class semantics land in the same neighbourhood of S, not that fine-grained audio-visual alignment holds.
- **Only five classes overlap** between ESC-50 and CIFAR-10 (four exact names plus 'bird' mapped to four ESC-50 bird classes), so G2 is an easy 5-way task; its high score should not be read as general audio-image alignment.
- **The S-space audio→text number is limited mostly by CLIP's text encoder** on near-synonym sound names (see the L1/L2 gap and the confusions), an instrument limit that cross-system comparisons inherit.
- **Intervals** are percentile bootstraps over test clips only; they do not cover prompt wording (see the variants row) or caption-split variation. Seed spread is shown separately.
- **Not run:** audio→caption retrieval on AudioCaps/Clotho (no audio was downloaded; Clotho audio was skipped by decision), COCO images, ImageBind (deferred to the end, optional, nothing derived from its weights may be shipped), and every channel experiment (Stage 1 and 2 are blocked on your review).
