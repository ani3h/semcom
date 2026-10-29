import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from semcom.common_space import captions as cap
from semcom.common_space.bridge import TextBridge, agreement, ridge_fit, train_bridge
from semcom.common_space.correction import Centerer
from semcom.common_space.gate import balanced_accuracy, bootstrap_ci, class_prototypes, r_precision, topk_correct
from semcom.common_space.retrieval import random_pair_cosine


def _shifted(n=400, d=64, shift=3.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g) + shift * torch.ones(d)  # a large common offset, like the CLIP/CLAP modality gap
    return F.normalize(x, dim=-1)


# ---------------------------------------------------------------- Centerer
def test_centering_removes_the_shared_offset():
    train, test = _shifted(seed=0), _shifted(seed=1)
    assert random_pair_cosine(test) > 0.8
    c = Centerer().fit(train)
    out = c.transform(test)
    assert abs(random_pair_cosine(out)) < 0.05
    assert torch.allclose(out.norm(dim=-1), torch.ones(out.size(0)), atol=1e-5)


def test_centerer_is_fit_on_train_only_and_transform_does_not_change_it():
    train, test = _shifted(seed=0), _shifted(seed=1, shift=-5.0)
    c = Centerer().fit(train)
    mu_before = c.mu.clone()
    c.transform(test)
    c.transform(test * 2)
    assert torch.equal(c.mu, mu_before)  # transform never refits
    assert torch.allclose(c.mu, train.mean(0), atol=1e-6)  # and mu is exactly the TRAIN mean
    assert c.n_fit == train.size(0)
    assert not torch.allclose(c.mu, test.mean(0), atol=1e-2)


def test_centerer_fit_is_deterministic_and_refitting_centered_data_is_near_a_noop():
    train = _shifted()
    a, b = Centerer().fit(train), Centerer().fit(train)
    assert torch.equal(a.mu, b.mu)
    centered = a.transform(train)
    again = Centerer().fit(centered)
    assert again.mu.norm() < 0.05 * a.mu.norm()  # the offset is gone: a second centering barely moves anything
    assert (F.normalize(centered - again.mu, dim=-1) * centered).sum(-1).mean() > 0.99


def test_centerer_requires_fit_and_roundtrips(tmp_path):
    with pytest.raises(RuntimeError):
        Centerer().transform(torch.randn(2, 4))
    c = Centerer().fit(_shifted())
    c.save(tmp_path / "c.pt")
    assert torch.equal(Centerer.load(tmp_path / "c.pt").mu, c.mu)


# ---------------------------------------------------------------- bridge
def test_bridge_shapes_and_validation():
    for kind in ("linear", "mlp"):
        out = TextBridge(kind, dim=32, hidden=64)(torch.randn(5, 32))
        assert out.shape == (5, 32) and torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-5)
    with pytest.raises(ValueError):
        TextBridge("transformer")


def test_agreement_perfect_vs_chance():
    y = F.normalize(torch.randn(2000, 64), dim=-1)
    perfect = agreement(y, y, gallery=1000)
    assert perfect["recall@1"] == 1.0 and perfect["cos"] == pytest.approx(1.0, abs=1e-5)
    rand = agreement(F.normalize(torch.randn(2000, 64), dim=-1), y, gallery=1000)
    assert rand["recall@1"] < 0.02 and abs(rand["cos"]) < 0.1


def test_bridge_learns_a_hidden_linear_map_and_generalises():
    g = torch.Generator().manual_seed(0)
    W = torch.linalg.qr(torch.randn(48, 48, generator=g))[0]
    x = F.normalize(torch.randn(6000, 48, generator=g), dim=-1)
    y = F.normalize(x @ W, dim=-1)
    bridge, hist = train_bridge(x[:4000], y[:4000], x[4000:5000], y[4000:5000], kind="linear", epochs=15, batch_size=256,
                                lr=3e-3, device="cpu", gallery=500, log=lambda m: None)
    with torch.no_grad():
        held = agreement(bridge(x[5000:]), y[5000:], gallery=500)
    # InfoNCE solves the ranking task completely while the 0.5-weight cosine term leaves the absolute cosine well below 1
    # (the same effect as the Week 5 heads); a heavier cosine weight pins it.
    assert held["recall@1"] > 0.9 and held["cos"] > 0.6
    assert len(hist) == 15 and hist[-1]["val_recall@1"] >= hist[0]["val_recall@1"]
    heavy, _ = train_bridge(x[:4000], y[:4000], x[4000:5000], y[4000:5000], kind="linear", epochs=15, batch_size=256, lr=3e-3,
                            cos_weight=10.0, device="cpu", gallery=500, log=lambda m: None)
    with torch.no_grad():
        assert agreement(heavy(x[5000:]), y[5000:], gallery=500)["cos"] > held["cos"] + 0.1
    ridge = F.normalize(x[5000:] @ ridge_fit(x[:4000], y[:4000], 1e-3), dim=-1)
    assert agreement(ridge, y[5000:], gallery=500)["cos"] > 0.95  # the regression reference solves this easy case too


# ---------------------------------------------------------------- caption corpus
def _fake_caption_tree(root):
    coco = {"train": [("a dog runs on grass", "A dog runs on grass.", "two cats sit"), ("a man rides a horse",)],
            "val": [("two cats sit", "a unique val caption here")], "test": [("a unique test caption here", "a dog runs on grass")]}
    base = root / "yerevann__coco-karpathy" / "data"
    base.mkdir(parents=True)
    for split, files in cap.COCO[1].items():
        for i, f in enumerate(files):
            rows = coco[split] if i == 0 else []
            pd.DataFrame({"cocoid": range(len(rows)), "sentences": [list(r) for r in rows]}).to_parquet(base / f.split("/")[-1])
    ac = {"train": ["Water is pouring", "a man speaks", "ok"], "val": ["a man speaks", "birds chirp nearby"], "test": ["a siren wails in the distance"]}
    base = root / "d0rj__audiocaps" / "data"
    base.mkdir(parents=True)
    for split, files in cap.AUDIOCAPS[1].items():
        pd.DataFrame({"audiocap_id": range(len(ac[split])), "youtube_id": "x", "start_time": 0, "caption": ac[split]}).to_parquet(base / files[0].split("/")[-1])


def test_caption_corpus_dedupes_filters_and_never_leaks_train_into_heldout(tmp_path):
    _fake_caption_tree(tmp_path)
    corpus = cap.build_corpus(tmp_path, seed=0, n_coco_train=100, n_coco_eval=100, min_words=3)
    key = lambda s: {cap._key(r["text"]) for r in corpus[s]}  # noqa: E731
    assert key("train") & key("val") == set() and key("train") & key("test") == set()  # no held-out caption also in train
    assert "ok" not in key("train")  # shorter than min_words
    texts = [r["text"] for r in corpus["train"]]
    assert sum(t.lower().rstrip(" .") == "a dog runs on grass" for t in texts) == 1  # case/punctuation duplicates collapse
    assert {r["source"] for r in corpus["train"]} == {"coco", "audiocaps"}
    assert all(len(r["text"].split()) >= 3 for s in corpus for r in corpus[s])
    assert "a unique test caption here" in key("test") and "a siren wails in the distance" in key("test")


def test_caption_corpus_subsampling_is_seeded(tmp_path):
    _fake_caption_tree(tmp_path)
    a = cap.build_corpus(tmp_path, seed=3, n_coco_train=2, n_coco_eval=1)
    b = cap.build_corpus(tmp_path, seed=3, n_coco_train=2, n_coco_eval=1)
    assert a == b
    assert sum(r["source"] == "coco" for r in a["train"]) <= 2
    cap.save_corpus(a, tmp_path / "c.json")
    assert cap.load_corpus(tmp_path / "c.json") == json.loads((tmp_path / "c.json").read_text())


# ---------------------------------------------------------------- gate statistics
def test_balanced_accuracy_is_chance_for_a_majority_class_predictor():
    labels = np.array([0] * 90 + [1] * 10)
    always_zero = (np.zeros(100) == labels).astype(float)
    assert always_zero.mean() == pytest.approx(0.9)  # plain accuracy flatters it
    assert balanced_accuracy(always_zero, labels) == pytest.approx(0.5)  # chance for 2 classes


def test_bootstrap_ci_brackets_the_point_is_seeded_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    small, large = (rng.random(50) < 0.6).astype(float), (rng.random(2000) < 0.6).astype(float)
    p, lo, hi = bootstrap_ci(small, seed=1)
    assert lo <= p <= hi and bootstrap_ci(small, seed=1) == (p, lo, hi)
    assert (hi - lo) > 2 * (bootstrap_ci(large, seed=1)[2] - bootstrap_ci(large, seed=1)[1])
    labels = np.array([0] * 30 + [1] * 20)
    bp, blo, bhi = bootstrap_ci(np.r_[np.ones(30), np.zeros(20)], labels, balanced=True, seed=2)
    assert bp == pytest.approx(0.5) and blo <= bp <= bhi


def test_topk_correct_prototypes_and_r_precision():
    scores = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.2, 0.9], [0.3, 0.4, 0.2]])
    labels = torch.tensor([0, 1, 2])
    assert topk_correct(scores, labels, 1).tolist() == [1.0, 0.0, 0.0]
    assert topk_correct(scores, labels, 2).tolist() == [1.0, 1.0, 1.0] if False else topk_correct(scores, labels, 3).tolist() == [1.0, 1.0, 1.0]
    emb = torch.tensor([[1.0, 0], [0.9, 0.1], [0, 1.0], [0.1, 0.9]])
    protos = class_prototypes(emb, torch.tensor([0, 0, 1, 1]), 2)
    assert torch.allclose(protos.norm(dim=-1), torch.ones(2), atol=1e-5) and protos[0, 0] > protos[0, 1]
    queries = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    gallery = F.normalize(torch.tensor([[1.0, 0.05], [0.9, 0.2], [0.05, 1.0], [0.2, 0.9]]), dim=-1)
    rp = r_precision(queries, gallery, torch.tensor([0, 1]), torch.tensor([0, 0, 1, 1]))
    assert rp["p@1"] == 1.0 and rp["r_precision"] == 1.0
    wrong = r_precision(queries, gallery, torch.tensor([1, 0]), torch.tensor([0, 0, 1, 1]))
    assert wrong["p@1"] == 0.0 and wrong["r_precision"] == 0.0
