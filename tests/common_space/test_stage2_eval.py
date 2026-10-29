import numpy as np
import pytest
import torch
import torch.nn.functional as F

from semcom.common_space import stage2_eval as ev
from semcom.common_space.fidelity import Reference, candidate_scores, derangement, make_reference
from semcom.common_space.stage2_stats import boot_idx


def _unit(n, d=32, seed=0):
    return F.normalize(torch.randn(n, d, generator=torch.Generator().manual_seed(seed)), dim=-1)


def _emb(a, sigma, draws=2, snrs=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    z = torch.stack([torch.stack([F.normalize(a + sigma[s] * torch.randn(a.shape, generator=g), dim=-1) for s in range(snrs)]) for _ in range(draws)])
    return z.half()


# ---------------------------------------------------------------- F from embeddings
def test_f_arrays_matches_the_per_item_definition_in_fidelity():
    a = F.normalize(_unit(120) - _unit(120).mean(0), dim=-1)
    z = _emb(a, [1.0, 0.5, 0.1])
    zc = F.normalize(a + 0.05 * torch.randn_like(a), dim=-1).half()
    out = ev.f_arrays(z, zc, a)
    ref = make_reference(zc.float(), a)
    assert out["ref"]["m_clean"] == pytest.approx(ref.m_clean, abs=1e-5) and out["ref"]["C"] == pytest.approx(ref.ceiling, abs=1e-5)
    s, d = 1, 0  # check one (SNR, draw) against the already-tested per-item implementation (draw-averaged arrays use the same linear algebra)
    zf = z[:, s].float().mean(0)  # not unit norm after averaging, so compare through the linear pieces instead
    sims = sum(z[dd, s].float() @ a.T for dd in range(2)) / 2
    own = sims.diag().numpy()
    base = ((sims.sum(1) - sims.diag()) / (a.size(0) - 1)).numpy()
    assert np.allclose(out["centered cosine"]["F"][s], own, atol=1e-5)
    assert np.allclose(out["F_abs"]["F"][s], (own - base) / (1 - out["ref"]["b_aa"]), atol=1e-5)
    assert np.allclose(out["F_abs"]["F"], out["F_ret"]["F"] * ref.ceiling, atol=1e-4)  # F_abs = F_ret x C holds on every item and SNR
    assert "raw cosine" not in out  # no raw baseline unless a raw-trained head supplies one


def test_f_arrays_orders_snr_and_separates_matched_from_mismatched_pairs():
    a = F.normalize(_unit(200, seed=2) - _unit(200, seed=2).mean(0), dim=-1)
    out = ev.f_arrays(_emb(a, [2.0, 0.8, 0.2]), F.normalize(a + 0.01 * torch.randn_like(a), dim=-1).half(), a)
    means = out["F_abs"]["F"].mean(1)
    assert means[0] < means[1] < means[2]  # less noise, more fidelity
    assert (out["F_abs"]["F"].mean(1) > out["F_abs"]["F_mis"].mean(1) + 0.05).all()
    assert out["F_abs"]["clean"].mean() > means[2]


def test_f_arrays_returns_the_raw_cosine_baseline_when_given():
    a = F.normalize(_unit(60, seed=3) - _unit(60, seed=3).mean(0), dim=-1)
    raw = {"cos": torch.rand(2, 3, 60), "cos_mis": torch.rand(2, 3, 60), "cos_clean": torch.rand(60)}
    out = ev.f_arrays(_emb(a, [1, 0.5, 0.1]), a.half(), a, raw)
    assert out["raw cosine"]["F"].shape == (3, 60) and np.allclose(out["raw cosine"]["F"], raw["cos"].mean(0).numpy())


def test_seed_summary_reports_the_scale_free_criteria_and_flags_monotonicity():
    a = F.normalize(_unit(150, seed=4) - _unit(150, seed=4).mean(0), dim=-1)
    out = ev.f_arrays(_emb(a, [2.0, 0.8, 0.2]), a.half(), a)
    s = ev.seed_summary(out["F_abs"]["F"], out["F_abs"]["F_mis"], out["F_abs"]["clean"], [0.0, 10.0, 25.0])
    assert s["strictly_increasing"] and s["dynamic_range"] > 0 and set(s) >= {"auc_at_0dB", "pooled_spearman", "item_noise_over_range", "snr_at_90pct_clean"}


# ---------------------------------------------------------------- class retrieval
def test_topk_correct_matches_a_hand_computed_case():
    scores = torch.tensor([[0.9, 0.1, 0.0], [0.2, 0.3, 0.9], [0.5, 0.4, 0.3]])
    labels = torch.tensor([0, 1, 2])
    assert ev.topk_correct(scores, labels, 1).tolist() == [1.0, 0.0, 0.0]
    assert ev.topk_correct(scores, labels, 2).tolist() == [1.0, 1.0, 0.0]
    assert ev.topk_correct(scores, labels, 3).tolist() == [1.0, 1.0, 1.0]


def test_class_correct_shapes_averaging_over_draws_and_subsets():
    protos = F.normalize(torch.eye(4), dim=-1)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])
    z = torch.zeros(3, 2, 6, 4)
    z[..., :] = F.normalize(torch.eye(4)[labels], dim=-1)  # every query exactly on its class prototype
    z[0, 1, 0] = F.normalize(torch.tensor([0.0, 1.0, 0.0, 0.0]), dim=-1)  # one wrong draw for item 0 at SNR index 1
    out = ev.class_correct(z.half(), protos, labels, ks=(1, 2))
    assert out[1].shape == (2, 6) and out[1][0].tolist() == [1.0] * 6
    assert out[1][1, 0] == pytest.approx(2 / 3) and out[2][1, 0] == pytest.approx(1.0)  # averaged over the 3 draws; still top-2
    sub = ev.class_correct(z.half(), protos, labels[:3], ks=(1,), sel=torch.tensor([0, 1, 2]))
    assert sub[1].shape == (2, 3)
    static = ev.class_correct_static(F.normalize(torch.eye(4)[labels], dim=-1), protos, labels, ks=(1,))
    assert static[1].tolist() == [1.0] * 6


def test_class_prototypes_are_unit_and_class_means():
    emb = F.normalize(torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9]]), dim=-1)
    p = ev.class_prototypes(emb, torch.tensor([0, 0, 1, 1]), 2)
    assert torch.allclose(p.norm(dim=-1), torch.ones(2), atol=1e-5) and p[0, 0] > p[0, 1] and p[1, 1] > p[1, 0]


def test_self_retrieval_arrays_perfect_vs_noisy_and_chance():
    a = _unit(80, 64, seed=5)
    perfect = ev.self_retrieval_arrays(a.half()[None, None].repeat(2, 2, 1, 1), a)
    assert perfect["recall@1"].min() == 1.0 and perfect["cos_to_anchor"].min() > 0.99
    noisy = ev.self_retrieval_arrays(_emb(a, [5.0, 5.0, 5.0]), a)
    assert noisy["recall@1"].mean() < perfect["recall@1"].mean() and noisy["recall@5"].mean() >= noisy["recall@1"].mean()
    static = ev.self_retrieval_static(a.half(), a)
    assert static["recall@1"].shape == (80,) and static["recall@1"].min() == 1.0


# ---------------------------------------------------------------- summaries
def test_summarize_separates_item_ci_from_seed_spread():
    rng = np.random.default_rng(0)
    seeds = [np.clip(rng.normal(0.6 + 0.05 * k, 0.2, (3, 100)), 0, 1) for k in range(3)]
    idx = boot_idx(100, 200)
    out = ev.summarize(seeds, idx)
    assert len(out["curve"]) == 3 and out["n_seeds"] == 3 and len(out["spread"]) == 3
    assert out["curve"][0]["lo"] <= out["curve"][0]["point"] <= out["curve"][0]["hi"]
    assert out["spread"][0]["sd"] == pytest.approx(np.std([x[0].mean() for x in seeds], ddof=1))
    assert "spread" not in ev.summarize(seeds[:1], idx)  # one seed has no seed spread
    labels = np.array([0] * 60 + [1] * 40)
    bal = ev.summarize([np.vstack([(labels == 0).astype(float)] * 2)], idx, labels=labels, balanced=True)
    assert bal["curve"][0]["point"] == pytest.approx(0.5)  # a majority-class predictor is at chance under balanced accuracy
    st = ev.summarize_static(np.ones(100), idx)
    assert st["value"]["point"] == 1.0 and st["n_items"] == 100


def test_paired_difference_is_the_item_level_difference_and_detects_a_real_gap():
    rng = np.random.default_rng(0)
    base = rng.random((3, 200))
    a = [np.clip(base + 0.1, 0, 1) for _ in range(2)]  # two seeds, uniformly 0.1 better than B on every item
    out = ev.paired_difference(a, [base], boot_idx(200, 200))
    assert all(c["lo"] > 0.05 and c["point"] == pytest.approx(0.1, abs=0.02) for c in out)
    same = ev.paired_difference([base], [base], boot_idx(200, 200))
    assert all(c["point"] == 0.0 and c["lo"] == 0.0 and c["hi"] == 0.0 for c in same)  # identical systems: zero difference, zero-width interval
    labels = np.array([0] * 150 + [1] * 50)
    bal = ev.paired_difference([np.vstack([np.ones(200)] * 2)], [np.vstack([(labels == 0).astype(float)] * 2)], boot_idx(200, 100), labels, True)
    assert bal[0]["point"] == pytest.approx(0.5)  # balanced: A is perfect (1.0); B is right only on the majority class (0.5)
