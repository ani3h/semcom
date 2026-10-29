import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from semcom.common_space import domains as dom
from semcom.common_space.bridge import TextBridge
from semcom.common_space.correction import Centerer
from semcom.common_space.fidelity import (CANDIDATES, Reference, analyze, anchor_baseline, auc, candidate_scores, choose_variant,
                                          derangement, make_reference, snr_at_fraction, unrelated_baseline)


def _anchors(n=300, d=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = F.normalize(torch.randn(n, d, generator=g), dim=-1)
    return F.normalize(a - a.mean(0), dim=-1)  # centered, as S anchors are


def _noisy(a, sigma, seed=1):  # NB: callers must not use seed 0 (that is the stream _anchors() uses)
    g = torch.Generator().manual_seed(seed)
    return F.normalize(a + sigma * torch.randn(a.shape, generator=g), dim=-1)


def _scores(a, sigma, ref, seed=1):
    z = _noisy(a, sigma, seed)
    return candidate_scores(z, a, z, a, ref)


# ---------------------------------------------------------------- F definitions
def test_f_abs_equals_f_ret_times_ceiling_exactly():
    a = _anchors()
    ref = make_reference(_noisy(a, 0.3), a)
    s = _scores(a, 1.0, ref)
    assert torch.allclose(s["F_abs"], s["F_ret"] * ref.ceiling, atol=1e-5)


def test_f_ret_is_one_at_the_clean_reference_and_f_abs_is_one_for_a_perfect_embedding():
    a = _anchors()
    z_clean = _noisy(a, 0.5)
    ref = make_reference(z_clean, a)
    assert candidate_scores(z_clean, a, z_clean, a, ref)["F_ret"].mean().item() == pytest.approx(1.0, abs=1e-5)
    perfect = candidate_scores(a, a, a, a, make_reference(a, a))
    assert perfect["F_abs"].mean().item() == pytest.approx(1.0, abs=1e-5) and perfect["centered cosine"].mean().item() == pytest.approx(1.0, abs=1e-5)


def test_f_abs_keeps_encoder_loss_while_f_ret_hides_it():
    """Two systems with the same channel behaviour but different clean quality: F_ret cannot tell them apart, F_abs can."""
    a = _anchors(400)
    out = {}
    for name, enc_noise in (("good encoder", 0.2), ("poor encoder", 1.0)):
        z_clean = _noisy(a, enc_noise)
        ref = make_reference(z_clean, a)
        out[name] = candidate_scores(_noisy(a, enc_noise * 1.5, seed=2), a, _noisy(a, enc_noise * 1.5, seed=2), a, ref)
    assert out["good encoder"]["F_abs"].mean() > out["poor encoder"]["F_abs"].mean() + 0.2
    assert abs(out["good encoder"]["F_ret"].mean() - out["poor encoder"]["F_ret"].mean()) < 0.35  # saturates / hides the gap


def test_f_decreases_monotonically_with_noise_and_is_bounded_by_the_perfect_value():
    a = _anchors(400)
    ref = Reference(m_clean=1.0, b_aa=anchor_baseline(a))
    means = [_scores(a, s, ref)["F_abs"].mean().item() for s in (0.05, 0.3, 1.0, 3.0)]
    assert all(x > y for x, y in zip(means, means[1:])) and means[0] <= 1.0 + 1e-6
    assert abs(_scores(a, 50.0, ref)["F_abs"].mean().item()) < 0.1  # pure noise carries no fidelity


def test_unrelated_baseline_is_near_zero_in_centered_space_and_large_in_raw_space():
    raw = F.normalize(torch.randn(300, 64) + 4.0, dim=-1)
    assert anchor_baseline(raw) > 0.8
    assert abs(anchor_baseline(_anchors())) < 0.05
    z = _anchors()
    assert unrelated_baseline(z, z).shape == (300,)


def test_auc_perfect_chance_and_reversed():
    pos, neg = np.linspace(1, 2, 100), np.linspace(-1, 0, 100)
    assert auc(pos, neg) == 1.0 and auc(neg, pos) == 0.0
    rng = np.random.default_rng(0)
    assert abs(auc(rng.random(2000), rng.random(2000)) - 0.5) < 0.04


def test_derangement_has_no_fixed_points():
    for seed in range(5):
        assert (derangement(50, seed) != torch.arange(50)).all()


def test_snr_at_fraction_interpolates_and_reports_unreached_cases():
    snrs = [0.0, 10.0, 20.0]
    assert snr_at_fraction(snrs, [0.2, 0.6, 1.0], 1.0)["snr"] == pytest.approx(17.5)  # 0.9 is 3/4 of the way from 0.6 (10 dB) to 1.0 (20 dB)
    assert snr_at_fraction(snrs, [0.95, 0.99, 1.0], 1.0)["status"] == "at_or_below_lowest_grid_point"
    assert snr_at_fraction(snrs, [0.1, 0.2, 0.3], 1.0)["status"] == "not_reached_by_grid_max"
    assert snr_at_fraction(snrs, [0.1, 0.2, 0.3], -1.0)["status"] == "undefined"


def _analysis(a, sigmas, draws=3):
    ref = make_reference(_noisy(a, sigmas[-1] / 2), a)
    per, mism = {}, {}
    perm = derangement(a.size(0))
    for k, (snr, sig) in enumerate(zip((0.0, 10.0, 20.0), sigmas)):
        rows, mrows = [], []
        for d in range(draws):
            z = _noisy(a, sig, seed=100 + 10 * k + d)  # offset: seed 0 would replay the stream that built the anchors
            rows.append(candidate_scores(z, a, z, a, ref)["F_abs"].numpy())
            mrows.append(candidate_scores(z, a, z, a, ref, perm)["F_abs"].numpy())
        per[snr], mism[snr] = np.stack(rows), np.stack(mrows)
    clean = candidate_scores(_noisy(a, 0.01, seed=7), a, _noisy(a, 0.01, seed=7), a, ref)["F_abs"].numpy()
    return analyze(per, mism, clean, n_boot=200)


def test_analyze_detects_a_monotone_trend_and_reports_all_statistics():
    r = _analysis(_anchors(400), sigmas=(3.0, 1.0, 0.3))  # less noise at higher SNR
    assert r["strictly_increasing"] and r["steps_significant"] == r["n_steps"] == 2
    assert r["dynamic_range"] > 0.3 and r["spearman_pooled"] > 0.3
    assert all(0.5 < v <= 1.0 for v in r["auc"].values()) and r["auc"]["20.0"] > r["auc"]["0.0"]
    assert set(r) >= {"means", "clean", "dynamic_range", "spearman_pooled", "snr_at_90pct_clean", "auc", "item_sd_over_range_at_10dB"}


def test_analyze_flags_a_non_monotone_or_flat_response():
    r = _analysis(_anchors(400), sigmas=(1.0, 1.0, 1.0))  # no SNR dependence at all
    assert not r["strictly_increasing"] or r["steps_significant"] < r["n_steps"]
    assert abs(r["dynamic_range"]) < 0.05


def test_choose_variant_follows_the_preregistered_order():
    def pair(inc, rng, auc0):
        return {"strictly_increasing": inc, "dynamic_range": rng, "auc": {"0.0": auc0, "25.0": 0.99}}

    stats = {"cos": {"a": pair(True, 0.05, 0.9), "b": pair(True, 0.05, 0.9)},
             "cos5": {"a": pair(False, 0.5, 0.99), "b": pair(True, 0.5, 0.99)},
             "cen": {"a": pair(True, 0.2, 0.95), "b": pair(True, 0.2, 0.95)}}
    chosen, summary = choose_variant(stats)
    assert chosen == "cen" and summary["cen"]["n_strictly_increasing"] == 2  # 2 increasing beats cos5's 1; the range breaks the tie with cos
    stats["cen"]["b"]["strictly_increasing"] = False
    assert choose_variant(stats)[0] == "cos"  # now cos has 2 strictly increasing pairs and the others only 1


# ---------------------------------------------------------------- domains
def test_get_domain_fits_once_caches_and_attaches_the_bridge_only_for_audio(tmp_path, monkeypatch):
    anchors = tmp_path / "anchors"
    (anchors).mkdir()
    for task in ("cifar10", "audio"):
        torch.save({"anchors": _anchors(200, 512, seed=3).half()}, anchors / f"{task}_train.pt")
    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    torch.save({"state": TextBridge("linear").state_dict(), "kind": "linear", "hidden": 1024, "seed": 42}, bridge_dir / "linear_s42.pt")
    (bridge_dir / "results.json").write_text(json.dumps({"seeds": {"42": {"chosen": "linear"}}}))
    space = tmp_path / "space"
    img = dom.get_domain("cifar10", anchors, space, bridge_dir)
    aud = dom.get_domain("audio", anchors, space, bridge_dir)
    assert img.bridge is None and aud.bridge is not None
    assert (space / "centerer_cifar10.pt").exists() and (space / "centerer_audio.pt").exists()
    mu = img.centerer.mu.clone()
    assert torch.equal(dom.get_domain("cifar10", anchors, space, bridge_dir).centerer.mu, mu)  # second call loads the cache
    x = _anchors(20, 512, seed=9)
    assert img.to_S(x).shape == (20, 512) and torch.allclose(aud.to_S(x).norm(dim=-1), torch.ones(20), atol=1e-4)
    with pytest.raises(ValueError):
        dom.get_domain("video", anchors, space, bridge_dir)


def test_esc50_centering_refuses_anchors_that_include_later_folds(tmp_path):
    folds = torch.tensor([1] * 400 + [2] * 400 + [3] * 400 + [4] * 400 + [5] * 400)
    root = tmp_path / "anchors"
    root.mkdir()
    torch.save({"anchors": _anchors(2000, 8).half(), "folds": folds}, root / "esc50_all_clap_16k.pt")
    assert dom.esc50_train_anchors(root).shape[0] == 1200  # exactly folds 1-3
    torch.save({"anchors": _anchors(2000, 8).half(), "folds": torch.clamp(folds, max=3)}, root / "esc50_all_clap_16k.pt")
    with pytest.raises(AssertionError):
        dom.esc50_train_anchors(root)  # a fit set that is not folds 1-3 only is rejected
