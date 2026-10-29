import numpy as np
import pytest
import torch
import torch.nn.functional as F

from semcom.common_space import stage2_stats as st


def test_mean_ci_brackets_the_point_and_narrows_with_more_items():
    rng = np.random.default_rng(0)
    small, large = rng.random(40), rng.random(4000)
    a, b = st.mean_ci(small, st.boot_idx(40)), st.mean_ci(large, st.boot_idx(4000))
    assert a["lo"] <= a["point"] <= a["hi"] and b["lo"] <= b["point"] <= b["hi"]
    assert (a["hi"] - a["lo"]) > 3 * (b["hi"] - b["lo"])
    assert st.mean_ci(small, st.boot_idx(40, seed=1)) == st.mean_ci(small, st.boot_idx(40, seed=1))  # seeded


def test_curve_ci_uses_the_same_resamples_for_every_condition():
    x = np.vstack([np.full(50, 0.2) + np.random.default_rng(0).normal(0, 0.05, 50), np.full(50, 0.8)])
    out = st.mean_ci_curve(x, st.boot_idx(50))
    assert out[0]["lo"] < out[0]["point"] < out[0]["hi"] and out[1]["lo"] == out[1]["hi"] == pytest.approx(0.8)


def test_seed_spread_is_a_separate_mean_and_sd_over_seeds():
    s = st.seed_spread([0.5, 0.6, 0.7])
    assert s["mean"] == pytest.approx(0.6) and s["sd"] == pytest.approx(0.1) and s["values"] == [0.5, 0.6, 0.7]
    assert st.seed_spread([0.4])["sd"] == 0.0


def test_balanced_accuracy_removes_class_imbalance_and_ci_is_stratified():
    labels = np.array([0] * 90 + [1] * 10)
    always0 = (labels == 0).astype(float)
    assert always0.mean() == pytest.approx(0.9) and st.balanced_acc(always0, labels) == pytest.approx(0.5)
    out = st.balanced_acc_ci(always0, labels, n_boot=200)
    assert out["point"] == pytest.approx(0.5) and out["lo"] <= 0.5 <= out["hi"]
    curve = st.curve_balanced_acc_ci(np.vstack([always0, np.ones(100)]), labels, n_boot=100)
    assert curve[0]["point"] == pytest.approx(0.5) and curve[1]["point"] == pytest.approx(1.0)


def test_diff_ci_detects_a_real_gap_and_straddles_zero_without_one():
    rng = np.random.default_rng(0)
    gap = st.diff_ci(rng.normal(1.0, 0.2, 200), rng.normal(0.5, 0.2, 200))
    assert gap["lo"] > 0.3 and gap["point"] == pytest.approx(0.5, abs=0.1)
    none = st.diff_ci(rng.normal(0, 1, 200), rng.normal(0, 1, 200))
    assert none["lo"] < 0 < none["hi"]


def test_class_correct_and_self_retrieval():
    protos = F.normalize(torch.eye(4), dim=-1)
    q = F.normalize(torch.tensor([[1.0, 0.1, 0, 0], [0.1, 1.0, 0, 0], [0.4, 0.1, 1.0, 0]]), dim=-1)
    labels = torch.tensor([0, 1, 1])
    assert st.class_correct(q, protos, labels, 1).tolist() == [1.0, 1.0, 0.0]
    assert st.class_correct(q, protos, labels, 2).tolist() == [1.0, 1.0, 0.0]  # query 3 has its true class in 3rd place
    assert st.class_correct(q, protos, labels, 3).tolist() == [1.0, 1.0, 1.0]
    g = F.normalize(torch.randn(60, 32), dim=-1)
    perfect = st.self_retrieval(g, g)
    assert perfect["recall@1"].mean() == 1.0 and perfect["cos_to_anchor"].mean() == pytest.approx(1.0, abs=1e-5) and perfect["margin"].mean() > 0.9
    assert st.self_retrieval(F.normalize(torch.randn(60, 32), dim=-1), g)["recall@1"].mean() < 0.2


def _curves(noise_by_snr, n=300, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.8, 0.1, n)
    F_ = np.stack([base + rng.normal(0, s, n) - 0.8 * s for s in noise_by_snr])
    mis = np.stack([rng.normal(0.0, 0.1 + s, n) for s in noise_by_snr])
    return F_, mis, base


def test_f_curve_stats_on_a_monotone_response_has_cis_and_the_expected_fields():
    snrs = [0.0, 10.0, 20.0, 25.0]
    F_, mis, clean = _curves([0.5, 0.2, 0.05, 0.02])
    out = st.f_curve_stats(F_, mis, clean, snrs, st.boot_idx(300, 200))
    assert out["strictly_increasing"] and out["dynamic_range"]["lo"] > 0 and out["spearman_pooled"]["point"] > 0
    assert out["means"][0]["lo"] <= out["means"][0]["point"] <= out["means"][0]["hi"] and len(out["means"]) == 4
    assert set(out["auc"]) == {"0.0", "10.0"} and out["auc"]["0.0"]["point"] > 0.6 and out["auc"]["10.0"]["point"] > out["auc"]["0.0"]["point"]
    assert out["snr_at_90pct_clean"]["status"] in ("interpolated", "at_or_below_lowest_grid_point", "not_reached_by_grid_max")


def test_f_curve_stats_on_a_flat_response_is_not_strictly_increasing_with_confidence():
    snrs = [0.0, 10.0, 20.0, 25.0]
    F_, mis, clean = _curves([0.2, 0.2, 0.2, 0.2])
    out = st.f_curve_stats(F_, mis, clean, snrs, st.boot_idx(300, 200))
    assert out["frac_resamples_strictly_increasing"] < 0.5 and abs(out["dynamic_range"]["point"]) < 0.1


def _crit(auc, rho, noise, seeds=3):
    return [{"auc_at_0dB": auc + 0.001 * k, "pooled_spearman": rho, "item_noise_over_range": noise} for k in range(seeds)]


def test_amended_rule_uses_only_scale_free_criteria_and_needs_two_of_three():
    pairs = ["a", "b", "c"]
    crit = {"cos5": {p: _crit(0.92, 0.29, 0.97) for p in pairs}, "cen": {p: _crit(0.96, 0.42, 0.61) for p in pairs}}
    chosen, detail = st.choose_final_variant(crit)
    assert chosen == "cen" and detail["criteria_won"] == {"cos5": 0, "cen": 3}
    # two of three wins is enough, and the lower-is-better criterion is handled in the right direction
    crit["cos5"] = {p: _crit(0.92, 0.50, 0.40) for p in pairs}
    chosen, detail = st.choose_final_variant(crit)
    assert chosen == "cos5" and detail["criteria"]["item_noise_over_range"]["winner"] == "cos5" and detail["criteria"]["auc_at_0dB"]["winner"] == "cen"


def test_amended_rule_ignores_scale_and_requires_exactly_two_candidates():
    base = {p: _crit(0.9, 0.3, 0.8) for p in "abc"}
    chosen, _ = st.choose_final_variant({"cos5": base, "cen": {p: _crit(0.95, 0.4, 0.5) for p in "abc"}})
    assert chosen == "cen"  # decided by the three scale-free criteria alone: nothing else is an input
    with pytest.raises(AssertionError):
        st.choose_final_variant({"cos5": base})


def test_pair_criteria_are_scale_free():
    snrs = [0.0, 10.0, 25.0]
    F_, mis, _ = _curves([0.5, 0.2, 0.05], seed=3)
    a = st.pair_criteria(F_, mis, snrs)
    b = st.pair_criteria(3.7 * F_, 3.7 * mis, snrs)  # a pure rescaling of F must change none of the three criteria
    assert all(a[k] == pytest.approx(b[k], rel=1e-6) for k in a)
