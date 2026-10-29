import pytest
import torch
import torch.nn.functional as F

from semcom.common_space.adapters import detokenize
from semcom.common_space.anchors import anchor_path, load_anchors, save_anchors
from semcom.common_space.heads import build_head
from semcom.common_space.loader import AnchoredData
from semcom.common_space.losses import AnchorLoss
from semcom.common_space.retrieval import alignment, class_topk, mean_cosine, random_pair_cosine, recall_at_k, unrelated_cosine


def _unit(n, d=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_recall_at_k_known_ranks():
    gallery = torch.eye(5)
    queries = torch.eye(5).clone()
    queries[0] = F.normalize(torch.tensor([1.0, 2.0, 0, 0, 0]), dim=0)  # true match (col 0) ranks 2nd behind col 1
    r = recall_at_k(queries, gallery, ks=(1, 2, 5))
    assert r["recall@1"] == pytest.approx(0.8) and r["recall@2"] == 1.0 and r["recall@5"] == 1.0


def test_recall_perfect_and_chance_level():
    g = _unit(200)
    assert recall_at_k(g, g, ks=(1,))["recall@1"] == 1.0
    assert recall_at_k(_unit(200, seed=1), g, ks=(1,))["recall@1"] < 0.05


def test_class_topk():
    prompts = torch.eye(4)
    emb = F.normalize(torch.tensor([[1.0, 0.1, 0, 0], [0.1, 1.0, 0, 0], [0.3, 0.4, 1.0, 0]]), dim=-1)
    labels = torch.tensor([0, 1, 1])  # third sample is closest to class 2 but labelled 1 -> top-1 wrong, top-2 right
    r = class_topk(emb, prompts, labels, ks=(1, 2, 5))
    assert r["top1"] == pytest.approx(2 / 3) and r["top2"] == 1.0 and "top5" not in r  # k>#classes dropped


def test_random_pair_cosine_extremes_and_mean_cosine():
    assert abs(random_pair_cosine(torch.ones(50, 8) / 8**0.5) - 1.0) < 1e-5
    assert abs(random_pair_cosine(_unit(500, d=512))) < 0.02  # unrelated high-dim vectors ~ orthogonal
    a = _unit(10)
    assert abs(mean_cosine(a, a) - 1.0) < 1e-5


def test_anchor_loss_terms_and_temperature_clamp():
    z, a = _unit(16), _unit(16, seed=1)
    both = AnchorLoss(cos_weight=0.5)(z, a)
    nce_only = AnchorLoss(cos_weight=0.0)(z, a)
    assert torch.allclose(both["total"], both["nce"] + 0.5 * both["cos"])
    assert torch.allclose(nce_only["total"], nce_only["nce"])  # ablation: cosine term off, still logged
    assert nce_only["cos"] > 0
    loss = AnchorLoss(init_temp=0.07, min_temp=0.02, max_temp=0.3)
    loss.log_inv_temp.data.fill_(10.0)
    assert abs(loss.temperature.item() - 0.02) < 1e-6
    loss.log_inv_temp.data.fill_(-10.0)
    assert abs(loss.temperature.item() - 0.3) < 1e-6


def test_anchor_loss_prefers_the_matching_pairing():
    a = _unit(32)
    lf = AnchorLoss(cos_weight=0.0)
    assert lf(a, a)["nce"] < lf(a, a.roll(1, dims=0))["nce"]


def test_detokenize_attaches_punctuation_and_drops_unk():
    assert detokenize(["the", "vote", ",", "mr", "<unk>", "president", "(", "applause", ")", "."]) == \
        "the vote, mr president (applause)."


def test_anchor_cache_roundtrip_and_renormalises(tmp_path):
    p = anchor_path(tmp_path, "cifar10", "test")
    x = _unit(20, d=512) * 3.0  # un-normalised on purpose
    save_anchors(p, x, model="stub")
    blob = load_anchors(p)
    assert blob["model"] == "stub" and blob["anchors"].dtype == torch.float32
    assert torch.allclose(blob["anchors"].norm(dim=-1), torch.ones(20), atol=1e-5)


def test_anchored_data_keeps_index_alignment_and_rejects_stale_cache():
    class DS:
        def __len__(self):
            return 6

        def __getitem__(self, i):
            return {"input": torch.tensor(float(i)), "label": None, "raw": None}

        def get_crop(self, utt, start):
            return torch.tensor(float(utt * 100 + start))

    anchors = torch.arange(6.0).view(6, 1)
    items, a = AnchoredData(DS(), anchors).fetch([4, 1])
    assert [it["input"].item() for it in items] == [4.0, 1.0] and a.flatten().tolist() == [4.0, 1.0]
    crops = (torch.tensor([2, 2, 5]), torch.tensor([0, 7, 3]))
    items, a = AnchoredData(DS(), torch.arange(3.0).view(3, 1), crops=crops).fetch([1, 2])
    assert [it["input"].item() for it in items] == [207.0, 503.0] and a.flatten().tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="stale"):
        AnchoredData(DS(), torch.zeros(5, 1))


def test_build_head_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_head("transformer_xl", sigma=1.0)


def test_alignment_margin_separates_matched_from_unrelated():
    a = _unit(50, d=128)
    perfect = alignment(a, a)
    assert abs(perfect["cos_to_anchor"] - 1.0) < 1e-5 and abs(perfect["cos_unrelated"]) < 0.1
    assert perfect["margin"] > 0.9
    shuffled = alignment(a.roll(1, dims=0), a)  # every embedding paired with the wrong anchor
    assert abs(shuffled["margin"]) < 0.1
    assert abs(unrelated_cosine(torch.ones(4, 3) / 3**0.5, torch.ones(4, 3) / 3**0.5) - 1.0) < 1e-5
