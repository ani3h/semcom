import numpy as np
import pytest
import torch

from encoders.builders import TASKS, ListDataset  # tiny CPU-sized transceivers + synthetic data
from semcom.channel.awgn import AWGNChannel
from semcom.channel.rayleigh import RayleighChannel
from semcom.common_space.adapters import make_adapter
from semcom.common_space.heads import build_head
from semcom.common_space.learned import LearnedHeadSpace
from semcom.common_space.loader import AnchoredData, Week4Model
from semcom.common_space.losses import AnchorLoss
from semcom.common_space.training import EarlyStopper, LRSchedule, SymbolCache, cosine_factor, validate

from .stubs import StubAnchors
from .test_spaces import HEAD_CFG


# ---------------------------------------------------------------- schedules
def _opt(lr=1.0):
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def test_cosine_schedule_matches_torch_lambda_lr_exactly():
    base, warm, total = 1e-3, 5, 40
    ours_opt, ref_opt = _opt(base), _opt(base)
    ours = LRSchedule(ours_opt, base, warm, "cosine", total, 0.05)
    ref = torch.optim.lr_scheduler.LambdaLR(ref_opt, cosine_factor(warm, total, 0.05))
    for _ in range(total):
        assert ours.lr == np.float64(ref_opt.param_groups[0]["lr"]) or abs(ours.lr - ref_opt.param_groups[0]["lr"]) < 1e-12
        ours.step(), ref.step()


def test_plateau_schedule_warms_up_holds_then_decays_after_patience_and_respects_the_floor():
    opt = _opt(1.0)
    s = LRSchedule(opt, 1.0, warmup=4, kind="plateau", min_frac=0.05, factor=0.5, patience=2)
    lrs = [s.lr]
    for _ in range(4):
        s.step()
        lrs.append(s.lr)
    assert lrs[0] < lrs[1] < lrs[2] < lrs[3] <= 1.0 and lrs[-1] == 1.0  # linear warmup to the base rate
    s.epoch_end(True), s.epoch_end(False)
    assert s.lr == 1.0  # one bad epoch is within patience
    s.epoch_end(False)
    assert abs(s.lr - 0.5) < 1e-9  # second consecutive bad epoch triggers the decay
    s.epoch_end(True)
    assert abs(s.lr - 0.5) < 1e-9  # an improvement resets the counter and keeps the rate
    for _ in range(40):
        s.epoch_end(False)
    assert abs(s.lr - 0.05) < 1e-9  # never below min_frac x base


def test_early_stopper_counts_consecutive_non_improvements():
    es = EarlyStopper(patience=3)
    results = [es.update(v, e + 1) for e, v in enumerate([5.0, 4.0, 4.5, 4.2, 4.0, 3.9, 4.1, 4.0, 4.0])]
    assert [r[0] for r in results] == [True, True, False, False, False, True, False, False, False]
    assert [r[1] for r in results] == [False, False, False, False, True, False, False, False, True]
    assert es.best == 3.9 and es.best_epoch == 6
    assert not EarlyStopper(None).update(1.0, 1)[1] and not EarlyStopper(None).update(2.0, 2)[1]  # patience=None never stops


# ---------------------------------------------------------------- symbol cache
def _setup(name):
    task_fn, ds_fn = TASKS[name]
    task, ds = task_fn(), ds_fn()
    model = task.build_model().eval().requires_grad_(False)
    w4 = Week4Model(name, "awgn", task, model, None)
    adapter = make_adapter(name, StubAnchors(), task)
    kind, kw = HEAD_CFG[name]
    head = build_head(kind, sigma=1.0, **kw).eval()
    n = len(ds)
    data = AnchoredData(ds, torch.nn.functional.normalize(torch.randn(n, 512), dim=-1))
    return w4, adapter, head, data


def test_cached_embeddings_equal_the_uncached_path_for_the_same_noise_seed():
    """The cache is only a speed-up: head(channel(cached symbols)) must equal LearnedHeadSpace.embed_received exactly."""
    for name in ("text", "cifar10", "audio"):
        w4, adapter, head, data = _setup(name)
        cache = SymbolCache.build(w4, adapter, data, torch.device("cpu"), chunk=3)
        space = LearnedHeadSpace(w4.model, head, adapter)
        for channel in (AWGNChannel(), RayleighChannel()):
            idx = np.array([0, 2, 1]) if len(data) >= 3 else np.arange(len(data))
            items, _ = data.fetch(idx)
            batch = w4.spec.collate(items)
            torch.manual_seed(7)
            ref = space.embed_received(batch, channel, 10.0)
            torch.manual_seed(7)
            rx, side = cache.get(idx)
            got = head(channel(rx, 10.0), **side)
            assert got.shape == ref.shape and torch.allclose(got, ref, atol=1e-5), name


def test_text_cache_slices_each_batch_to_its_own_longest_sentence():
    w4, adapter, head, data = _setup("text")
    cache = SymbolCache.build(w4, adapter, data, torch.device("cpu"), chunk=4)
    assert cache.sym.shape[1] == w4.spec.tokenizer.max_len + 1 and cache.lengths is not None
    rx, side = cache.get(np.array([0, 1]))
    assert rx.size(1) == int(cache.lengths[[0, 1]].max()) and side["mask"].shape == rx.shape[:2]
    assert (side["mask"].sum(1) == cache.lengths[[0, 1]]).all()


def test_symbol_cache_roundtrips_on_disk_and_rebuilds_when_stale(tmp_path):
    w4, adapter, head, data = _setup("cifar10")
    p = tmp_path / "c" / "cifar10_val.pt"
    a = SymbolCache.load_or_build(p, w4, adapter, data, torch.device("cpu"))
    b = SymbolCache.load_or_build(p, w4, adapter, data, torch.device("cpu"))
    assert p.exists() and torch.equal(a.sym, b.sym) and not list(p.parent.glob("*.tmp"))
    short = AnchoredData(ListDataset(data.ds.items[:4]), data.anchors[:4])  # a cache for a different number of items is ignored, not trusted
    c = SymbolCache.load_or_build(p, w4, adapter, short, torch.device("cpu"))
    assert len(c) == 4 and len(SymbolCache.load_or_build(p, w4, adapter, short, torch.device("cpu"))) == 4


def test_validation_is_deterministic_and_reports_every_term():
    w4, adapter, head, data = _setup("mnist")
    cache = SymbolCache.build(w4, adapter, data, torch.device("cpu"))
    loss = AnchorLoss()
    a = validate(head, loss, data.anchors, AWGNChannel(), 10.0, torch.device("cpu"), 4, 3, cache)
    b = validate(head, loss, data.anchors, AWGNChannel(), 10.0, torch.device("cpu"), 4, 3, cache)
    assert a == b and set(a) == {"val/total", "val/nce", "val/cos", "val/cos_to_anchor"}


# ---------------------------------------------------------------- sync-free training paths
def test_get_dev_is_identical_to_get_for_text_and_fixed_size_symbols():
    from semcom.common_space.training import clip_and_guard  # noqa: F401  (import check)

    for name in ("text", "cifar10"):
        w4, adapter, head, data = _setup(name)
        cache = SymbolCache.build(w4, adapter, data, torch.device("cpu"), chunk=3)
        perm = np.random.default_rng(0).permutation(len(data))
        bs = 4
        widths = cache.widths(perm, bs, len(data) // bs)
        for b in range(len(data) // bs):
            idx = perm[b * bs:(b + 1) * bs]
            ref_rx, ref_side = cache.get(idx)
            rx, side = cache.get_dev(torch.as_tensor(idx), None if widths is None else widths[b])
            w = ref_rx.size(1)
            assert rx.size(1) >= w and torch.equal(rx[:, :w], ref_rx)  # the real columns are identical
            if name == "text":
                assert rx.size(1) % 8 == 0 or rx.size(1) == cache.sym.size(1)  # bucketed width (multiple of 8, or the cache width)
                assert torch.equal(side["mask"][:, :w], ref_side["mask"]) and not side["mask"][:, w:].any()  # extra columns are masked padding
            else:
                assert not side and not ref_side
        assert (widths is None) == (name != "text")


def test_clip_and_guard_clips_and_zeroes_non_finite_entries_without_skipping_state():
    from semcom.common_space.training import clip_and_guard

    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])
    norm = clip_and_guard([p], 1.0)
    assert norm.item() == pytest.approx(5.0) and p.grad.norm().item() == pytest.approx(1.0, abs=1e-4)  # clipped to the max norm
    p.grad = torch.tensor([float("nan"), float("inf"), 3.0, 4.0])
    clip_and_guard([p], 100.0)
    assert torch.isfinite(p.grad).all() and p.grad.tolist() == [0.0, 0.0, 3.0, 4.0]  # bad entries zeroed, finite ones untouched


def test_text_head_output_is_unchanged_by_bucketed_extra_padding():
    """The numerical-equivalence claim behind width bucketing, on a real head and real masking."""
    w4, adapter, head, data = _setup("text")
    cache = SymbolCache.build(w4, adapter, data, torch.device("cpu"))
    idx = torch.arange(len(data))
    rx, side = cache.get_dev(idx, int(cache.lengths.max()))
    rx2, side2 = cache.get_dev(idx, cache.sym.size(1))  # maximum padding
    head.eval()
    with torch.no_grad():
        assert torch.allclose(head(rx, **side), head(rx2, **side2), atol=1e-5)


# ---------------------------------------------------------------- length-sorted chunking (a speed-up that must not change the computation)
def test_sorting_within_batches_keeps_batch_membership_and_orders_by_length():
    from semcom.common_space.training import sort_within_batches

    rng = np.random.default_rng(0)
    lengths = rng.integers(3, 40, size=64)
    perm = rng.permutation(64)
    out = sort_within_batches(perm, lengths, 16, 4)
    for b in range(4):
        sl = slice(b * 16, (b + 1) * 16)
        assert set(out[sl]) == set(perm[sl])  # same members, so the same in-batch negatives
        assert (np.diff(lengths[out[sl]]) <= 0).all()  # descending length inside the batch


def test_chunk_widths_always_cover_the_longest_sentence_in_each_chunk_and_respect_the_cap():
    from semcom.common_space.training import chunk_bounds, chunk_widths, sort_within_batches

    rng = np.random.default_rng(1)
    lengths = rng.integers(1, 41, size=96)
    perm = sort_within_batches(rng.permutation(96), lengths, 32, 3)
    for k in (1, 2, 3, 4):
        widths = chunk_widths(perm, lengths, 32, 3, k, bucket=4, cap=41)
        for b in range(3):
            ln = lengths[perm[b * 32:(b + 1) * 32]]
            for (a, c), w in zip(chunk_bounds(32, k), widths[b]):
                assert w >= ln[a:c].max() and w <= 41 and (w % 4 == 0 or w == 41)
    assert sum(b - a for a, b in chunk_bounds(32, 3)) == 32 and chunk_bounds(32, 3)[0][0] == 0


def test_chunked_head_forward_equals_the_full_batch_forward_and_the_loss_is_order_invariant():
    from semcom.common_space.training import chunk_bounds, chunk_widths, chunked_head_forward, sort_within_batches

    w4, adapter, head, data = _setup("text")
    cache = SymbolCache.build(w4, adapter, data, torch.device("cpu"))
    head.eval()
    n = len(data)
    lengths = cache.lengths_np
    perm = sort_within_batches(np.arange(n), lengths, n, 1)
    idx = torch.as_tensor(perm)
    ident = _Identity()
    full_rx, full_side = cache.get_dev(idx, int(lengths.max()))
    with torch.no_grad():
        ref = head(full_rx, **full_side)  # one padded pass
        got = chunked_head_forward(head, ident, 10.0, cache.sym, cache.lengths, idx, chunk_bounds(n, 3),
                                   chunk_widths(perm, lengths, n, 1, 3, cap=cache.sym.size(1))[0])
    assert torch.allclose(got, ref, atol=1e-5)
    anchors = torch.nn.functional.normalize(torch.randn(n, 512), dim=-1)
    loss = AnchorLoss(cos_weight=0.5)
    a = loss(got, anchors[idx])["total"]
    flip = torch.arange(n - 1, -1, -1)  # any reordering of the batch rows, with anchors reordered identically
    b = loss(got[flip], anchors[idx][flip])["total"]
    assert torch.allclose(a, b, atol=1e-5)


class _Identity:
    def __call__(self, x, snr):
        return x
