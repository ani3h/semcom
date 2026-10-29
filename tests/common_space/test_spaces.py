import pytest
import torch

from encoders.builders import TASKS  # tiny CPU-sized Week 4 transceivers + synthetic data (tests/encoders)
from semcom.channel.awgn import AWGNChannel
from semcom.channel.base import Channel
from semcom.channel.rayleigh import RayleighChannel
from semcom.common_space.adapters import make_adapter
from semcom.common_space.base import EMBED_DIM
from semcom.common_space.direct import DirectPretrainedSpace
from semcom.common_space.heads import SymbolConditioner, build_head
from semcom.common_space.learned import LearnedHeadSpace
from semcom.common_space.losses import AnchorLoss
from semcom.common_space.retrieval import recall_at_k

from .stubs import StubAnchors

HEAD_CFG = {
    "cifar10": ("mlp", dict(in_dim=1024, hidden=64)),
    "mnist": ("mlp", dict(in_dim=256, hidden=64)),
    "text": ("text_attention", dict(sym_dim=16, d_model=32, nhead=4, ff=64)),
    "audio": ("audio_conv", dict(frame_dim=32, hidden=32)),
}
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


class Identity(Channel):
    def forward(self, x, snr_db):
        return x


def build(name, device="cpu"):
    task_fn, ds_fn = TASKS[name]
    task, ds = task_fn(), ds_fn()
    model = task.build_model().to(device)
    adapter = make_adapter(name, StubAnchors(text_device=device), task)
    kind, kw = HEAD_CFG[name]
    head = build_head(kind, sigma=1.0, **kw).to(device)
    batch = task.to_device(task.collate([ds[i] for i in range(len(ds))]), device)
    return task, model, adapter, head, batch


@pytest.mark.parametrize("channel_cls", [AWGNChannel, RayleighChannel])
@pytest.mark.parametrize("name", list(TASKS))
def test_learned_space_dim_and_unit_norm(name, channel_cls):
    _, model, adapter, head, batch = build(name)
    space = LearnedHeadSpace(model, head.eval(), adapter)
    for emb in (space.embed_source(batch), space.embed_received(batch, channel_cls(), 10.0)):
        assert emb.shape == (next(iter(v for v in batch.values() if torch.is_tensor(v))).size(0), EMBED_DIM)
        assert torch.allclose(emb.norm(dim=-1), torch.ones(emb.size(0)), atol=1e-4)


@pytest.mark.parametrize("name", ["text", "cifar10", "audio"])
def test_direct_space_dim_and_unit_norm(name):
    _, model, adapter, _, batch = build(name)
    space = DirectPretrainedSpace(model, adapter)
    emb = space.embed_received(batch, AWGNChannel(), 10.0)
    assert emb.shape[1] == EMBED_DIM
    assert torch.allclose(emb.norm(dim=-1), torch.ones(emb.size(0)), atol=1e-4)


def test_direct_space_rejects_mnist_explicitly():
    _, model, adapter, _, _ = build("mnist")
    with pytest.raises(NotImplementedError, match="class logits"):
        DirectPretrainedSpace(model, adapter)


@pytest.mark.parametrize("name", list(TASKS))
def test_gradients_reach_only_the_head(name):
    _, model, adapter, head, batch = build(name)
    space = LearnedHeadSpace(model, head, adapter)
    anchors = space.embed_source(batch)
    loss = AnchorLoss()(space.embed_received(batch, AWGNChannel(), 10.0), anchors)["total"]
    loss.backward()
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert not anchors.requires_grad
    head_grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert head_grads and all(torch.isfinite(g).all() for g in head_grads)
    assert any(g.abs().sum() > 0 for g in head_grads)


def test_contrastive_loss_overfits_a_small_batch():
    _, model, adapter, head, batch = build("cifar10")
    space = LearnedHeadSpace(model, head, adapter)
    anchors = space.embed_source(batch)
    loss_fn = AnchorLoss()
    opt = torch.optim.Adam(list(head.parameters()) + list(loss_fn.parameters()), lr=3e-3)
    torch.manual_seed(0)
    first = last = None
    for _ in range(300):
        out = loss_fn(space.embed_received(batch, AWGNChannel(), 20.0), anchors)
        opt.zero_grad(); out["total"].backward(); opt.step()
        first = out["total"].item() if first is None else first
        last = out["total"].item()
    assert last < 0.2 * first
    head.eval()
    assert recall_at_k(space.embed_received(batch, Identity(), 20.0), anchors, ks=(1,))["recall@1"] == 1.0


def test_text_head_ignores_padding_length():
    task, model, adapter, head, _ = build("text")
    space = LearnedHeadSpace(model, head.eval(), adapter)
    _, ds_fn = TASKS["text"]
    ds = ds_fn()
    alone = task.collate([ds[0]])
    with_longer = task.collate([ds[0], ds[3]])  # ds[3] is longer, so ds[0] gets extra padding
    assert with_longer["src"].size(1) > alone["src"].size(1)
    a = space.embed_received(alone, Identity(), 10.0)[0]
    b = space.embed_received(with_longer, Identity(), 10.0)[0]
    assert torch.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("name", list(TASKS))
def test_whole_path_runs_on_available_device(name):
    _, model, adapter, head, batch = build(name, device=DEVICE)
    a = LearnedHeadSpace(model, head.eval(), adapter)
    for emb in (a.embed_source(batch), a.embed_received(batch, AWGNChannel(), 10.0),
                a.embed_received(batch, RayleighChannel(), 0.0)):
        assert emb.device.type == DEVICE.type and torch.isfinite(emb).all()
    if adapter.supports_direct:
        b = DirectPretrainedSpace(model, adapter).embed_received(batch, RayleighChannel(), 0.0)
        assert b.device.type == DEVICE.type and torch.isfinite(b).all()


def test_symbol_conditioner_is_bounded_and_near_identity_for_small_inputs():
    cond = SymbolConditioner(sigma=2.0, clip=4.0)
    x = torch.tensor([0.0, 0.1, -0.1, 1e4, -1e4])
    y = cond(x)
    assert y.abs().max() <= 4.0 + 1e-6
    assert torch.allclose(y[:3], x[:3] / 2.0, atol=1e-3)  # divides by sigma, ~linear near zero
