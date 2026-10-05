import pytest
import torch

from semcom.channel.awgn import AWGNChannel
from semcom.channel.rayleigh import RayleighChannel
from semcom.encoders.base import Transceiver
from semcom.encoders.text.tokenizer import EOS_ID, PAD_ID

from .builders import TASKS

CHANNELS = [AWGNChannel, RayleighChannel]


def _batch_and_args(name):
    """(model, input tensor, side-info kwargs for decode/forward) for one tiny modality."""
    task_fn, ds_fn = TASKS[name]
    task, ds = task_fn(), ds_fn()
    batch = task.collate([ds[i] for i in range(len(ds))])
    model = task.build_model()
    if name == "text":
        return model, batch["src"], dict(src_mask=batch["src"] != PAD_ID, tgt_in=batch["tgt_in"]), batch
    return model, batch["x"], {}, batch


@pytest.mark.parametrize("name", TASKS)
def test_is_transceiver_with_even_symbol_dim(name):
    model, x, _, _ = _batch_and_args(name)
    assert isinstance(model, Transceiver)
    symbols = model.encode(x)
    assert symbols.shape[-1] % 2 == 0
    assert symbols.shape[0] == x.shape[0]


def test_output_shapes():
    model, x, side, batch = _batch_and_args("text")
    logits = model(x, AWGNChannel(), 10.0, **side)
    assert logits.shape == (*batch["tgt_in"].shape, model.vocab_size)
    ids = model(x, AWGNChannel(), 10.0, src_mask=side["src_mask"])
    assert ids.dim() == 2 and ids.shape[0] == x.shape[0] and ids.shape[1] <= x.shape[1]

    model, x, side, _ = _batch_and_args("cifar10")
    assert model(x, AWGNChannel(), 10.0).shape == x.shape
    assert model.encode(x).shape == (x.shape[0], 1024)  # 512 complex symbols = 1/6 of 3072 values

    model, x, side, _ = _batch_and_args("mnist")
    assert model(x, AWGNChannel(), 10.0).shape == (x.shape[0], 10)
    assert model.encode(x).shape == (x.shape[0], 256)

    model, x, side, _ = _batch_and_args("audio")
    assert model(x, AWGNChannel(), 10.0).shape == x.shape


def test_text_padding_rows_are_constant_and_nonzero():
    model, src, _, _ = _batch_and_args("text")
    model.eval()
    sym = model.encode(src)
    pad_rows = sym[src == PAD_ID]
    assert pad_rows.numel() > 0
    assert torch.equal(pad_rows, torch.ones_like(pad_rows))  # constant row, never zero power


def test_text_greedy_decoding_stops_at_eos_with_pad_after():
    model, src, side, _ = _batch_and_args("text")
    model.eval()
    ids = model(src, AWGNChannel(), 10.0, src_mask=side["src_mask"])
    for row in ids:
        eos = (row == EOS_ID).nonzero()
        if len(eos):
            assert (row[eos[0, 0] + 1 :] == PAD_ID).all()


@pytest.mark.parametrize("channel_cls", CHANNELS)
@pytest.mark.parametrize("name", TASKS)
def test_gradient_reaches_encoder_through_channel(name, channel_cls):
    """Channel-in-the-loop training only works if the loss gradient crosses the channel."""
    task_fn, ds_fn = TASKS[name]
    task, ds = task_fn(), ds_fn()
    model = task.build_model()
    batch = task.collate([ds[i] for i in range(len(ds))])
    loss = task.loss(model, batch, channel_cls(), 10.0)
    assert torch.isfinite(loss)
    loss.backward()
    encoder_params = [p for n, p in model.named_parameters() if n.startswith(("enc", "src_embed", "channel_encoder"))]
    assert encoder_params
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder_params)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
