import pytest
import torch

from semcom.channel.awgn import AWGNChannel
from semcom.channel.rayleigh import RayleighChannel

CHANNELS = [AWGNChannel, RayleighChannel]


def _accelerator():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return None


@pytest.mark.parametrize("channel_cls", CHANNELS)
def test_gradient_flows_through_channel(channel_cls):
    x = torch.randn(4, 16, requires_grad=True)
    y = channel_cls()(x, 10.0)
    y.pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.skipif(_accelerator() is None, reason="no accelerator available")
@pytest.mark.parametrize("channel_cls", CHANNELS)
def test_runs_on_accelerator_and_stays_there(channel_cls):
    device = _accelerator()
    x = torch.randn(4, 16, device=device, requires_grad=True)
    y = channel_cls()(x, 10.0)
    assert y.device.type == device
    assert y.shape == x.shape
    y.pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
