import torch

from semcom.channel.awgn import AWGNChannel
from semcom.channel.rayleigh import RayleighChannel


def test_output_shape_matches_input():
    ch = RayleighChannel()
    x = torch.randn(4, 16)
    y = ch(x, snr_db=10.0)
    assert y.shape == x.shape


def test_mse_decreases_monotonically_as_snr_increases():
    ch = RayleighChannel()
    x = torch.randn(8, 64)
    snrs = [0, 5, 10, 15, 20, 25]

    mses = []
    for snr in snrs:
        torch.manual_seed(1)
        y = ch(x, float(snr))
        mses.append(torch.nn.functional.mse_loss(y, x).item())

    assert all(mses[i] >= mses[i + 1] for i in range(len(mses) - 1))


def test_rayleigh_differs_from_awgn_at_same_snr():
    x = torch.randn(16, 64)

    torch.manual_seed(1)
    y_awgn = AWGNChannel()(x, 15.0)
    torch.manual_seed(1)
    y_rayleigh = RayleighChannel()(x, 15.0)

    assert not torch.allclose(y_awgn, y_rayleigh)
    # Rayleigh's multiplicative fading adds distortion beyond AWGN's purely additive noise.
    assert (y_rayleigh - x).var() > (y_awgn - x).var()
