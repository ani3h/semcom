import torch

from semcom.channel.awgn import AWGNChannel


def test_output_shape_matches_input():
    ch = AWGNChannel()
    x = torch.randn(4, 16)
    y = ch(x, snr_db=10.0)
    assert y.shape == x.shape


def test_mse_decreases_monotonically_as_snr_increases():
    ch = AWGNChannel()
    x = torch.randn(8, 64)
    snrs = [0, 5, 10, 15, 20, 25]

    mses = []
    for snr in snrs:
        torch.manual_seed(1)
        y = ch(x, float(snr))
        mses.append(torch.nn.functional.mse_loss(y, x).item())

    assert all(mses[i] >= mses[i + 1] for i in range(len(mses) - 1))


def test_odd_last_dim_raises():
    ch = AWGNChannel()
    x = torch.randn(4, 15)
    try:
        ch(x, 10.0)
        assert False, "expected ValueError for odd last dim"
    except ValueError:
        pass
