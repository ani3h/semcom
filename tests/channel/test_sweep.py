import torch

from semcom.channel.awgn import AWGNChannel
from semcom.channel.sweep import snr_sweep


def test_sweep_covers_full_range_and_shapes():
    ch = AWGNChannel()
    x = torch.randn(4, 8)
    results = snr_sweep(ch, x, snr_range_db=(0, 5), step_db=1.0)

    assert set(results.keys()) == {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}
    for y in results.values():
        assert y.shape == x.shape
