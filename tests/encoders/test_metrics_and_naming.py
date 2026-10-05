import pytest
import torch

from semcom.encoders.audio.metrics import si_sdr
from semcom.encoders.image.task import psnr
from semcom.encoders.text.task import bleu
from semcom.encoders.text.tokenizer import WordTokenizer, split_tokens
from semcom.tracking.run_naming import make_run_name


def test_bleu_identical_and_disjoint():
    refs = ["the cat sat on the mat", "a dog barks loudly at night"]
    assert bleu(refs, refs, 4) == pytest.approx(1.0)
    assert bleu(["x y z w q r", "p o i u y t"], refs, 1) == 0.0


def test_psnr_orders_by_noise_level():
    x = torch.rand(4, 3, 8, 8)
    assert (psnr(x + 0.01 * torch.randn_like(x), x) > psnr(x + 0.1 * torch.randn_like(x), x)).all()
    assert torch.isfinite(psnr(x, x)).all()


def test_si_sdr_is_scale_invariant_and_penalizes_noise():
    ref = torch.randn(2, 4000)
    est = ref + 0.3 * torch.randn_like(ref)
    assert torch.allclose(si_sdr(3.0 * est, ref), si_sdr(0.1 * est, ref), atol=1e-2)
    assert (si_sdr(ref + 0.01 * torch.randn_like(ref), ref) > si_sdr(ref + 0.5 * torch.randn_like(ref), ref)).all()


def test_tokenizer_splits_punctuation_and_marks_unknowns():
    assert split_tokens("Hello, World!") == ["hello", ",", "world", "!"]
    tok = WordTokenizer.fit(["a b c a b a"], max_vocab=50, min_freq=2)
    assert "c" not in tok.vocab  # below min_freq
    batch = tok.make_batch(["a b c"])
    assert batch["src"].shape == batch["tgt_in"].shape == batch["tgt_out"].shape
    assert tok.ids_to_tokens(batch["tgt_out"][0]) == ["a", "b", "<unk>"]


def test_tokenizer_truncates_to_max_len():
    tok = WordTokenizer.fit(["a b"], min_freq=1, max_len=3)
    assert len(tok.encode("a a a a a a")) == 3


def test_run_name_accepts_number_or_range_label():
    assert "-snr10-" in make_run_name("image", "cifar10", "awgn", 10.0, 42)
    name = make_run_name("text", "europarl", "rayleigh", "U0-25", 7)
    assert name.startswith("text-europarl-rayleigh-snrU0-25-seed7-")
