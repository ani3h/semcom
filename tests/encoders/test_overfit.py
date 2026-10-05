"""Can each model actually learn? Overfit a handful of samples with the channel in the loop
(AWGN at a moderate SNR) before committing to a full training run."""
import torch

from semcom.channel.awgn import AWGNChannel

from .builders import audio_dataset, audio_task, cifar_dataset, cifar_task, mnist_dataset, mnist_task
from .builders import overfit, text_dataset, text_task

CHANNEL = AWGNChannel()


def test_text_overfits_and_recovers_sentences():
    task, ds = text_task(), text_dataset()
    model, first, last = overfit(task, ds, CHANNEL, snr_db=15.0, steps=250, lr=3e-3)
    assert last < 0.2 * first
    res = task.evaluate(model, ds, CHANNEL, 15.0, torch.device("cpu"))
    assert res["bleu1"] > 0.9
    assert res["exact_match"] >= 0.75


def test_cifar_reconstruction_overfits():
    task, ds = cifar_task(), cifar_dataset()
    model, first, last = overfit(task, ds, CHANNEL, snr_db=20.0, steps=300, lr=2e-3)
    assert last < 0.5 * first
    res = task.evaluate(model, ds, CHANNEL, 20.0, torch.device("cpu"))
    assert res["psnr"] > 18.0


def test_mnist_classifier_overfits():
    task, ds = mnist_task(), mnist_dataset()
    model, first, last = overfit(task, ds, CHANNEL, snr_db=20.0, steps=200, lr=3e-3)
    assert last < 0.2 * first
    assert task.evaluate(model, ds, CHANNEL, 20.0, torch.device("cpu"))["accuracy"] == 1.0


def test_audio_reconstruction_overfits():
    task, ds = audio_task(), audio_dataset()
    model, first, last = overfit(task, ds, CHANNEL, snr_db=20.0, steps=200, lr=2e-3)
    assert last < 0.5 * first
    assert task.evaluate(model, ds, CHANNEL, 20.0, torch.device("cpu"))["si_sdr"] > 5.0
