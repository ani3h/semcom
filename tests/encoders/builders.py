"""Tiny CPU-sized models + synthetic data for each modality (no downloads)."""
import math

import torch

from semcom.encoders.audio.model import SEAudioTransceiver
from semcom.encoders.audio.task import AudioReconTask
from semcom.encoders.image.model import JSCCClassifier, JSCCImage
from semcom.encoders.image.task import CIFARReconTask, MNISTClassTask
from semcom.encoders.text.model import TextTransceiver
from semcom.encoders.text.task import TextTask

SENTENCES = [
    "the commission has presented its proposal today",
    "we must act now to protect the environment",
    "thank you very much mr president",
    "this is a perfectly ordinary sentence about agriculture",
    "the committee will vote on the report tomorrow",
    "member states should respect the common rules",
    "i would like to thank the rapporteur for her work",
    "the debate is closed and the vote will take place",
]
AUDIO_LEN = 2560  # 20 frames of 128 samples


class ListDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def text_task() -> TextTask:
    task = TextTask(
        model=dict(d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0, symbols_per_token=16),
        tokenizer=dict(max_vocab=200, min_freq=1, max_len=20),
        eval=dict(max_items=None, batch_size=8),
    )
    task.fit(ListDataset([{"input": s} for s in SENTENCES]))
    return task


def text_dataset():
    return ListDataset([{"input": s, "label": None, "raw": None} for s in SENTENCES])


def cifar_task() -> CIFARReconTask:
    return CIFARReconTask(model=dict(channels=16, width=16), eval=dict(batch_size=8))


def cifar_dataset(n=8):
    g = torch.Generator().manual_seed(0)
    low = torch.rand(n, 3, 4, 4, generator=g)  # smooth-ish images: upsampled low-res noise
    imgs = torch.nn.functional.interpolate(low, size=32, mode="bilinear", align_corners=False)
    return ListDataset([{"input": im, "label": torch.tensor(0), "raw": None} for im in imgs])


def mnist_task() -> MNISTClassTask:
    return MNISTClassTask(model=dict(channels=16), eval=dict(batch_size=16))


def mnist_dataset(n=16):
    g = torch.Generator().manual_seed(0)
    return ListDataset([
        {"input": torch.rand(1, 28, 28, generator=g), "label": torch.tensor(i % 10), "raw": None} for i in range(n)
    ])


def audio_task() -> AudioReconTask:
    task = AudioReconTask(model=dict(frame_len=128, channels=16, num_blocks=1, symbols_per_frame=32),
                          eval=dict(max_items=4, batch_size=4, pesq=False))
    task.fit(audio_dataset())
    return task


def audio_dataset(n=4):
    t = torch.arange(AUDIO_LEN) / 16000
    clips = [0.1 * torch.sin(2 * math.pi * f * t) + 0.05 * torch.sin(2 * math.pi * 2.3 * f * t) for f in (180, 260, 340, 420)[:n]]
    return ListDataset([{"input": c, "label": None, "raw": None} for c in clips])


# name -> (task factory, dataset factory)
TASKS = {
    "text": (text_task, text_dataset),
    "cifar10": (cifar_task, cifar_dataset),
    "mnist": (mnist_task, mnist_dataset),
    "audio": (audio_task, audio_dataset),
}


def overfit(task, dataset, channel, snr_db, steps, lr):
    """Train a fresh model on the whole (tiny) dataset as one batch; return (model, first_loss, last_loss)."""
    torch.manual_seed(0)
    model = task.build_model()
    batch = task.collate([dataset[i] for i in range(len(dataset))])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    first = last = None
    for _ in range(steps):
        model.train()
        loss = task.loss(model, batch, channel, snr_db)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    return model, first, last
