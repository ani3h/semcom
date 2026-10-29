from pathlib import Path

import numpy as np

from ..common_space.captions import load_corpus
from .base import SemComDataset
from .prompts import all_prompts


class CaptionDataset(SemComDataset):
    """Caption-domain text (COCO + AudioCaps) from the Stage 0 corpus (data_cache/captions/corpus.json).

    Splits: train = corpus train minus a seeded hold-out, val = that hold-out (carved from TRAIN, used only to monitor
    training), test = the corpus test split. The corpus val and test splits are never used for vocabulary construction,
    transceiver training, head training or any checkpoint selection. Items are shuffled by a seeded permutation so that
    the first N of any split are a mix of both sources.

    prompt_repeat > 0 appends every class-prompt sentence that many times to the TRAIN split only. The transceiver needs
    this: many prompt words occur fewer than 3 times (or never) in the captions, and a vocabulary entry that is never
    trained cannot be transmitted. Heads and anchors use prompt_repeat=0 so the prompts never reach them.
    """

    modality = "text"
    task_type = "transmission"

    def __init__(self, root: str, split: str = "train", holdout: int = 5000, seed: int = 42, prompt_repeat: int = 0):
        super().__init__(split)
        corpus = load_corpus(Path(root) / "captions" / "corpus.json")
        train_rows = corpus["train"]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(train_rows))
        held, fit = perm[:holdout], perm[holdout:]
        if split == "train":
            self.corpus_split, self.corpus_indices = "train", fit
        elif split == "val":
            self.corpus_split, self.corpus_indices = "train", held
        elif split == "test":
            self.corpus_split, self.corpus_indices = "test", rng.permutation(len(corpus["test"]))
        else:
            raise ValueError(f"unknown split {split!r}; use train/val/test")
        rows = corpus[self.corpus_split]
        self.texts = [rows[i]["text"] for i in self.corpus_indices]
        self.sources = [rows[i]["source"] for i in self.corpus_indices]
        self.n_prompt_items = 0
        if split == "train" and prompt_repeat > 0:
            prompts = all_prompts(root) * prompt_repeat
            self.texts += prompts
            self.sources += ["prompt"] * len(prompts)
            self.n_prompt_items = len(prompts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {"input": self.texts[idx], "label": None, "raw": self.sources[idx]}
