import re
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from .base import SemComDataset
from .splits import train_val_test_indices

REPO_ID = "Helsinki-NLP/europarl"


def _load_sentences(
    root: Path, lang: str, pair_lang: str, max_samples: int, min_words: int, max_words: int, seed: int
) -> list[str]:
    """Single-language sentences from the Europarl parallel corpus.

    Only the `lang` column is ever read: `pair_lang` just selects which
    parallel config ("de-en") to download, so no other language reaches the
    pipeline (DeepSC-style monolingual sentence recovery, not translation).
    Filtered to [min_words, max_words] words, exact-duplicate sentences removed
    (avoids train/test leakage), then a seeded random sample of max_samples,
    cached as one sentence per line.
    """
    config = "-".join(sorted([lang, pair_lang]))
    cache_file = root / "europarl" / f"{lang}_from_{config}_n{max_samples}_w{min_words}-{max_words}_s{seed}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8").split("\n")

    shards = sorted(
        f for f in HfApi().list_repo_files(REPO_ID, repo_type="dataset")
        if f.startswith(f"{config}/") and f.endswith(".parquet")
    )
    if not shards:
        raise RuntimeError(f"No parquet shards found for config {config!r} in {REPO_ID}")

    sentences: list[str] = []
    for shard in shards:
        path = hf_hub_download(REPO_ID, shard, repo_type="dataset", cache_dir=root / "hf_cache")
        col = pq.read_table(path, columns=["translation"]).column("translation")
        sentences.extend(pc.struct_field(col, [lang]).to_pylist())

    cleaned = (re.sub(r"\s+", " ", s).strip() for s in sentences)
    unique = list(dict.fromkeys(s for s in cleaned if min_words <= len(s.split()) <= max_words))
    if max_samples is not None and max_samples < len(unique):
        keep = np.sort(np.random.default_rng(seed).choice(len(unique), size=max_samples, replace=False))
        unique = [unique[i] for i in keep]

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("\n".join(unique), encoding="utf-8")
    return unique


class EuroparlDataset(SemComDataset):
    modality = "text"
    task_type = "transmission"

    def __init__(
        self,
        root: str,
        split: str = "train",
        lang: str = "en",
        pair_lang: str = "de",
        max_samples: int = 100000,
        min_words: int = 4,
        max_words: int = 30,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
    ):
        super().__init__(split)
        self.lang = lang
        root = Path(root)
        self.sentences = _load_sentences(root, lang, pair_lang, max_samples, min_words, max_words, seed)

        cache_path = root / "europarl" / f"splits_{lang}_{pair_lang}_n{max_samples}_s{seed}.json"
        train_idx, val_idx, test_idx = train_val_test_indices(
            len(self.sentences), val_fraction, test_fraction, seed, cache_path
        )
        indices_by_split = {"train": train_idx, "val": val_idx, "test": test_idx}
        self._indices = indices_by_split[split]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        return {"input": self.sentences[int(self._indices[idx])], "label": None, "raw": None}
