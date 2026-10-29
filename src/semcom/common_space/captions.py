"""Caption corpus for the CLAP->CLIP text bridge (and, later, the caption-domain text transceiver).

Sources, both text-only and already split by item (image / audio clip), so no item appears in two splits:
  COCO (yerevann/coco-karpathy): Karpathy train+restval / validation / test, 5 captions per image.
  AudioCaps (d0rj/audiocaps)   : train / validation / test, one caption per row.
Within a split captions are lower-cased for comparison and de-duplicated; any validation/test caption whose text
also occurs in train is dropped, so held-out agreement cannot come from memorised duplicates. The held-out
test split must stay out of vocabulary construction and training of anything downstream.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

COCO = ("yerevann/coco-karpathy", {"train": ["data/train-00000-of-00001.parquet", "data/restval-00000-of-00001.parquet"],
                                   "val": ["data/validation-00000-of-00001.parquet"], "test": ["data/test-00000-of-00001.parquet"]})
AUDIOCAPS = ("d0rj/audiocaps", {"train": ["data/train-00000-of-00001-dfc9e682a70b4097.parquet"],
                                "val": ["data/validation-00000-of-00001-56ae1ba28294095a.parquet"],
                                "test": ["data/test-00000-of-00001-069405200b9b7a9f.parquet"]})


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _key(text: str) -> str:
    return text.lower().rstrip(" .")


def _read(root: Path, repo: str, files: list[str]) -> pd.DataFrame:
    base = Path(root) / repo.replace("/", "__")
    return pd.concat([pd.read_parquet(base / f) for f in files], ignore_index=True)


def load_raw(root, source: str, split: str) -> list[dict]:
    """All captions of one source/split as {text, source, item}. COCO: every one of the 5 captions per image."""
    repo, files = COCO if source == "coco" else AUDIOCAPS
    df = _read(root, repo, files[split])
    if source == "coco":
        return [{"text": s, "source": "coco", "item": f"coco:{r.cocoid}"} for r in df.itertuples() for s in r.sentences]
    return [{"text": r.caption, "source": "audiocaps", "item": f"ac:{r.audiocap_id}"} for r in df.itertuples()]


def _filter(rows: list[dict], min_words: int, seen: set[str] | None = None) -> list[dict]:
    out, local = [], set()
    for r in rows:
        t = clean(r["text"])
        k = _key(t)
        if len(t.split()) < min_words or k in local or (seen is not None and k in seen):
            continue
        local.add(k)
        out.append({**r, "text": t})
    return out


def build_corpus(root, seed: int = 42, n_coco_train: int = 150000, n_coco_eval: int = 5000, min_words: int = 3) -> dict:
    """{split: [{text, source, item}]} for split in train/val/test. All AudioCaps captions are kept; COCO is
    sub-sampled (seeded) to n_coco_train / n_coco_eval captions so the audio-style captions are not drowned out."""
    rng = np.random.default_rng(seed)
    corpus, train_keys = {}, set()
    for split, n_coco in (("train", n_coco_train), ("val", n_coco_eval), ("test", n_coco_eval)):
        coco = _filter(load_raw(root, "coco", split), min_words, train_keys if split != "train" else None)
        if len(coco) > n_coco:
            coco = [coco[i] for i in np.sort(rng.choice(len(coco), size=n_coco, replace=False))]
        ac = _filter(load_raw(root, "audiocaps", split), min_words, train_keys if split != "train" else None)
        # a caption present in both sources keeps its first occurrence only
        merged = _filter(coco + ac, min_words)
        corpus[split] = merged
        if split == "train":
            train_keys = {_key(r["text"]) for r in merged}
    return corpus


def save_corpus(corpus: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(corpus))


def load_corpus(path) -> dict:
    return json.loads(Path(path).read_text())
