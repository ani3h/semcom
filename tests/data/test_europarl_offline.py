import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import semcom.data.text_europarl as te
from semcom.data.text_europarl import EuroparlDataset

PAIRS = [
    ("Resumption of the session", "Wiederaufnahme der Sitzungsperiode"),
    ("Thank you very much Mr President", "Vielen Dank Herr Präsident"),
    ("Thank you very much Mr President", "Vielen Dank Herr Präsident"),  # exact duplicate
    ("Yes", "Ja"),  # too short
    ("The Commission has presented its proposal to Parliament today", "Die Kommission hat heute ihren Vorschlag vorgelegt"),
    ("We must act now to protect the environment of all our citizens", "Wir müssen jetzt handeln um die Umwelt zu schützen"),
    ("This is another perfectly ordinary sentence about agriculture", "Dies ist ein weiterer ganz gewöhnlicher Satz"),
    ("Please   check   the   spacing   here   as well", "Bitte prüfen Sie die Abstände hier auch"),
] * 5  # repeat so there are enough unique-after-dedupe rows to split; duplicates are removed anyway
GERMAN_ONLY = {de for _, de in PAIRS}


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    shard = tmp_path / "train-00000-of-00001.parquet"
    table = pa.table({"translation": [{"de": de, "en": en} for en, de in PAIRS]})
    pq.write_table(table, shard)

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type=None):
            return ["de-en/train-00000-of-00001.parquet", "README.md"]

    monkeypatch.setattr(te, "HfApi", FakeApi)
    monkeypatch.setattr(te, "hf_hub_download", lambda *a, **k: str(shard))
    return tmp_path


def _all_items(root, **kw):
    kwargs = dict(root=str(root), lang="en", pair_lang="de", max_samples=100, val_fraction=0.2, test_fraction=0.2, seed=1)
    kwargs.update(kw)
    return {s: EuroparlDataset(split=s, **kwargs) for s in ("train", "val", "test")}


def test_only_english_text_reaches_items(fake_hub):
    sets = _all_items(fake_hub / "root")
    texts = [ds[i]["input"] for ds in sets.values() for i in range(len(ds))]
    assert texts, "expected some sentences"
    assert not (set(texts) & GERMAN_ONLY)
    assert all(ds[0]["raw"] is None for ds in sets.values())  # German row never exposed


def test_filters_dedupes_and_normalizes_whitespace(fake_hub):
    sets = _all_items(fake_hub / "root")
    texts = [ds[i]["input"] for ds in sets.values() for i in range(len(ds))]
    assert len(texts) == len(set(texts))  # deduplicated -> no train/test leakage
    assert "Yes" not in texts  # shorter than min_words
    assert "Please check the spacing here as well" in texts
    assert all(4 <= len(t.split()) <= 30 for t in texts)


def test_splits_are_disjoint_and_cover_everything(fake_hub):
    sets = _all_items(fake_hub / "root")
    per_split = [{ds[i]["input"] for i in range(len(ds))} for ds in sets.values()]
    assert sum(len(s) for s in per_split) == len(set().union(*per_split)) == 6  # 6 unique sentences survive filtering
    assert per_split[0].isdisjoint(per_split[1]) and per_split[0].isdisjoint(per_split[2])


def test_is_deterministic_for_same_seed(fake_hub):
    a = _all_items(fake_hub / "root_a")["train"]
    b = _all_items(fake_hub / "root_b")["train"]
    assert [a[i]["input"] for i in range(len(a))] == [b[i]["input"] for i in range(len(b))]
