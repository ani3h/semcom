import csv
import json
import pickle

import numpy as np
import pytest

from semcom.data.prompts import ESC50_TEMPLATES, all_prompts, prompt_sets
from semcom.data.text_captions import CaptionDataset
from semcom.encoders.text import task as text_task
from semcom.encoders.text.tokenizer import UNK_ID, WordTokenizer


def _fake_root(tmp_path, n_train=60, n_test=20):
    root = tmp_path / "data"
    (root / "captions").mkdir(parents=True)
    rows = lambda pfx, n: [{"text": f"{pfx} caption number {i} about a cat", "source": "coco" if i % 2 else "audiocaps", "item": f"{pfx}{i}"} for i in range(n)]  # noqa: E731
    (root / "captions" / "corpus.json").write_text(json.dumps({"train": rows("tr", n_train), "val": rows("va", 10), "test": rows("te", n_test)}))
    (root / "cifar-100-python").mkdir()
    pickle.dump({"fine_label_names": ["apple", "aquarium_fish", "baby"] + [f"thing{i}" for i in range(97)]}, open(root / "cifar-100-python" / "meta", "wb"))
    meta = root / "esc50" / "ESC-50-master" / "meta"
    meta.mkdir(parents=True)
    with open(meta / "esc50.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "fold", "target", "category"])
        w.writeheader()
        for t, c in enumerate(["dog", "sea_waves", "door_wood_creaks"] + [f"sound{i}" for i in range(47)]):
            w.writerow({"filename": f"{t}.wav", "fold": 1, "target": t, "category": c})
    return root


# ---------------------------------------------------------------- prompts + vocabulary union
def test_prompt_sets_cover_every_dataset_and_the_esc50_templates(tmp_path):
    sets = prompt_sets(_fake_root(tmp_path))
    assert len(sets["cifar10_photo"]) == 10 and len(sets["cifar100_photo"]) == 100 and len(sets["mnist_digit"]) == 10
    assert sets["mnist_digit"][3] == "a handwritten digit 3" and "a photo of a aquarium fish" in sets["cifar100_photo"]
    esc = [k for k in sets if k.startswith("esc50")]
    assert len(esc) == len(ESC50_TEMPLATES) and all(len(sets[k]) == 50 for k in esc)
    assert "This is a sound of sea waves." in sets[f"esc50 {ESC50_TEMPLATES[0]}"]  # underscores become spaces
    assert len(all_prompts(_fake_root(tmp_path / "again"))) == sum(len(v) for v in sets.values())


def test_vocabulary_is_the_union_of_frequent_caption_words_and_every_prompt_word():
    captions = ["a dog runs"] * 5 + ["the cat sits"] * 5 + ["rare word appears once"]
    tok = WordTokenizer.fit(captions, max_vocab=50, min_freq=3, required_texts=["a handwritten digit 7", "this is a sound of frogs."])
    assert tok.count_unk(["a handwritten digit 7", "this is a sound of frogs."]) == 0  # forced in although never seen in captions
    assert "dog" in tok.vocab and "cat" in tok.vocab  # frequent caption words stay
    assert "appears" not in tok.vocab  # a rare caption word below min_freq is still excluded
    assert tok.vocab.count("a") == 1  # no duplicates when a word is both forced and frequent


def test_required_words_survive_a_tight_vocabulary_cap_or_fail_loudly():
    captions = [f"word{i} filler" for i in range(100)] * 4
    tok = WordTokenizer.fit(captions, max_vocab=14, min_freq=1, required_texts=["handwritten digit"])
    assert tok.count_unk(["handwritten digit"]) == 0 and len(tok) <= 14
    with pytest.raises(ValueError):
        WordTokenizer.fit(captions, max_vocab=5, required_texts=["handwritten digit"])  # cap smaller than the required words


def test_text_task_fit_reports_the_vocabulary_and_asserts_zero_unk_on_every_prompt_set(tmp_path, monkeypatch):
    root = _fake_root(tmp_path)
    ds = CaptionDataset(str(root), split="train", prompt_repeat=2)
    task = text_task.TextTask(model={}, tokenizer={"max_vocab": 500, "min_freq": 3, "max_len": 40, "required_prompts": True, "prompt_root": str(root)})
    task.fit(ds)
    rep = task.vocab_report
    assert rep["vocab_size"] == len(task.tokenizer) and set(rep["unk_per_prompt_set"].values()) == {0}
    assert rep["n_training_sentences"] == len(ds) and rep["n_required_prompt_words"] > 10
    assert task.state_dict()["vocab_report"]["vocab_size"] == rep["vocab_size"]  # provenance travels with the checkpoint
    monkeypatch.setattr(WordTokenizer, "count_unk", lambda self, sents: 1)
    with pytest.raises(AssertionError, match="unk"):
        task.fit(ds)


# ---------------------------------------------------------------- caption splits
def test_caption_splits_are_disjoint_deterministic_and_heldout_stays_out_of_train(tmp_path):
    root = _fake_root(tmp_path)
    kw = dict(root=str(root), holdout=15, seed=3)
    tr, va, te = (CaptionDataset(split=s, **kw) for s in ("train", "val", "test"))
    assert (len(tr), len(va), len(te)) == (45, 15, 20)
    assert not (set(tr.texts) & set(va.texts))  # the monitoring hold-out is carved out of train, not shared with it
    assert all(t.startswith("te ") for t in te.texts) and all(t.startswith("tr ") for t in tr.texts + va.texts)
    assert not any(t.startswith("va ") for ds in (tr, va, te) for t in ds.texts)  # the corpus val split is never used
    assert CaptionDataset(split="train", **kw).texts == tr.texts  # seeded
    assert {s for s in te.sources[:10]} == {"coco", "audiocaps"}  # shuffled, so a prefix mixes both sources


def test_prompts_are_added_to_the_train_split_only(tmp_path):
    root = _fake_root(tmp_path)
    base = len(CaptionDataset(str(root), "train", holdout=15, prompt_repeat=0))
    tr = CaptionDataset(str(root), "train", holdout=15, prompt_repeat=3)
    n_prompts = len(all_prompts(root))
    assert len(tr) == base + 3 * n_prompts and tr.n_prompt_items == 3 * n_prompts and tr.sources.count("prompt") == 3 * n_prompts
    for split in ("val", "test"):
        ds = CaptionDataset(str(root), split, holdout=15, prompt_repeat=3)
        assert ds.n_prompt_items == 0 and "prompt" not in ds.sources
    with pytest.raises(ValueError):
        CaptionDataset(str(root), "heldout")


# ---------------------------------------------------------------- held-out paraphrase prompts (Stage 2 optimism check)
def _heldout_tokenizer(root, extra_words=()):
    from semcom.data.prompts import HELDOUT_TEMPLATES, class_names

    words = " ".join(t.replace("{c}", c) for task, ts in HELDOUT_TEMPLATES.items() for t in ts for c in class_names(task, root))
    return WordTokenizer.fit([words + " " + " ".join(extra_words)] * 3, max_vocab=2000, min_freq=1)


def test_heldout_templates_are_disjoint_from_the_training_templates_and_counted(tmp_path):
    from semcom.data.prompts import HELDOUT_TEMPLATES, TRAINING_TEMPLATES, template_prompts

    root = _fake_root(tmp_path)
    for task in HELDOUT_TEMPLATES:
        assert len(HELDOUT_TEMPLATES[task]) >= 3  # the protocol asks for at least 3 new templates per task
        assert not set(HELDOUT_TEMPLATES[task]) & set(TRAINING_TEMPLATES[task])
    held = template_prompts("cifar10", HELDOUT_TEMPLATES, root)
    assert len(held) == 3 * 10 and {p["class_idx"] for p in held} == set(range(10))
    assert "a picture of a cat" in [p["text"] for p in held]
    esc_train = template_prompts("esc50", TRAINING_TEMPLATES, root)  # the "{}" placeholder style of the training templates also formats
    assert len(esc_train) == 3 * 50 and "This is a sound of dog." in [p["text"] for p in esc_train]


def test_assert_heldout_clean_passes_on_clean_vocab_and_fires_on_unk_or_overlap(tmp_path):
    from semcom.data.prompts import assert_heldout_clean, class_names, template_prompts, HELDOUT_TEMPLATES

    root = _fake_root(tmp_path)
    tok = _heldout_tokenizer(root)
    report = assert_heldout_clean(tok, ["something unrelated entirely"], str(root))
    assert report["cifar10"]["unk"] == 0 and report["esc50"]["n_prompts"] == 150
    small = WordTokenizer.fit(["only a few words here"] * 3, max_vocab=50, min_freq=1)  # most held-out words are out of vocabulary
    with pytest.raises(AssertionError, match="<unk>"):
        assert_heldout_clean(small, [], str(root))
    leaked = [template_prompts("cifar10", HELDOUT_TEMPLATES, root)[0]["text"].upper() + "."]  # same sentence modulo case and trailing punctuation
    with pytest.raises(AssertionError, match="training data"):
        assert_heldout_clean(tok, leaked, str(root))


def test_heldout_template_hash_is_stable_and_tracks_edits(monkeypatch):
    from semcom.data import prompts

    h = prompts.heldout_hash()
    assert h == prompts.heldout_hash() and len(h) == 64
    edited = {**prompts.HELDOUT_TEMPLATES, "cifar10": prompts.HELDOUT_TEMPLATES["cifar10"] + ["a new template {c}"]}
    monkeypatch.setattr(prompts, "HELDOUT_TEMPLATES", edited)
    assert prompts.heldout_hash() != h  # any later edit of the templates is detectable against the protocol record


def test_real_heldout_templates_are_in_vocabulary_and_unseen_if_the_checkpoint_exists():
    """On the real data: zero <unk> and no verbatim training overlap (the assertion the protocol promised). Skipped when the data is absent."""
    from pathlib import Path

    import torch

    ckpt = Path("outputs/week4/caption_text/awgn/best.pt")
    if not ckpt.exists() or not Path("data_cache/captions/corpus.json").exists():
        pytest.skip("real caption data not available")
    from semcom.common_space.captions import load_corpus
    from semcom.data.prompts import all_prompts, assert_heldout_clean

    spec = torch.load(ckpt, weights_only=False)["spec"]
    tok = WordTokenizer(spec["vocab"], spec["max_len"])
    train = [r["text"] for r in load_corpus("data_cache/captions/corpus.json")["train"]] + all_prompts("data_cache")
    rep = assert_heldout_clean(tok, train, "data_cache")
    assert {k: v["unk"] for k, v in rep.items()} == {"cifar10": 0, "mnist": 0, "esc50": 0}
