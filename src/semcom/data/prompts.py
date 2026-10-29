"""Every class-prompt set used by the cross-modal tests, built from the class names on disk.

These are the query templates (not held-out data). The caption-domain text transceiver builds its vocabulary from the union
of these and the caption train split and asserts that every set tokenizes without <unk>.
"""
import csv
import pickle
from pathlib import Path

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
ESC50_TEMPLATES = ("This is a sound of {}.", "{}", "the sound of {}")  # the pre-registered template first, then the variants


def cifar100_classes(root) -> list[str]:
    meta = pickle.load(open(Path(root) / "cifar-100-python" / "meta", "rb"), encoding="latin1")
    return [c.replace("_", " ") for c in meta["fine_label_names"]]


def esc50_classes(root) -> list[str]:
    rows = csv.DictReader(open(Path(root) / "esc50" / "ESC-50-master" / "meta" / "esc50.csv"))
    return [c.replace("_", " ") for _, c in sorted({(int(r["target"]), r["category"]) for r in rows})]


def prompt_sets(root="data_cache") -> dict[str, list[str]]:
    esc = esc50_classes(root)
    sets = {
        "cifar10_photo": [f"a photo of a {c}" for c in CIFAR10_CLASSES],
        "cifar100_photo": [f"a photo of a {c}" for c in cifar100_classes(root)],
        "mnist_digit": [f"a handwritten digit {n}" for n in range(10)],
        "cifar10_sound": [f"This is a sound of {c}." for c in CIFAR10_CLASSES],
    }
    for t in ESC50_TEMPLATES:
        sets[f"esc50 {t}"] = [t.format(c) for c in esc]
    return sets


def all_prompts(root="data_cache") -> list[str]:
    return [p for ps in prompt_sets(root).values() for p in ps]


# ---------------------------------------------------------------------------------------------------------------------------
# Stage 2 text-prompt optimism check. The 280 prompt sentences above were oversampled into the text transceiver's training data,
# so results on them are optimistic. These paraphrase templates were written ONCE, up front (2026-10-24, before any Stage 2 seed was
# trained or evaluated), after checking only two things: every word is already in the caption transceiver's vocabulary, and no
# generated sentence occurs verbatim in the caption training data. They were never used for vocabulary construction, transceiver
# training or head training. Do not edit them after results exist; the protocol record stores their hash.
TRAINING_TEMPLATES = {  # the templates present in the transceiver's training-time prompt sentences
    "cifar10": ["a photo of a {c}"],
    "mnist": ["a handwritten digit {c}"],
    "esc50": list(ESC50_TEMPLATES),
}
HELDOUT_TEMPLATES = {
    "cifar10": ["a picture of a {c}", "an image of a {c}", "there is a {c} in this photo"],
    "mnist": ["a handwritten number {c}", "the digit {c} written by hand", "an image of the number {c}"],
    "esc50": ["a recording of {c}", "you can hear {c}", "audio of {c} playing"],
}


def class_names(task: str, root="data_cache") -> list[str]:
    return {"cifar10": CIFAR10_CLASSES, "mnist": [str(n) for n in range(10)], "esc50": esc50_classes(root)}[task]


def template_prompts(task: str, templates: dict[str, list[str]], root="data_cache") -> list[dict]:
    """[{text, class_idx, template}] for every (class, template) of `task`."""
    return [{"text": t.replace("{}", "{c}").format(c=c), "class_idx": i, "template": t} for t in templates[task] for i, c in enumerate(class_names(task, root))]


def heldout_hash() -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(HELDOUT_TEMPLATES, sort_keys=True).encode()).hexdigest()


def assert_heldout_clean(tokenizer, training_sentences, root="data_cache") -> dict:
    """Every held-out prompt tokenizes with zero <unk> and is not a training sentence (compared without case or trailing
    punctuation). `training_sentences` is every text the transceiver or heads were trained on (captions + training prompts)."""
    def norm(s):
        from ..encoders.text.tokenizer import split_tokens

        t = split_tokens(s)
        while t and t[-1] in {".", "!", "?", ","}:
            t = t[:-1]
        return " ".join(t)

    seen = {norm(s) for s in training_sentences}
    report = {}
    for task in HELDOUT_TEMPLATES:
        prompts = [p["text"] for p in template_prompts(task, HELDOUT_TEMPLATES, root)]
        unk = tokenizer.count_unk(prompts)
        overlap = [p for p in prompts if norm(p) in seen]
        assert unk == 0, f"held-out prompts for {task} contain {unk} <unk> tokens"
        assert not overlap, f"held-out prompts for {task} occur in the training data: {overlap[:3]}"
        report[task] = {"n_prompts": len(prompts), "unk": unk, "verbatim_in_training_data": 0}
    return report
