import torch
import torch.nn.functional as F
from sacrebleu.metrics import BLEU

from ..base import TaskSpec
from .model import TextTransceiver
from .tokenizer import PAD_ID, WordTokenizer, split_tokens


def bleu(hyps: list[str], refs: list[str], n: int) -> float:
    """Corpus BLEU-n in [0, 1] on pre-tokenized (space-joined) text."""
    metric = BLEU(tokenize="none", max_ngram_order=n, smooth_method="none", effective_order=False)
    return metric.corpus_score(hyps, [refs]).score / 100.0


def token_accuracy(hyps: list[str], refs: list[str]) -> float:
    """Position-wise token accuracy over all reference tokens (a missing or shifted token counts as wrong)."""
    hit = total = 0
    for h, r in zip(hyps, refs):
        ht, rt = h.split(), r.split()
        total += len(rt)
        hit += sum(a == b for a, b in zip(ht, rt))
    return hit / max(total, 1)


class TextTask(TaskSpec):
    modality = "text"
    metric_names = ["bleu1", "bleu4", "exact_match", "token_acc"]
    primary_metric = "bleu1"

    def __init__(self, model: dict, tokenizer: dict, eval: dict | None = None):
        self.model_cfg = dict(model)
        self.tok_cfg = dict(tokenizer)
        # required_prompts: also put every class-prompt word in the vocabulary and assert zero <unk> on each prompt set
        self.required_prompts = bool(self.tok_cfg.pop("required_prompts", False))
        self.prompt_root = self.tok_cfg.pop("prompt_root", "data_cache")
        self.vocab_report: dict = {}
        eval = dict(eval or {})
        self.max_items = eval.get("max_items", 2000)
        self.eval_batch_size = eval.get("batch_size", 250)
        self.eval_seed = eval.get("seed", 1234)
        self.tokenizer: WordTokenizer | None = None

    def fit(self, train_dataset) -> None:
        sentences = [train_dataset[i]["input"] for i in range(len(train_dataset))]
        if not self.required_prompts:
            self.tokenizer = WordTokenizer.fit(sentences, **self.tok_cfg)
            return
        from ...data.prompts import prompt_sets

        sets = prompt_sets(self.prompt_root)
        required = [p for ps in sets.values() for p in ps]
        self.tokenizer = WordTokenizer.fit(sentences, required_texts=required, **self.tok_cfg)
        unk = {name: self.tokenizer.count_unk(ps) for name, ps in sets.items()}
        assert not any(unk.values()), f"<unk> in prompt sets: {unk}"
        in_train = {t for s in sentences for t in split_tokens(s)}
        self.vocab_report = {"vocab_size": len(self.tokenizer), "n_training_sentences": len(sentences),
                             "n_required_prompt_words": len({t for p in required for t in split_tokens(p)}),
                             "prompt_words_absent_from_training_sentences": sum(1 for t in {t for p in required for t in split_tokens(p)} if t not in in_train),
                             "unk_per_prompt_set": unk, "prompt_set_sizes": {k: len(v) for k, v in sets.items()}}

    def state_dict(self) -> dict:
        return {"vocab": self.tokenizer.vocab, "max_len": self.tokenizer.max_len, "vocab_report": self.vocab_report}

    def load_state_dict(self, state: dict) -> None:
        self.tokenizer = WordTokenizer(state["vocab"], state["max_len"])
        self.vocab_report = state.get("vocab_report", {})

    def build_model(self) -> TextTransceiver:
        return TextTransceiver(vocab_size=len(self.tokenizer), **self.model_cfg)

    def collate(self, items: list[dict]) -> dict:
        batch = self.tokenizer.make_batch([it["input"] for it in items])
        batch["texts"] = [it["input"] for it in items]  # raw originals; the common-space anchors embed these
        return batch

    def loss(self, model, batch, channel, snr_db):
        src = batch["src"]
        logits = model(src, channel, snr_db, src_mask=src != PAD_ID, tgt_in=batch["tgt_in"])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch["tgt_out"].reshape(-1), ignore_index=PAD_ID)

    @torch.no_grad()
    def evaluate(self, model, dataset, channel, snr_db, device) -> dict[str, float]:
        model.eval()
        torch.manual_seed(self.eval_seed + int(round(snr_db * 10)))
        n = len(dataset) if self.max_items is None else min(self.max_items, len(dataset))
        hyps, refs = [], []
        for start in range(0, n, self.eval_batch_size):
            items = [dataset[i] for i in range(start, min(start + self.eval_batch_size, n))]
            batch = self.to_device(self.collate(items), device)
            src = batch["src"]
            pred = model(src, channel, snr_db, src_mask=src != PAD_ID)
            for p, t in zip(pred.cpu(), batch["tgt_out"].cpu()):
                hyps.append(" ".join(self.tokenizer.ids_to_tokens(p)))
                refs.append(" ".join(self.tokenizer.ids_to_tokens(t)))
        return {
            "bleu1": bleu(hyps, refs, 1),
            "bleu4": bleu(hyps, refs, 4),
            "exact_match": sum(h == r for h, r in zip(hyps, refs)) / len(refs),
            "token_acc": token_accuracy(hyps, refs),
        }
