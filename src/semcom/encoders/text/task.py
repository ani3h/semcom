import torch
import torch.nn.functional as F
from sacrebleu.metrics import BLEU

from ..base import TaskSpec
from .model import TextTransceiver
from .tokenizer import PAD_ID, WordTokenizer


def bleu(hyps: list[str], refs: list[str], n: int) -> float:
    """Corpus BLEU-n in [0, 1] on pre-tokenized (space-joined) text."""
    metric = BLEU(tokenize="none", max_ngram_order=n, smooth_method="none", effective_order=False)
    return metric.corpus_score(hyps, [refs]).score / 100.0


class TextTask(TaskSpec):
    modality = "text"
    metric_names = ["bleu1", "bleu4", "exact_match"]
    primary_metric = "bleu1"

    def __init__(self, model: dict, tokenizer: dict, eval: dict | None = None):
        self.model_cfg = dict(model)
        self.tok_cfg = dict(tokenizer)
        eval = dict(eval or {})
        self.max_items = eval.get("max_items", 2000)
        self.eval_batch_size = eval.get("batch_size", 250)
        self.eval_seed = eval.get("seed", 1234)
        self.tokenizer: WordTokenizer | None = None

    def fit(self, train_dataset) -> None:
        sentences = [train_dataset[i]["input"] for i in range(len(train_dataset))]
        self.tokenizer = WordTokenizer.fit(sentences, **self.tok_cfg)

    def state_dict(self) -> dict:
        return {"vocab": self.tokenizer.vocab, "max_len": self.tokenizer.max_len}

    def load_state_dict(self, state: dict) -> None:
        self.tokenizer = WordTokenizer(state["vocab"], state["max_len"])

    def build_model(self) -> TextTransceiver:
        return TextTransceiver(vocab_size=len(self.tokenizer), **self.model_cfg)

    def collate(self, items: list[dict]) -> dict[str, torch.Tensor]:
        return self.tokenizer.make_batch([it["input"] for it in items])

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
        }
