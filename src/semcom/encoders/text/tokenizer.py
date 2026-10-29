import re
from collections import Counter

import torch

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
_SPECIALS = [PAD, BOS, EOS, UNK]
_TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9']")


def split_tokens(sentence: str) -> list[str]:
    """Lowercased words and individual punctuation marks."""
    return _TOKEN_RE.findall(sentence.lower())


class WordTokenizer:
    def __init__(self, vocab: list[str], max_len: int = 40):
        assert vocab[: len(_SPECIALS)] == _SPECIALS, "vocab must start with the special tokens"
        self.vocab = vocab
        self.max_len = max_len
        self._index = {tok: i for i, tok in enumerate(vocab)}

    @classmethod
    def fit(cls, sentences: list[str], max_vocab: int = 20000, min_freq: int = 3, max_len: int = 40,
            required_texts: list[str] | None = None):
        """Vocabulary from `sentences` (words seen at least min_freq times, most frequent first, up to max_vocab) UNIONED with
        every token of `required_texts` regardless of frequency, so those texts never contain <unk>."""
        counts = Counter(tok for s in sentences for tok in split_tokens(s))
        required = list(dict.fromkeys(t for s in (required_texts or []) for t in split_tokens(s) if t not in _SPECIALS))
        budget = max_vocab - len(_SPECIALS) - len(required)
        if budget < 0:
            raise ValueError("max_vocab is smaller than the required words")
        frequent = [w for w, c in counts.most_common() if c >= min_freq and w not in set(required)][:budget]
        return cls(_SPECIALS + required + frequent, max_len)

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, sentence: str) -> list[int]:
        """Token ids truncated to max_len (no BOS/EOS)."""
        return [self._index.get(t, UNK_ID) for t in split_tokens(sentence)[: self.max_len]]

    def count_unk(self, sentences: list[str]) -> int:
        """Number of <unk> tokens over `sentences` (tokens beyond max_len are not counted: they are truncated, not unknown)."""
        return sum(i == UNK_ID for s in sentences for i in self.encode(s))

    def ids_to_tokens(self, ids) -> list[str]:
        """Token strings up to (excluding) the first EOS; PAD/BOS are dropped."""
        out = []
        for i in ids:
            i = int(i)
            if i == EOS_ID:
                break
            if i not in (PAD_ID, BOS_ID):
                out.append(self.vocab[i])
        return out

    def make_batch(self, sentences: list[str]) -> dict[str, torch.Tensor]:
        """src = tokens+EOS (what is transmitted), tgt_in = BOS+tokens, tgt_out = tokens+EOS."""
        seqs = [self.encode(s) for s in sentences]
        width = max(len(s) for s in seqs) + 1
        src = torch.full((len(seqs), width), PAD_ID, dtype=torch.long)
        tgt_in = torch.full_like(src, PAD_ID)
        for row, seq in enumerate(seqs):
            n = len(seq)
            src[row, :n] = torch.tensor(seq, dtype=torch.long)
            src[row, n] = EOS_ID
            tgt_in[row, 0] = BOS_ID
            tgt_in[row, 1 : n + 1] = torch.tensor(seq, dtype=torch.long)
        return {"src": src, "tgt_in": tgt_in, "tgt_out": src.clone()}
