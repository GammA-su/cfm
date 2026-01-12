from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", re.UNICODE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class SimpleTokenizer:
    vocab: Dict[str, int]
    inv_vocab: Dict[int, str]
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    sep_token: str = "<sep>"
    unk_token: str = "<unk>"

    @classmethod
    def build(cls, texts: Iterable[str], vocab_max: int | None = None) -> "SimpleTokenizer":
        tokens = set()
        for text in texts:
            tokens.update(TOKEN_PATTERN.findall(text))
        special = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]
        vocab_items = special + sorted(tokens)
        if vocab_max:
            vocab_items = vocab_items[:vocab_max]
        vocab = {tok: i for i, tok in enumerate(vocab_items)}
        inv_vocab = {i: tok for tok, i in vocab.items()}
        return cls(vocab=vocab, inv_vocab=inv_vocab)

    def encode(self, text: str) -> List[int]:
        ids = []
        for tok in TOKEN_PATTERN.findall(text):
            ids.append(self.vocab.get(tok, self.vocab[self.unk_token]))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        tokens = [self.inv_vocab.get(i, self.unk_token) for i in ids]
        filtered = [t for t in tokens if t not in {self.bos_token, self.eos_token, self.sep_token, self.pad_token}]
        return " ".join(filtered).strip()

    def save(self, path: Path) -> None:
        payload = {
            "vocab": self.vocab,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "sep_token": self.sep_token,
            "unk_token": self.unk_token,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> "SimpleTokenizer":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        vocab = payload["vocab"]
        inv_vocab = {int(v): k for k, v in vocab.items()}
        return cls(
            vocab=vocab,
            inv_vocab=inv_vocab,
            pad_token=payload["pad_token"],
            bos_token=payload["bos_token"],
            eos_token=payload["eos_token"],
            sep_token=payload["sep_token"],
            unk_token=payload["unk_token"],
        )

    @property
    def pad_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.vocab[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.vocab[self.eos_token]

    @property
    def sep_id(self) -> int:
        return self.vocab[self.sep_token]


def pad_sequences(seqs: List[List[int]], pad_id: int, max_len: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if not max_len:
        max_len = max(len(s) for s in seqs)
    batch = []
    mask = []
    for seq in seqs:
        padded = seq[:max_len] + [pad_id] * max(0, max_len - len(seq))
        attention = [1] * min(len(seq), max_len) + [0] * max(0, max_len - len(seq))
        batch.append(padded)
        mask.append(attention)
    return torch.tensor(batch, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


__all__ = ["set_seed", "SimpleTokenizer", "pad_sequences"]
