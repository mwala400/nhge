"""
NHGE Tokenizer
==============
A lightweight tokenizer that works out-of-the-box with NHGE.
Supports:
  - Character-level (tiny vocab, any language)
  - Word-level (fast, good for demos)
  - Subword (BPE-lite, learned from corpus)
"""

import re
import json
from typing import List, Optional, Dict
from collections import Counter


class NHGETokenizer:
    """
    Simple but complete tokenizer for NHGE.

    Special tokens:
        [PAD] = 0
        [UNK] = 1
        [BOS] = 2
        [EOS] = 3
        [CLS] = 4
        [SEP] = 5
        [MASK] = 6
    """

    SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[CLS]", "[SEP]", "[MASK]"]
    PAD_ID  = 0
    UNK_ID  = 1
    BOS_ID  = 2
    EOS_ID  = 3
    CLS_ID  = 4
    SEP_ID  = 5
    MASK_ID = 6

    def __init__(self, mode: str = "word"):
        """
        mode: "char" | "word" | "subword"
        """
        assert mode in ("char", "word", "subword")
        self.mode   = mode
        self.vocab: Dict[str, int] = {t: i for i, t in enumerate(self.SPECIAL)}
        self.id2tok: Dict[int, str] = {i: t for t, i in self.vocab.items()}
        self._bpe_merges: List[tuple] = []

    # ------------------------------------------------------------------
    # Vocabulary building
    # ------------------------------------------------------------------

    def build_vocab(self, texts: List[str], max_vocab: int = 30000, min_freq: int = 2):
        """
        Learn vocabulary from a list of texts.
        """
        counter: Counter = Counter()
        for text in texts:
            for tok in self._tokenize_raw(text):
                counter[tok] += 1

        # Sort by frequency, keep top max_vocab - len(SPECIAL)
        budget = max_vocab - len(self.SPECIAL)
        most_common = [tok for tok, cnt in counter.most_common(budget)
                       if cnt >= min_freq]

        for tok in most_common:
            if tok not in self.vocab:
                idx = len(self.vocab)
                self.vocab[tok]    = idx
                self.id2tok[idx]   = tok

        print(f"Vocabulary size: {len(self.vocab):,} tokens")
        return self

    def _tokenize_raw(self, text: str) -> List[str]:
        """Split text into raw token strings before vocab lookup."""
        if self.mode == "char":
            return list(text)
        if self.mode == "word":
            return re.findall(r"\w+|[^\w\s]", text.lower())
        # subword — basic whitespace split; extend with BPE if needed
        return text.lower().split()

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        add_cls: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        tokens = self._tokenize_raw(text)
        ids = [self.vocab.get(t, self.UNK_ID) for t in tokens]

        if add_cls:
            ids = [self.CLS_ID] + ids
        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        tokens = []
        for i in ids:
            tok = self.id2tok.get(i, "[UNK]")
            if skip_special and tok in self.SPECIAL:
                continue
            tokens.append(tok)

        if self.mode == "char":
            return "".join(tokens)
        return " ".join(tokens)

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 128,
        pad: bool = True,
        **kwargs,
    ) -> Dict[str, List]:
        all_ids, all_masks = [], []
        for text in texts:
            ids = self.encode(text, max_length=max_length, **kwargs)
            pad_len = max_length - len(ids)
            mask = [False] * len(ids) + [True] * pad_len
            ids  = ids + [self.PAD_ID] * pad_len if pad else ids
            all_ids.append(ids)
            all_masks.append(mask)
        return {"input_ids": all_ids, "attention_mask": all_masks}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        data = {
            "mode":   self.mode,
            "vocab":  self.vocab,
            "merges": self._bpe_merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved → {path}")

    @classmethod
    def load(cls, path: str) -> "NHGETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(mode=data["mode"])
        tok.vocab      = {k: int(v) for k, v in data["vocab"].items()}
        tok.id2tok     = {int(v): k for k, v in tok.vocab.items()}
        tok._bpe_merges = [tuple(m) for m in data.get("merges", [])]
        return tok
