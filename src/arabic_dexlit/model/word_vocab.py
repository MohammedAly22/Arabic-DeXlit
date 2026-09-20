"""Word-level vocabulary for the dual-conditioned rewriter.

Sizing this is not obvious, and the measurement matters. The corpus holds
~199,000 distinct word types -- Arabic morphology inflates that badly -- so even
a 64K vocabulary covers only 92.5% of tokens. A plain word-level seq2seq would
therefore be crippled by unknown words.

The copy mechanism changes the arithmetic. Measured on the corpus, **90.8% of
target tokens falling outside a 32K vocabulary are present verbatim in the
source**, so they are recoverable by pointing rather than generating. A modest
vocabulary plus copy beats a large vocabulary without it, at a fraction of the
embedding cost.

Hence 32K: large enough that frequent words are generated fluently, small enough
to keep the output projection cheap, with copy covering the tail.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


class WordVocab:
    """Bidirectional word <-> id mapping, shared by source and target."""

    def __init__(self, words: Iterable[str]) -> None:
        self.itos: list[str] = list(SPECIALS)
        seen = set(SPECIALS)
        for w in words:
            if w and w not in seen:
                seen.add(w)
                self.itos.append(w)
        self.stoi: dict[str, int] = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str], max_len: int, *, add_eos: bool = True) -> list[int]:
        ids = [self.stoi.get(t, UNK_ID) for t in tokens[: max_len - 1]]
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(
        self,
        ids: list[int],
        src_tokens: list[str] | None = None,
        copy_positions: list[int] | None = None,
    ) -> list[str]:
        """Ids back to words.

        ``src_tokens`` matters: a copied token can carry an id that is ``<unk>``
        in the vocabulary, and the only faithful rendering is the source word
        itself. Without this, exactly the rare words copying exists to preserve
        would come back as ``<unk>``.
        """
        out: list[str] = []
        for step, i in enumerate(ids):
            if i in (PAD_ID, BOS_ID):
                continue
            if i == EOS_ID:
                break
            w = self.itos[i] if 0 <= i < len(self.itos) else UNK
            # Recover an out-of-vocabulary copy from the source position the
            # pointer selected. Without this the rare words the copy mechanism
            # exists to preserve are precisely the ones lost to <unk>.
            if (
                w == UNK
                and src_tokens is not None
                and copy_positions is not None
                and step < len(copy_positions)
            ):
                pos = copy_positions[step]
                if 0 <= pos < len(src_tokens):
                    w = src_tokens[pos]
            out.append(w)
        return out

    # --- construction -----------------------------------------------------
    @classmethod
    def build(
        cls,
        rows: Iterable[dict],
        *,
        max_size: int = 32000,
        min_count: int = 2,
    ) -> "WordVocab":
        """Build from both sides of the corpus.

        Source and target share one vocabulary because 84% of output tokens are
        copied from the input -- separate vocabularies would duplicate almost
        everything and break the copy mechanism's id alignment.
        """
        counts: Counter = Counter()
        for r in rows:
            counts.update(r["src_tokens"])
            counts.update(r["tgt"].split())
        words = [w for w, c in counts.most_common(max_size) if c >= min_count]
        return cls(words)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"words": self.itos}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "WordVocab":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        v = cls.__new__(cls)
        v.itos = data["words"]
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v

    def coverage(self, rows: Iterable[dict]) -> dict[str, float]:
        """Token coverage, and how much of the remainder copying can recover."""
        total = oov = copyable = 0
        for r in rows:
            src = set(r["src_tokens"])
            for t in r["tgt"].split():
                total += 1
                if t not in self.stoi:
                    oov += 1
                    copyable += t in src
        return {
            "token_coverage": 1.0 - oov / max(1, total),
            "oov_recoverable_by_copy": copyable / max(1, oov),
        }
