"""Closed-vocabulary head over the English targets, plus a character fallback.

Why this exists
---------------
Measured on the real corpus: 1,245,593 training spans reduce to only **15,326
unique English targets**. The top 1,000 cover 93.3% of test spans and the full
lexicon covers 99.4%. Only 0.6% is genuinely unseen -- and that tail is almost
entirely emails and URLs, which are *compositional* rather than arbitrary.

Free character-by-character generation was therefore doing difficult, error-prone
work for the 99.4% of cases where a lookup is exact by construction. The smoke
run showed exactly the resulting failure: `فريندس` decoding to `devers` and
`deversion` -- plausible English orthography, wrong word, because a from-scratch
character decoder has no lexical prior.

The design
----------
Two heads over one shared encoder:

* **Word head** -- a softmax over the lexicon. When it is confident, the output
  is a *whole dictionary word*, so misspellings are impossible.
* **Character head** -- the existing autoregressive decoder, used only when the
  word head is unsure. This keeps the open vocabulary that brand names, unseen
  jargon and URLs need.

The two are trained jointly on the same spans, so the word head learns which
inputs it can answer and the character head still sees everything.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

# Reserved slot 0: "this span is not in the lexicon, use the character decoder".
OOV = "<oov>"
OOV_ID = 0


class Lexicon:
    """Maps English targets to word ids, and back."""

    def __init__(self, words: list[str]) -> None:
        # OOV first so its id is stable at 0 regardless of corpus.
        self.itos: list[str] = [OOV] + [w for w in words if w != OOV]
        self.stoi: dict[str, int] = {w: i for i, w in enumerate(self.itos)}
        # Case-insensitive index, so "Meeting" can still resolve to "meeting"
        # when only one casing was seen in training.
        self._lower: dict[str, int] = {}
        for w, i in self.stoi.items():
            self._lower.setdefault(w.lower(), i)

    def __len__(self) -> int:
        return len(self.itos)

    def get(self, word: str) -> int:
        """Word id, or ``OOV_ID`` when the word is not covered."""
        i = self.stoi.get(word)
        if i is not None:
            return i
        return self._lower.get(word.lower(), OOV_ID)

    def word(self, idx: int) -> str:
        return self.itos[idx] if 0 <= idx < len(self.itos) else OOV

    # --- construction -----------------------------------------------------
    @classmethod
    def from_spans(
        cls,
        targets: Iterable[str],
        *,
        max_size: int = 20000,
        min_count: int = 2,
    ) -> "Lexicon":
        """Build from training targets, keeping the frequent ones.

        ``min_count`` drops hapax targets: a word seen once is usually a typo or
        a one-off entity, and the character decoder handles those better than a
        softmax slot trained on a single example.
        """
        counts = Counter(t for t in targets if t)
        kept = [w for w, c in counts.most_common(max_size) if c >= min_count]
        return cls(kept)

    # --- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"words": self.itos}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Lexicon":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        lex = cls.__new__(cls)
        lex.itos = data["words"]
        lex.stoi = {w: i for i, w in enumerate(lex.itos)}
        lex._lower = {}
        for w, i in lex.stoi.items():
            lex._lower.setdefault(w.lower(), i)
        return lex

    def coverage(self, targets: Iterable[str]) -> float:
        """Share of ``targets`` this lexicon can answer. Useful as a sanity log."""
        seen = list(targets)
        if not seen:
            return 0.0
        return sum(1 for t in seen if self.get(t) != OOV_ID) / len(seen)
