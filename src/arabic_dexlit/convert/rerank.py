"""Candidate reranking: choose among proposals using evidence, not one guess.

The generators disagree on purpose. For ``فريندس`` the rule table proposes
``frinds``, the phonetic index proposes ``friends`` and ``French``, and the
neural converter proposes something of its own. Trusting any single generator
means inheriting its failure mode; scoring all of them against the same evidence
lets the strengths cover each other.

Each candidate is scored as a weighted sum:

    score = w_source * source_prior
          + w_phon   * phonetic_agreement
          + w_lex    * lexicon_frequency
          + w_neural * neural_confidence
          + w_ctx    * context_fit

* **phonetic agreement** is the decisive signal for generality. It compares the
  candidate's English phonetic key with the Arabic span's key, so a candidate
  that *sounds* like the input wins even if neither was ever seen in training.
* **context fit** is what separates ``friends`` from ``French`` in
  ``انا عندي فريندس في الشركة``: candidates are scored by how often they occur
  with the surrounding Arabic words in the training corpus.
* **lexicon frequency** breaks ties toward words people actually use.

Nothing here is learned end to end. The weights are explicit and inspectable, so
a bad ranking can be diagnosed by reading the component scores rather than by
retraining.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .phonetic import arabic_key, english_key

# Prior trust in each generator, before any evidence is considered.
SOURCE_PRIOR: dict[str, float] = {
    "parser": 1.00,      # deterministic and verified
    "acronym": 0.95,     # closed set
    "gazetteer": 0.92,   # observed entity mapping
    "lexicon": 0.80,     # seen in training for this exact form
    "neural": 0.70,      # learned, but can hallucinate
    "phonetic": 0.60,    # sounds right, may be the wrong word
    "rule": 0.15,        # phonetic skeleton, usually misspelled -- last resort
    "copy": 0.50,        # leave it alone
}


@dataclass
class ScoredCandidate:
    text: str
    source: str
    total: float = 0.0
    parts: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        bits = " ".join(f"{k}={v:.2f}" for k, v in self.parts.items())
        return f"<{self.text!r} {self.source} {self.total:.3f} [{bits}]>"


def phonetic_agreement(arabic_span: str, candidate: str) -> float:
    """How closely a candidate sounds like the Arabic span, in ``[0, 1]``.

    This is what lets the system answer words it has never seen: agreement is
    computed from the phonetic keys, not from any observed pairing.
    """
    a = arabic_key(arabic_span, drop_vowels=True)
    c = english_key(candidate, drop_vowels=True)
    if not a and not c:
        return 0.5
    if not a or not c:
        return 0.0
    if a == c:
        return 1.0
    # Normalised edit distance over the keys.
    prev = list(range(len(c) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cc in enumerate(c, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cc)))
        prev = cur
    return max(0.0, 1.0 - prev[-1] / max(len(a), len(c)))


class ContextModel:
    """How well a candidate fits the Arabic words around it.

    A small co-occurrence table: for each English word, which Arabic words were
    nearby when it appeared. Deliberately not a neural LM -- it trains in
    seconds, is inspectable, and the signal needed here is association, not
    fluency.
    """

    def __init__(self, window: int = 4) -> None:
        self.window = window
        self.cooc: dict[str, Counter] = defaultdict(Counter)
        self.word_total: Counter = Counter()
        self.context_total: Counter = Counter()
        self.n = 0

    def fit(self, rows: Iterable[dict]) -> "ContextModel":
        for r in rows:
            toks = r.get("src_tokens", [])
            for s in r.get("spans", []):
                target = s.get("target", "")
                if not target:
                    continue
                lo = max(0, s["start"] - self.window)
                hi = min(len(toks), s["end"] + self.window)
                ctx = toks[lo : s["start"]] + toks[s["end"] : hi]
                self.word_total[target] += 1
                self.n += 1
                for c in ctx:
                    self.cooc[target][c] += 1
                    self.context_total[c] += 1
        return self

    def score(self, candidate: str, context: list[str]) -> float:
        """Pointwise-mutual-information-style fit, squashed to ``[0, 1]``."""
        if not context or candidate not in self.word_total:
            return 0.5  # no evidence either way
        seen = self.cooc[candidate]
        total = max(1, self.word_total[candidate])
        acc = 0.0
        for c in context:
            joint = seen.get(c, 0) / total
            marginal = self.context_total.get(c, 0) / max(1, self.n)
            if joint > 0 and marginal > 0:
                acc += math.log(joint / marginal)
        if not acc:
            return 0.5
        return 1.0 / (1.0 + math.exp(-acc / max(1, len(context))))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "window": self.window,
                    "n": self.n,
                    "word_total": dict(self.word_total),
                    "context_total": dict(self.context_total),
                    # Only the strongest associations are kept: the tail is
                    # noise and would multiply the file size for nothing.
                    "cooc": {
                        w: dict(c.most_common(40)) for w, c in self.cooc.items()
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ContextModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(window=d.get("window", 4))
        m.n = d.get("n", 0)
        m.word_total = Counter(d.get("word_total", {}))
        m.context_total = Counter(d.get("context_total", {}))
        m.cooc = defaultdict(Counter, {w: Counter(c) for w, c in d.get("cooc", {}).items()})
        return m


class CandidateReranker:
    """Scores candidates from every generator on one comparable scale."""

    def __init__(
        self,
        *,
        context_model: ContextModel | None = None,
        lexicon_counts: Counter | None = None,
        w_source: float = 0.30,
        w_phonetic: float = 0.30,
        w_lexicon: float = 0.10,
        w_neural: float = 0.15,
        w_context: float = 0.15,
    ) -> None:
        self.ctx = context_model
        self.lex = lexicon_counts or Counter()
        self._lex_max = math.log1p(max(self.lex.values())) if self.lex else 1.0
        self.w = {
            "source": w_source,
            "phonetic": w_phonetic,
            "lexicon": w_lexicon,
            "neural": w_neural,
            "context": w_context,
        }

    def rank(
        self,
        arabic_span: str,
        candidates: Iterable[tuple[str, str, float]],
        context: list[str] | None = None,
    ) -> list[ScoredCandidate]:
        """Rank ``(text, source, neural_confidence)`` triples, best first."""
        out: list[ScoredCandidate] = []
        seen: set[str] = set()
        for text, source, conf in candidates:
            if not text or text in seen:
                continue
            seen.add(text)

            phon = phonetic_agreement(arabic_span, text)
            if source == "rule":
                # The rule output is derived from the same phonetic model the
                # key uses, so its agreement is 1.0 by construction -- circular
                # evidence, not a signal. Neutralise it, or a known misspelling
                # ("frinds") outranks the real word on a score it cannot lose.
                phon = 0.5
            parts = {
                "source": SOURCE_PRIOR.get(source, 0.5),
                "phonetic": phon,
                "lexicon": (
                    math.log1p(self.lex.get(text, 0)) / self._lex_max
                    if self._lex_max
                    else 0.0
                ),
                "neural": conf,
                "context": self.ctx.score(text, context or []) if self.ctx else 0.5,
            }
            total = sum(self.w[k] * v for k, v in parts.items())
            out.append(ScoredCandidate(text, source, total, parts))

        out.sort(key=lambda c: -c.total)
        return out

    def best(
        self,
        arabic_span: str,
        candidates: Iterable[tuple[str, str, float]],
        context: list[str] | None = None,
        threshold: float = 0.0,
    ) -> ScoredCandidate | None:
        ranked = self.rank(arabic_span, candidates, context)
        if not ranked or ranked[0].total < threshold:
            return None
        return ranked[0]
