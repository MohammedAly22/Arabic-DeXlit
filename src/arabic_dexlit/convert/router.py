"""Category router: sends each detected span to the right converter.

The organising principle is **deterministic where structure exists, neural where
ambiguity exists**, with copy-by-default whenever nothing is confident enough.

    span + category
          │
          ├── EMAIL / URL / NUMBER / TIME ─ parser ─────────► exact, or decline
          ├── ACRONYM ─────────────────── inventory lookup ─► closed set
          ├── ENTITY ──────────────────── gazetteer, then neural
          └── CS ──────────────────────── neural, with a rule prior
                          │
                          ▼
                 candidates + scores
                          │
                   confident enough?
                    ┌─────┴─────┐
                   yes          no
                    │            │
                 replace       COPY

Every path can decline. A span nothing is confident about is returned unchanged,
because for an ASR post-processor a missed conversion is a much cheaper error
than a corrupted one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import re

from .parsers import DETERMINISTIC_PARSERS, parse_span
from .phonetic import PhoneticIndex
from .rerank import CandidateReranker
from .translit_rules import transliterate_span


@dataclass
class Candidate:
    """One proposed conversion, with where it came from and how much to trust it."""

    text: str
    source: str          # parser | acronym | gazetteer | neural | rule | copy
    score: float         # 0..1, comparable within a span

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Candidate({self.text!r}, {self.source}, {self.score:.2f})"


@dataclass
class SpanResult:
    text: str
    source: str
    score: float
    candidates: list[Candidate] = field(default_factory=list)
    converted: bool = True


# Confidence a neural candidate must reach before it may replace the original.
# Deliberately high: the cost of a wrong edit (corrupting text nobody asked to
# change) is far greater than the cost of a missed one.
DEFAULT_THRESHOLD = 0.60

_LATIN = re.compile(r"[A-Za-z]")
_ARABIC = re.compile(r"[؀-ۿ]")

# Parsers are exact when they succeed, so they carry full confidence.
PARSER_SCORE = 1.0
ACRONYM_SCORE = 0.95
GAZETTEER_SCORE = 0.92


class ConversionRouter:
    """Routes spans by category and picks a winner among candidates."""

    def __init__(
        self,
        *,
        neural_fn: Callable[[list[list[str]], list[str], list[list[str]]], list[tuple[str, float]]] | None = None,
        acronyms=None,
        gazetteer: dict[str, str] | None = None,
        lexicon=None,
        threshold: float = DEFAULT_THRESHOLD,
        use_rule_fallback: bool = False,
        phonetic_index: "PhoneticIndex | None" = None,
        reranker: "CandidateReranker | None" = None,
    ) -> None:
        self.neural_fn = neural_fn
        self.acronyms = acronyms
        self.gazetteer = {k.strip(): v for k, v in (gazetteer or {}).items()}
        self.lexicon = lexicon
        self.threshold = threshold
        # Off by default: a phonetic skeleton ("mitinj") is usually worse than
        # leaving the original alone, so it is only used when explicitly asked.
        self.use_rule_fallback = use_rule_fallback
        # Generate-and-rerank. The phonetic index proposes words that *sound*
        # like the span -- which is how an unseen spelling is answered at all --
        # and the reranker weighs those against the rule and neural proposals
        # using context. Without these the router can only answer forms it has
        # literally seen before.
        self.phonetic_index = phonetic_index
        self.reranker = reranker

    # --- per-category candidate generation --------------------------------
    def _parser_candidate(self, tokens: list[str], category: str) -> Candidate | None:
        if category not in DETERMINISTIC_PARSERS:
            return None
        parsed = parse_span(tokens, category)
        return Candidate(parsed, "parser", PARSER_SCORE) if parsed else None

    def _acronym_candidate(self, tokens: list[str]) -> Candidate | None:
        """An acronym has a closed answer set, so look it up rather than generate."""
        if self.acronyms is None:
            return None
        from .parsers import LETTER_WORDS

        letters: list[str] = []
        for tok in tokens:
            t = tok.strip(".,!?؟،:;\"'()")
            if not t:
                continue
            if t.isalpha() and t.isascii():
                letters.append(t)          # already Latin: "PM"
            elif t in LETTER_WORDS:
                letters.append(LETTER_WORDS[t])
            else:
                return None
        if not letters:
            return None
        guess = "".join(letters).upper()
        idx = self.acronyms.get(guess)
        if idx is None:
            return None
        return Candidate(self.acronyms.at(idx), "acronym", ACRONYM_SCORE)

    def _gazetteer_candidate(self, tokens: list[str]) -> Candidate | None:
        """Entity resolution is lookup, not transliteration: مكروسوفت تيمس is
        ``Microsoft Teams`` because that company exists, not because the letters
        say so."""
        if not self.gazetteer:
            return None
        key = " ".join(tokens).strip()
        hit = self.gazetteer.get(key)
        if hit:
            return Candidate(hit, "gazetteer", GAZETTEER_SCORE)
        return None

    # --- routing ----------------------------------------------------------
    def convert(
        self,
        spans: list[tuple[list[str], str, list[str]]],
    ) -> list[SpanResult]:
        """Convert spans given as ``(tokens, category, context_tokens)``."""
        results: list[SpanResult | None] = [None] * len(spans)
        neural_queue: list[int] = []

        for i, (tokens, category, _ctx) in enumerate(spans):
            cands: list[Candidate] = []

            parsed = self._parser_candidate(tokens, category)
            if parsed:
                results[i] = SpanResult(parsed.text, "parser", parsed.score, [parsed])
                continue

            if category == "ACRONYM":
                acr = self._acronym_candidate(tokens)
                if acr:
                    results[i] = SpanResult(acr.text, "acronym", acr.score, [acr])
                    continue

            if category == "ENTITY":
                gaz = self._gazetteer_candidate(tokens)
                if gaz:
                    results[i] = SpanResult(gaz.text, "gazetteer", gaz.score, [gaz])
                    continue

            # Nothing deterministic applied. A span already in Latin script is
            # then its own answer: 34.7% of spans arrive this way, and sending
            # them to phonetic retrieval was the largest source of wrong edits,
            # since the index cheerfully proposes a different word that merely
            # sounds similar.
            joined = " ".join(tokens)
            if _LATIN.search(joined) and not _ARABIC.search(joined):
                results[i] = SpanResult(
                    joined, "copy", 1.0, [Candidate(joined, "copy", 1.0)],
                    converted=False,
                )
                continue

            # Genuinely ambiguous: generate candidates and rerank them.
            neural_queue.append(i)

        if neural_queue and self.neural_fn is not None:
            batch = [spans[i][0] for i in neural_queue]
            cats = [spans[i][1] for i in neural_queue]
            ctxs = [spans[i][2] for i in neural_queue]
            for i, (text, score) in zip(neural_queue, self.neural_fn(batch, cats, ctxs)):
                results[i] = self._decide(spans[i][0], text, score, spans[i][2])
        else:
            for i in neural_queue:
                results[i] = self._decide(spans[i][0], None, 0.0, spans[i][2])

        return [r for r in results if r is not None]

    def _decide(
        self,
        tokens: list[str],
        neural_text: str | None,
        score: float,
        context: list[str] | None = None,
    ) -> SpanResult:
        """Gather every proposal, rank them, and accept only a confident winner."""
        original = " ".join(tokens)
        span_text = original

        proposals: list[tuple[str, str, float]] = []
        if neural_text:
            proposals.append((neural_text, "neural", score))

        # Phonetic retrieval: the generality path. Proposes English words whose
        # sound matches the span, so a spelling never seen in training can still
        # be answered.
        if self.phonetic_index is not None:
            for cand in self.phonetic_index.lookup(span_text, limit=8):
                proposals.append((cand, "phonetic", 0.0))

        rule = transliterate_span(tokens)
        if rule and (self.use_rule_fallback or self.reranker is not None):
            proposals.append((rule, "rule", 0.0))

        if not proposals:
            return SpanResult(original, "copy", 1.0 - score, [], converted=False)

        if self.reranker is not None:
            ranked = self.reranker.rank(span_text, proposals, context or [])
            cands = [Candidate(c.text, c.source, c.total) for c in ranked]
            top = ranked[0]
            if top.total >= self.threshold:
                return SpanResult(top.text, top.source, top.total, cands)
            # Nothing cleared the bar: leave the span alone. A missed conversion
            # is recoverable by a reader; a confident wrong edit is not.
            cands.append(Candidate(original, "copy", 1.0 - top.total))
            return SpanResult(original, "copy", 1.0 - top.total, cands, converted=False)

        # No reranker configured: fall back to trusting the neural score alone.
        cands = [Candidate(t, srcname, c) for t, srcname, c in proposals]
        if neural_text and score >= self.threshold:
            return SpanResult(neural_text, "neural", score, cands)
        cands.append(Candidate(original, "copy", 1.0 - score))
        return SpanResult(original, "copy", 1.0 - score, cands, converted=False)
