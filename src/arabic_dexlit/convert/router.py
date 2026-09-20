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

from .parsers import DETERMINISTIC_PARSERS, parse_span
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
    ) -> None:
        self.neural_fn = neural_fn
        self.acronyms = acronyms
        self.gazetteer = {k.strip(): v for k, v in (gazetteer or {}).items()}
        self.lexicon = lexicon
        self.threshold = threshold
        # Off by default: a phonetic skeleton ("mitinj") is usually worse than
        # leaving the original alone, so it is only used when explicitly asked.
        self.use_rule_fallback = use_rule_fallback

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

            # Nothing deterministic applied: the neural converter decides.
            neural_queue.append(i)

        if neural_queue and self.neural_fn is not None:
            batch = [spans[i][0] for i in neural_queue]
            cats = [spans[i][1] for i in neural_queue]
            ctxs = [spans[i][2] for i in neural_queue]
            for i, (text, score) in zip(neural_queue, self.neural_fn(batch, cats, ctxs)):
                results[i] = self._decide(spans[i][0], text, score)
        else:
            for i in neural_queue:
                results[i] = self._decide(spans[i][0], None, 0.0)

        return [r for r in results if r is not None]

    def _decide(
        self, tokens: list[str], neural_text: str | None, score: float
    ) -> SpanResult:
        """Accept the neural answer only if it clears the bar; else keep the input."""
        original = " ".join(tokens)
        cands: list[Candidate] = []
        if neural_text:
            cands.append(Candidate(neural_text, "neural", score))
        if self.use_rule_fallback:
            rule = transliterate_span(tokens)
            if rule:
                cands.append(Candidate(rule, "rule", 0.30))

        if neural_text and score >= self.threshold:
            return SpanResult(neural_text, "neural", score, cands)

        # Copy-by-default. A missed conversion is recoverable by a human reader;
        # a confident wrong edit is not.
        cands.append(Candidate(original, "copy", 1.0 - score))
        return SpanResult(original, "copy", 1.0 - score, cands, converted=False)
