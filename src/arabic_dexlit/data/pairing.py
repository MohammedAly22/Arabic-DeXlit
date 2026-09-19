"""Turn a *target* sentence into a training example.

A target sentence is Arabic with English left in Latin script -- the form we
want the model to produce. Corpora of this form are plentiful; corpora of the
noisy ASR form are not. So we invert: rewrite the English pieces into Arabic
letters to synthesise the input, recording where each rewrite landed.

The output of this module is a triple that is *consistent by construction*:

    src_tokens -- what the model sees (ASR-style, English in Arabic letters)
    tags       -- one BIO tag per input token
    tgt_tokens -- the gold output

Because spans are recorded *during* generation rather than recovered by post-hoc
alignment, the labels cannot silently drift out of sync with the text.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from ..schema import OUTSIDE
from .translit import (
    LEXICON,
    speak_email,
    speak_number,
    spell_acronym,
    transliterate_word,
)

# A token is Latin-script if it holds any ASCII letter.
_LATIN = re.compile(r"[A-Za-z]")
_ARABIC = re.compile(r"[؀-ۿ]")
_EMAILISH = re.compile(
    r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    r"|^(?:https?://|www\.)\S+$"
)
_AFFIX = re.compile(r"^([^\w@]*)(.*?)([^\w@]*)$", re.UNICODE)


@dataclass
class Example:
    """One aligned training instance."""

    src_tokens: list[str]
    tags: list[str]
    tgt_tokens: list[str]
    spans: list[dict] = field(default_factory=list)
    dialect: str = "unk"
    source: str = "unk"

    @property
    def src(self) -> str:
        return " ".join(self.src_tokens)

    @property
    def tgt(self) -> str:
        return " ".join(self.tgt_tokens)

    def is_passthrough(self) -> bool:
        return all(t == OUTSIDE for t in self.tags)

    def to_dict(self) -> dict:
        return {
            "src": self.src,
            "tgt": self.tgt,
            "src_tokens": self.src_tokens,
            "tags": self.tags,
            "spans": self.spans,
            "dialect": self.dialect,
            "source": self.source,
        }


def classify_latin(tok: str) -> str:
    """Decide which conversion policy a Latin token needs."""
    core = tok.strip(".,!?؟،:;\"'()[]")
    if _EMAILISH.match(core):
        return "EMAIL"
    if core.isdigit():
        return "NUMBER"
    # All-caps and short => spoken letter by letter (AI, AIC, OIE, NLP).
    if core.isupper() and 2 <= len(core) <= 5 and core.isalpha():
        return "ACRONYM"
    return "CS"


def _split_affix(tok: str) -> tuple[str, str, str]:
    """Peel punctuation off a token: ``"work."`` -> ``("", "work", ".")``."""
    m = _AFFIX.match(tok)
    if not m:
        return "", tok, ""
    return m.group(1), m.group(2), m.group(3)


def _render(core: str, cat: str, rng: random.Random) -> list[str]:
    """Render one Latin core in Arabic script under the given policy."""
    if cat == "EMAIL":
        return speak_email(core, rng).split()
    if cat == "NUMBER":
        return speak_number(core, rng).split()
    if cat == "ACRONYM":
        return spell_acronym(core, rng).split()
    low = core.lower()
    form = rng.choice(LEXICON[low]) if low in LEXICON else transliterate_word(core, rng)
    return form.split()


def build_example(
    target_sentence: str,
    rng: random.Random,
    *,
    dialect: str = "unk",
    source: str = "unk",
    entity_prob: float = 0.35,
) -> Example | None:
    """Synthesise the ASR-style input for ``target_sentence`` and tag it.

    Returns ``None`` when the sentence holds no Latin content, since such a
    sentence teaches nothing about conversion. Pure-Arabic pass-through examples
    are added separately, in a controlled ratio, via :func:`passthrough_example`.
    """
    raw = target_sentence.strip()
    if not raw:
        return None
    toks = raw.split()
    if not toks:
        return None

    src: list[str] = []
    tags: list[str] = []
    tgt: list[str] = []
    spans: list[dict] = []

    i = 0
    while i < len(toks):
        tok = toks[i]
        if not _LATIN.search(tok):
            src.append(tok)
            tags.append(OUTSIDE)
            tgt.append(tok)
            i += 1
            continue

        # --- a run of Latin tokens starts here ---------------------------
        run_end = i
        while run_end < len(toks) and _LATIN.search(toks[run_end]):
            run_end += 1
        run = toks[i:run_end]

        # A multi-word Title Case run is usually a proper noun ("Orange
        # Innovation Egypt"). Group it into ONE entity span some of the time so
        # the model sees genuinely multi-token spans, not only single words.
        # A run that is *all* acronyms ("AI NLP") is two separate items, not one
        # entity, so require at least one ordinary Title Case word.
        _cores_preview = [c for c in (_split_affix(w)[1] for w in run) if c]
        as_entity = (
            len(run) >= 2
            and all(w[:1].isupper() for w in run if w[:1].isalpha())
            and any(classify_latin(c) == "CS" for c in _cores_preview)
            and rng.random() < entity_prob
        )
        groups = [run] if as_entity else [[w] for w in run]

        for grp in groups:
            lead, _, _ = _split_affix(grp[0])
            _, _, trail = _split_affix(grp[-1])
            cores = [c for c in (_split_affix(w)[1] for w in grp) if c]

            if not cores:  # punctuation-only group: copy it through
                for w in grp:
                    src.append(w)
                    tags.append(OUTSIDE)
                    tgt.append(w)
                continue

            cat = "ENTITY" if len(cores) > 1 else classify_latin(grp[0])

            span_start = len(src)
            pieces: list[str] = []
            for core in cores:
                # Within an ENTITY run, an acronym is still *spoken* letter by
                # letter ("Arabic NLP" -> ... ان ال بي), so each core keeps its
                # own pronunciation policy even though the span is one unit.
                piece_cat = classify_latin(core) if cat == "ENTITY" else cat
                pieces.extend(_render(core, piece_cat, rng))

            if not pieces:
                for w in grp:
                    src.append(w)
                    tags.append(OUTSIDE)
                    tgt.append(w)
                continue

            # Re-attach punctuation so the synthesised input reads naturally.
            pieces[0] = lead + pieces[0]
            pieces[-1] = pieces[-1] + trail

            for k, piece in enumerate(pieces):
                src.append(piece)
                tags.append(("B-" if k == 0 else "I-") + cat)

            tgt.append(lead + " ".join(cores) + trail)
            spans.append(
                {
                    "start": span_start,
                    "end": len(src),
                    "category": cat,
                    "target": " ".join(cores),
                }
            )
        i = run_end

    if not spans:
        return None
    return Example(src, tags, tgt, spans, dialect=dialect, source=source)


def passthrough_example(
    arabic_sentence: str, *, dialect: str = "unk", source: str = "unk"
) -> Example | None:
    """A pure-Arabic sentence: every tag is ``O``, output == input.

    These teach the single most important behaviour -- *do nothing* -- and are
    what make the model safe to drop into an existing pipeline.
    """
    s = arabic_sentence.strip()
    if not s or _LATIN.search(s) or not _ARABIC.search(s):
        return None
    toks = s.split()
    if not toks:
        return None
    return Example(
        toks, [OUTSIDE] * len(toks), list(toks), [], dialect=dialect, source=source
    )
