"""Canonical tag schema shared by the data pipeline, the model and inference.

The detector is a token tagger. Every token gets exactly one tag. The tag both
decides *whether* a token is rewritten and *how* stage 2 should rewrite it.

Design rule that makes the pass-through guarantee structural rather than learned:
``O`` is the identity tag. A token tagged ``O`` is copied to the output verbatim,
byte for byte. A fully-Arabic sentence is therefore all-``O`` and is returned
unchanged by construction -- no decoder ever sees it.
"""
from __future__ import annotations

# --- span categories -------------------------------------------------------
# Each category is a different *conversion policy* in stage 2, not just a label.
CATEGORIES = [
    "CS",       # transliterated English word/phrase:  انترن   -> intern
    "ACRONYM",  # letter-by-letter spelling:           ايه اي  -> AI
    "ENTITY",   # multi-word proper noun:  اورانج انوفيشن ايجيبت -> Orange Innovation Egypt
    "EMAIL",    # spoken email/URL:        ايه تي جيميل دوت كوم  -> @gmail.com
    "NUMBER",   # spoken digits/units:     تو تاوزند تwenty      -> 2020
]

OUTSIDE = "O"

# BIO tagging. B-/I- lets us keep adjacent-but-separate spans apart, which
# matters for ENTITY (a 3-word company name is ONE span, not three).
TAGS: list[str] = [OUTSIDE] + [f"{p}-{c}" for c in CATEGORIES for p in ("B", "I")]

TAG2ID: dict[str, int] = {t: i for i, t in enumerate(TAGS)}
ID2TAG: dict[int, str] = {i: t for t, i in TAG2ID.items()}

NUM_TAGS = len(TAGS)
OUTSIDE_ID = TAG2ID[OUTSIDE]

# Ignored positions in the loss (sub-word continuations, padding, specials).
IGNORE_INDEX = -100


def category_of(tag: str) -> str | None:
    """``"B-CS"`` -> ``"CS"``; ``"O"`` -> ``None``."""
    if tag == OUTSIDE:
        return None
    return tag.split("-", 1)[1]


def is_begin(tag: str) -> bool:
    return tag.startswith("B-")


def spans_from_tags(tags: list[str]) -> list[tuple[int, int, str]]:
    """Decode a BIO tag sequence into ``(start, end_exclusive, category)`` spans.

    Tolerant of malformed sequences: a stray ``I-X`` with no preceding ``B-X``
    opens a new span instead of being dropped, since at inference time the
    argmax output carries no guarantee of well-formedness.
    """
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    cat: str | None = None
    for i, tag in enumerate(tags):
        if tag == OUTSIDE:
            if start is not None:
                spans.append((start, i, cat))  # type: ignore[arg-type]
                start, cat = None, None
            continue
        c = category_of(tag)
        if is_begin(tag) or start is None or c != cat:
            if start is not None:
                spans.append((start, i, cat))  # type: ignore[arg-type]
            start, cat = i, c
    if start is not None:
        spans.append((start, len(tags), cat))  # type: ignore[arg-type]
    return spans
