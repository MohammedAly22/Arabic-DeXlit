"""Reject transliterations that collide with real Arabic words.

The problem
-----------
Short English words transliterate into strings that *are* ordinary Arabic words.
Measured on the corpus, this affected **9% of all training spans**:

===========  ==========================  ===========================
Arabic form  generated as                actually
===========  ==========================  ===========================
``مال``      ``mall``  (4,774 spans)     "money"
``ذي``       ``the``   (3,041 spans)     a real demonstrative
``دي``       ``day``                     Egyptian "this" (457 O-tagged)
``ان``       ``in``    (3,064 spans)     "that"
===========  ==========================  ===========================

The corpus therefore taught two contradictory things about the same string:
convert it here, leave it alone there. No model can satisfy both, and the
failure shows up as confident wrong edits on ordinary Arabic -- the single worst
behaviour for an ASR post-processor.

The fix
-------
When the transliterator produces a form that is a common standalone Arabic word,
that rendering is rejected and another sampled. The blocklist is derived from a
monolingual Arabic corpus -- text with no code-switching by construction -- so
it reflects actual usage rather than intuition.

Long forms are exempt: a collision only matters when the string is short enough
to be a real word, and a seven-letter transliteration that happens to appear in
Arabic text is almost certainly a genuine borrowing.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

# Collisions only bite for short strings. Longer transliterations that appear in
# Arabic text are overwhelmingly real borrowings (تكنولوجيا, انترنت) rather than
# accidents, and blocking those would lose genuine training signal.
MAX_COLLISION_LEN = 5

# A form must appear at least this often in monolingual Arabic to count as a
# real word rather than a typo or a stray token.
MIN_ARABIC_COUNT = 15

# Known-bad forms seen in the corpus audit, kept explicitly so the guard works
# even when no monolingual file is present.
SEED_COLLISIONS: frozenset[str] = frozenset(
    {
        "مال",      # مال  "money"        <- mall
        "ذي",            # ذي   demonstrative  <- the
        "دي",            # دي   "this"         <- day
        "ان",            # ان   "that"         <- in
        "نو",            # نو                  <- no
        "مي",            # مي                  <- my / me
        "اند",      # اند                 <- and
        "نيو",      # نيو                 <- new
        "جو",            # جو   "weather"      <- go
        "تي",            # تي                  <- tea / T
        "المال",  # المال "the money"
        "الا",      # الا                 <- a / I
        "ا",                  # ا                   <- I / a
        "تو",            # تو                  <- to / two
        "لا",            # لا   "no"
        "هو",            # هو   "he"
        "هي",            # هي   "she"
        "في",            # في   "in"
        "من",            # من   "from"
        "عن",            # عن   "about"
        "كان",      # كان  "was"
        "بس",            # بس   "but"
        "او",            # او   "or"
    }
)


@lru_cache(maxsize=1)
def arabic_word_set(
    mono_path: str = "data/raw/monolingual.txt",
    limit: int = 400_000,
) -> frozenset[str]:
    """Common short words from monolingual Arabic, plus the seed list.

    Built from text that contains no code-switching, so every entry is a word
    Arabic speakers actually write -- which is exactly the property that makes a
    transliteration colliding with it unsafe.
    """
    words: set[str] = set(SEED_COLLISIONS)
    path = Path(mono_path)
    if path.exists():
        counts: Counter = Counter()
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= limit:
                    break
                counts.update(line.split())
        words |= {
            w
            for w, n in counts.items()
            if n >= MIN_ARABIC_COUNT and len(w) <= MAX_COLLISION_LEN
        }
    return frozenset(words)


def collides(form: str, *, mono_path: str = "data/raw/monolingual.txt") -> bool:
    """True if this transliteration would be indistinguishable from Arabic."""
    f = form.strip()
    if not f or len(f) > MAX_COLLISION_LEN:
        return False
    return f in arabic_word_set(mono_path)
