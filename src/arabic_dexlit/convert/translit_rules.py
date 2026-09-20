"""Rule-based Arabic-script -> Latin transliteration.

This is the deterministic prior in the hybrid CS converter. Given ``فريندس`` it
produces ``frends`` -- phonetically right, orthographically wrong. On its own
that is not good enough, because transliteration is genuinely ambiguous:
``ميتينج`` maps equally well to ``meting`` or ``meeting``, and no table can
decide between them.

Its value is as *evidence*, not as an answer. The neural converter is given the
rule output alongside the raw span, so it starts from a phonetic skeleton rather
than from nothing, and the candidate ranker can score the rule output as one
hypothesis among several. Where the neural model is unsure, a phonetically
faithful fallback is still available.

It is also the inverse of :mod:`arabic_dexlit.data.translit`, which manufactures
training inputs by going English -> Arabic. Keeping both directions in the
project means the mapping is stated once, conceptually, and can be checked for
round-trip consistency.
"""
from __future__ import annotations

# Arabic grapheme -> Latin. Ordered longest-first so digraphs win.
# Several Arabic letters are genuinely ambiguous in this direction (ي is both
# "i" and "y"); the most common reading is chosen and the neural model corrects
# the rest.
_RULES: list[tuple[str, str]] = [
    # multi-character sequences first
    ("تش", "ch"),   # تش
    ("دج", "j"),    # دج
    ("كس", "x"),    # كس
    ("او", "ou"),   # او
    ("اي", "i"),    # اي
    # consonants
    ("ب", "b"), ("ت", "t"), ("ث", "th"), ("ج", "j"),
    ("ح", "h"), ("خ", "kh"), ("د", "d"), ("ذ", "th"),
    ("ر", "r"), ("ز", "z"), ("س", "s"), ("ش", "sh"),
    ("ص", "s"), ("ض", "d"), ("ط", "t"), ("ظ", "z"),
    ("ع", "a"), ("غ", "gh"), ("ف", "f"), ("ق", "q"),
    ("ك", "k"), ("ل", "l"), ("م", "m"), ("ن", "n"),
    ("ه", "h"), ("و", "o"), ("ي", "i"),
    # letters borrowed for foreign sounds
    ("پ", "p"), ("چ", "ch"), ("ڤ", "v"), ("گ", "g"),
    ("ژ", "zh"),
    # alef variants and hamza
    ("ا", "a"), ("أ", "a"), ("إ", "i"), ("آ", "a"),
    ("ى", "a"), ("ة", "a"), ("ء", ""), ("ئ", "i"),
    ("ؤ", "o"),
]

# Diacritics and the tatweel carry no information for this direction.
_STRIP = set("ًٌٍَُِّْـ")

_MAX_LEN = max(len(a) for a, _ in _RULES)
_MAP = dict(_RULES)


def arabic_to_latin(word: str) -> str:
    """Transliterate one Arabic-script word into a Latin skeleton.

    ``فريندس`` -> ``frends``. Greedy longest-match; unknown characters are
    dropped rather than passed through, so the result is always plain ASCII and
    safe to hand to a spell-corrector or a model.
    """
    text = "".join(c for c in word if c not in _STRIP)
    out: list[str] = []
    i = 0
    while i < len(text):
        for n in range(_MAX_LEN, 0, -1):
            chunk = text[i : i + n]
            if chunk in _MAP:
                out.append(_MAP[chunk])
                i += n
                break
        else:
            i += 1  # unmappable: skip
    result = "".join(out)

    # Collapse runs: Arabic marks gemination with shadda, so a doubled Latin
    # letter here is an artefact of the mapping rather than the pronunciation.
    collapsed: list[str] = []
    for ch in result:
        if collapsed and collapsed[-1] == ch:
            continue
        collapsed.append(ch)
    return "".join(collapsed)


def transliterate_span(tokens: list[str]) -> str:
    """Rule output for a whole span, preserving word boundaries."""
    return " ".join(arabic_to_latin(t) for t in tokens if t)
