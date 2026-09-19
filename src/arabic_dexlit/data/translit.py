"""English -> Arabic-script transliteration, used to *manufacture* training inputs.

Why this exists
---------------
Parallel data for this task is scarce, but corpora of the *target* form (Arabic
with English kept in Latin script) are plentiful -- e.g. SDAIA ArE-CSTD, 330K
sentences. This module inverts them: given the target, it synthesises the
ASR-style input by rewriting the English words into Arabic letters, exactly the
failure mode we want the model to undo. One clean corpus becomes aligned pairs.

Fidelity matters more than linguistic purity here. A real Arabic ASR system is
inconsistent -- it renders the same English word several different ways
depending on speaker and acoustic context. So ``transliterate`` is deliberately
*stochastic*: each call samples from the plausible renderings. Training over
multiple epochs therefore exposes the model to the natural variation it will
meet at inference, instead of a single canonical spelling it could memorise.
"""
from __future__ import annotations

import random
import re

# --- grapheme mapping ------------------------------------------------------
# Ordered longest-first; matching is greedy so digraphs win over single letters.
# Values are weighted alternatives: (arabic_form, weight).
_GRAPHEMES: list[tuple[str, list[tuple[str, float]]]] = [
    # digraphs / trigraphs
    ("tch", [("تش", 1.0)]),
    ("sch", [("ش", 0.6), ("سك", 0.4)]),
    ("ch",  [("تش", 0.7), ("ش", 0.3)]),
    ("sh",  [("ش", 1.0)]),
    ("th",  [("ث", 0.5), ("ذ", 0.3), ("ت", 0.2)]),
    ("ph",  [("ف", 1.0)]),
    ("gh",  [("ج", 0.5), ("غ", 0.5)]),
    ("ck",  [("ك", 1.0)]),
    ("qu",  [("كو", 0.7), ("كي", 0.3)]),
    ("ng",  [("نج", 0.6), ("نغ", 0.4)]),
    ("oo",  [("و", 0.8), ("و", 0.2)]),
    ("ee",  [("ي", 1.0)]),
    ("ea",  [("ي", 0.7), ("ie", 0.0), ("يا", 0.3)]),
    ("ou",  [("او", 0.6), ("و", 0.4)]),
    ("ow",  [("او", 0.7), ("و", 0.3)]),
    ("ai",  [("ي", 0.5), ("اي", 0.5)]),
    ("ay",  [("ي", 0.5), ("اي", 0.5)]),
    ("oa",  [("و", 1.0)]),
    ("au",  [("و", 0.6), ("او", 0.4)]),
    ("ie",  [("ي", 1.0)]),
    ("ei",  [("ي", 0.6), ("اي", 0.4)]),
    # single consonants
    ("b", [("ب", 1.0)]),
    ("c", [("ك", 0.8), ("س", 0.2)]),
    ("d", [("د", 0.95), ("ض", 0.05)]),
    ("f", [("ف", 1.0)]),
    ("g", [("ج", 0.75), ("ق", 0.10), ("غ", 0.15)]),
    ("h", [("ه", 0.85), ("ح", 0.15)]),
    ("j", [("ج", 1.0)]),
    ("k", [("ك", 1.0)]),
    ("l", [("ل", 1.0)]),
    ("m", [("م", 1.0)]),
    ("n", [("ن", 1.0)]),
    ("p", [("ب", 1.0)]),
    ("q", [("ك", 0.6), ("ق", 0.4)]),
    ("r", [("ر", 1.0)]),
    ("s", [("س", 0.94), ("ص", 0.06)]),
    ("t", [("ت", 0.94), ("ط", 0.06)]),
    ("v", [("ف", 1.0)]),
    ("w", [("و", 1.0)]),
    ("x", [("كس", 1.0)]),
    ("y", [("ي", 1.0)]),
    ("z", [("ز", 0.94), ("ذ", 0.06)]),
    # vowels
    ("a", [("ا", 0.80), ("", 0.20)]),
    ("e", [("ي", 0.45), ("", 0.40), ("ا", 0.15)]),
    ("i", [("ي", 0.75), ("", 0.25)]),
    ("o", [("و", 0.88), ("", 0.12)]),
    ("u", [("و", 0.72), ("ا", 0.16), ("", 0.12)]),
]

_MAX_GRAPHEME = max(len(g) for g, _ in _GRAPHEMES)
_GRAPHEME_MAP = dict(_GRAPHEMES)

# Letter names, for acronyms spelled out loud: AI -> ايه اي
LETTER_NAMES: dict[str, list[str]] = {
    "a": ["ايه", "اي"], "b": ["بي"], "c": ["سي"], "d": ["دي"], "e": ["اي"],
    "f": ["اف"], "g": ["جي"], "h": ["اتش", "ايتش"], "i": ["اي"], "j": ["جيه"],
    "k": ["كيه", "كي"], "l": ["ال"], "m": ["ام"], "n": ["ان"], "o": ["او"],
    "p": ["بي"], "q": ["كيو"], "r": ["ار"], "s": ["اس"], "t": ["تي"],
    "u": ["يو"], "v": ["في"], "w": ["دبليو"], "x": ["اكس"], "y": ["واي"],
    "z": ["زد", "زي"],
}

# Spoken digits.
DIGIT_NAMES: dict[str, list[str]] = {
    "0": ["زيرو"], "1": ["وان"], "2": ["تو"], "3": ["ثري"], "4": ["فور"],
    "5": ["فايف"], "6": ["سكس"], "7": ["سفن"], "8": ["ايت"], "9": ["ناين"],
}

# Symbols that get spoken aloud inside emails and URLs.
SYMBOL_NAMES: dict[str, list[str]] = {
    "@": ["ات", "اد"], ".": ["دوت"], "-": ["داش"], "_": ["اندرسكور"],
    "/": ["سلاش"], ":": ["كولون"],
}


# Words a speaker says as a unit rather than sounding out letter by letter.
# Without these, greedy grapheme mapping turns "gmail" into gibberish.
LEXICON: dict[str, list[str]] = {
    "gmail": ["جيميل"], "hotmail": ["هوتميل"], "yahoo": ["ياهو"],
    "outlook": ["اوتلوك"], "com": ["كوم"], "net": ["نت"], "org": ["اورج"],
    "www": ["دبليو دبليو دبليو"], "co": ["كو"], "edu": ["ايديو"],
    "google": ["جوجل"], "facebook": ["فيسبوك"], "whatsapp": ["واتساب"],
    "youtube": ["يوتيوب"], "linkedin": ["لينكد ان"], "icloud": ["اي كلاود"],
}


def _choose(alts: list[tuple[str, float]], rng: random.Random) -> str:
    forms = [a for a, _ in alts]
    weights = [w for _, w in alts]
    return rng.choices(forms, weights=weights, k=1)[0]


def transliterate_word(word: str, rng: random.Random | None = None) -> str:
    """Render a single English word in Arabic script, sampling among variants."""
    rng = rng or random
    w = word.lower()
    if w in LEXICON:
        return rng.choice(LEXICON[w])
    out: list[str] = []
    i = 0
    first = True
    while i < len(w):
        for n in range(_MAX_GRAPHEME, 0, -1):
            chunk = w[i : i + n]
            if chunk in _GRAPHEME_MAP:
                form = _choose(_GRAPHEME_MAP[chunk], rng)
                # Arabic cannot begin a word with a bare vowel letter: an
                # initial vowel is carried by alef. So "intern" surfaces as
                # انترن (not ينترن) and "offer" as اوفر (not وفر).
                if first and chunk[0] in "aeiou":
                    if not form:
                        form = "ا"
                    elif form[0] == "ي" and chunk == "i":
                        form = "ا"          # initial short /i/ -> انترن
                    elif form[0] in "يو":
                        form = "ا" + form
                    elif form[0] != "ا":
                        form = "ا" + form
                out.append(form)
                i += n
                first = False
                break
        else:
            i += 1  # unmappable character (punctuation inside a token): skip
    res = "".join(out)
    # Arabic marks gemination with shadda, not a repeated letter, so ASR output
    # shows "بيزا" rather than "بيززا". Collapse runs introduced by English
    # double consonants.
    collapsed: list[str] = []
    for ch in res:
        if collapsed and collapsed[-1] == ch and ch not in "اوي":
            continue
        collapsed.append(ch)
    res = "".join(collapsed)
    return res or "ا"


def spell_acronym(acr: str, rng: random.Random | None = None) -> str:
    """``"AI"`` -> ``"ايه اي"`` -- letter names, as a speaker would say them."""
    rng = rng or random
    parts = []
    for ch in acr.lower():
        if ch.isalpha() and ch in LETTER_NAMES:
            parts.append(rng.choice(LETTER_NAMES[ch]))
        elif ch.isdigit():
            parts.append(rng.choice(DIGIT_NAMES[ch]))
    return " ".join(parts)


def speak_number(num: str, rng: random.Random | None = None) -> str:
    """Digit-by-digit reading: ``"2024"`` -> ``"تو زيرو تو فور"``."""
    rng = rng or random
    return " ".join(rng.choice(DIGIT_NAMES[d]) for d in num if d.isdigit())


def speak_email(addr: str, rng: random.Random | None = None) -> str:
    """``"ali@gmail.com"`` -> ``"ايه ال اي ات جيميل دوت كوم"``.

    Local parts are usually spelled letter-by-letter while domains are said as
    words -- which is how people actually dictate an address.
    """
    rng = rng or random
    out: list[str] = []
    for tok in re.findall(r"[A-Za-z]+|\d+|[@._\-/:]", addr):
        if tok in SYMBOL_NAMES:
            out.append(rng.choice(SYMBOL_NAMES[tok]))
        elif tok.isdigit():
            out.append(speak_number(tok, rng))
        elif tok.lower() in LEXICON:
            out.append(rng.choice(LEXICON[tok.lower()]))
        elif len(tok) <= 3 or tok.isupper():
            out.append(spell_acronym(tok, rng))
        else:
            out.append(transliterate_word(tok, rng))
    return " ".join(out)
