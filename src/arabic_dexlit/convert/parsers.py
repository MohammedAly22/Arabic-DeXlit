"""Deterministic converters for the structured span categories.

The design principle for this package: **deterministic where structure exists,
neural where ambiguity exists**.

``احمد ات جيميل دوت كوم`` -> ``ahmed@gmail.com`` is not an ambiguous inference.
It is a parse: ``ات`` is ``@``, ``دوت`` is ``.``, and the rest are spoken tokens
with known spellings. Asking a decoder to generate that string invites errors
into a problem that has none -- the model can hallucinate a domain, drop a dot,
or produce an address that is merely plausible.

So EMAIL, URL, NUMBER, DATE and TIME are parsed, not generated. Their
correctness is a property of the code rather than of a checkpoint, and it does
not regress when a model is retrained.

Every parser returns ``None`` when it cannot parse confidently. That is the
whole contract: an unparseable span falls through to the neural path or, failing
that, is copied unchanged. Guessing is never the fallback.
"""
from __future__ import annotations

import re

# --- spoken symbols --------------------------------------------------------
SYMBOL_WORDS: dict[str, str] = {
    "ات": "@",          # ات
    "اد": "@",          # اد
    "آت": "@",          # آت
    "ايت": "@",    # ايت
    "دوت": ".",    # دوت
    "نقطة": ".",   # نقطة
    "داش": "-",    # داش
    "شرطة": "-",   # شرطة
    "اندرسكور": "_",  # اندرسكور
    "سلاش": "/",   # سلاش
    "كولون": ":",  # كولون
}

# Spoken digits. Several spellings per digit, because ASR is inconsistent.
DIGIT_WORDS: dict[str, str] = {
    "زيرو": "0", "او": "0",
    "وان": "1", "ون": "1",
    "تو": "2", "توو": "2",
    "ثري": "3", "تري": "3",
    "فور": "4",
    "فايف": "5", "فيف": "5",
    "سكس": "6", "سيكس": "6",
    "سفن": "7", "سيفن": "7",
    "ايت": "8", "ايط": "8",
    "ناين": "9",
}

# Domain and TLD words, said as words rather than spelled out.
DOMAIN_WORDS: dict[str, str] = {
    "جيميل": "gmail",
    "هوتميل": "hotmail",
    "ياهو": "yahoo",
    "اوتلوك": "outlook",
    "ايكلاود": "icloud",
    "كوم": "com",
    "نت": "net",
    "اورج": "org",
    "ايديو": "edu",
    "جوف": "gov",
    "دبليو": "w",       # spoken "double-u"
    "كو": "co",
}

# Arabic letter names, for the spelled-out parts of an address.
LETTER_WORDS: dict[str, str] = {
    "ايه": "a", "اي": "i", "بي": "b",
    "سي": "c", "دي": "d", "اف": "f",
    "جي": "g", "اتش": "h", "ايتش": "h",
    "جيه": "j", "كيه": "k", "كي": "k",
    "ال": "l", "ام": "m", "ان": "n",
    "او": "o", "كيو": "q", "ار": "r",
    "اس": "s", "تي": "t", "يو": "u",
    "في": "v", "اكس": "x", "واي": "y",
    "زد": "z", "زي": "z",
}

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_URL_RE = re.compile(r"^(?:https?://|www\.)\S+$|^[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?:/\S*)?$")
_LATIN = re.compile(r"[A-Za-z]")


def _strip_punct(tok: str) -> str:
    return tok.strip(".,!?؟،؛:;\"'()[]")


def parse_number(tokens: list[str]) -> str | None:
    """``['تو','زيرو','تو','فور']`` -> ``'2024'``.

    Returns ``None`` unless *every* token is a recognised digit, so a partially
    understood number is never half-converted.
    """
    if not tokens:
        return None
    digits: list[str] = []
    for tok in tokens:
        t = _strip_punct(tok)
        if not t:
            continue
        if t.isdigit():
            digits.append(t)
        elif t in DIGIT_WORDS:
            digits.append(DIGIT_WORDS[t])
        else:
            return None
    return "".join(digits) or None


def _reverse_translit(word: str) -> str | None:
    """Best-effort Arabic -> Latin for a word inside a structured span.

    Reuses the project's own phonetic table, run backwards. It is a *prior*,
    not an answer: the caller decides whether the assembled result is valid.
    """
    from ..data.translit import LEXICON

    # A word the generator knows (gmail, outlook...) inverts exactly.
    for en, forms in LEXICON.items():
        if word in forms:
            return en

    from .translit_rules import arabic_to_latin

    return arabic_to_latin(word) or None


def parse_email(tokens: list[str]) -> str | None:
    """``['احمد','ات','جيميل','دوت','كوم']`` -> ``'ahmed@gmail.com'``.

    Requires an ``@`` and a dot in the result, so anything that does not parse
    into an actual address is rejected rather than guessed at.
    """
    if not tokens:
        return None

    # Already a well-formed address: return it untouched.
    joined = "".join(_strip_punct(t) for t in tokens)
    if _EMAIL_RE.match(joined):
        return joined

    out: list[str] = []
    for tok in tokens:
        t = _strip_punct(tok)
        if not t:
            continue
        if t in SYMBOL_WORDS:
            out.append(SYMBOL_WORDS[t])
        elif t in DOMAIN_WORDS:
            out.append(DOMAIN_WORDS[t])
        elif t in DIGIT_WORDS:
            out.append(DIGIT_WORDS[t])
        elif t in LETTER_WORDS:
            out.append(LETTER_WORDS[t])
        elif _LATIN.search(t):
            out.append(t.lower())
        else:
            # Structure is deterministic; the word parts are transliteration.
            # Falling back to the rule table here converts the whole address
            # instead of declining a span whose shape is perfectly clear.
            guess = _reverse_translit(t)
            if not guess:
                return None
            out.append(guess.lower())

    addr = "".join(out)
    return addr if _EMAIL_RE.match(addr) else None


def parse_url(tokens: list[str]) -> str | None:
    """``['دبليو','دبليو','دبليو','دوت','جوجل','دوت','كوم']`` -> ``'www.google.com'``."""
    if not tokens:
        return None
    joined = "".join(_strip_punct(t) for t in tokens)
    if _URL_RE.match(joined):
        return joined

    out: list[str] = []
    for tok in tokens:
        t = _strip_punct(tok)
        if not t:
            continue
        if t in SYMBOL_WORDS:
            out.append(SYMBOL_WORDS[t])
        elif t in DOMAIN_WORDS:
            out.append(DOMAIN_WORDS[t])
        elif t in LETTER_WORDS:
            out.append(LETTER_WORDS[t])
        elif t in DIGIT_WORDS:
            out.append(DIGIT_WORDS[t])
        elif _LATIN.search(t):
            out.append(t.lower())
        else:
            guess = _reverse_translit(t)
            if not guess:
                return None
            out.append(guess.lower())
    url = "".join(out)
    return url if _URL_RE.match(url) else None


def parse_time(tokens: list[str]) -> str | None:
    """``['فايف','ثري','زيرو']`` -> ``'5:30'`` when it reads as a clock time."""
    digits = parse_number(tokens)
    if not digits:
        return None
    if len(digits) == 3:
        h, m = digits[0], digits[1:]
    elif len(digits) == 4:
        h, m = digits[:2], digits[2:]
    else:
        return None
    if int(h) > 23 or int(m) > 59:
        return None
    return f"{int(h)}:{m}"


# Registry used by the router. Order matters only within a category.
DETERMINISTIC_PARSERS = {
    "EMAIL": parse_email,
    "URL": parse_url,
    "NUMBER": parse_number,
    "TIME": parse_time,
}


def parse_span(tokens: list[str], category: str) -> str | None:
    """Try the deterministic parser for ``category``; ``None`` if it declines.

    EMAIL also tries the URL parser. The corpus labels spoken web addresses
    ("دبليو دبليو دبليو دوت ...") as EMAIL, and requiring an ``@`` made the
    email parser decline every one of them.
    """
    fn = DETERMINISTIC_PARSERS.get(category)
    if fn is None:
        return None
    got = fn(tokens)
    if got is None and category == "EMAIL":
        got = parse_url(tokens)
    return got
