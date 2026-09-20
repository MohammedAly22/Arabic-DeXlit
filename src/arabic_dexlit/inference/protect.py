"""Deterministic protection layer -- runs *before* the neural detector.

Some tokens must never be altered, and their identity is decidable by
inspection rather than by a model: an email address, a URL, an IP, a decimal
number, a year, punctuation, or English that already arrived in Latin script.
Sending those through a learned model buys nothing and risks everything --
``test@gmail.com`` has exactly one correct output, and a detector that flags it
can only make it worse.

So they are recognised by rule and pinned to ``O`` before inference, which turns
"probably preserved" into "cannot be modified". The neural detector then only
sees the genuinely ambiguous tokens, which is also what it was trained on.

The ordering matters and is deliberate:

    protect (rules)  ->  detect (neural)  ->  convert (neural)  ->  rebuild

A pinned token is removed from the model's decision space entirely; it is not
merely down-weighted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# --- patterns --------------------------------------------------------------
# Deliberately strict: a false positive here silently disables conversion for a
# token that needed it, which is worse than letting the detector decide.
_EMAIL = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_URL = re.compile(
    r"^(?:https?://|www\.)\S+$"
    r"|^[A-Za-z0-9.\-]+\.(?:com|org|net|edu|gov|io|ai|co|ly|me|eg|sa|ae)(?:/\S*)?$",
    re.IGNORECASE,
)
_IP = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_NUMERIC = re.compile(r"^[+\-]?\d+(?:[.,:]\d+)*%?$")
_TIME = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")
_DATE = re.compile(r"^\d{1,4}[/\-]\d{1,2}(?:[/\-]\d{1,4})?$")
_HASHTAG = re.compile(r"^[@#]\w+$")
_FILE = re.compile(r"^\S+\.(?:pdf|docx?|xlsx?|pptx?|csv|json|png|jpe?g|mp[34]|zip)$", re.I)
_PUNCT_ONLY = re.compile(r"^[^\w]+$", re.UNICODE)
_LATIN = re.compile(r"[A-Za-z]")
_ARABIC = re.compile(r"[؀-ۿ]")

# Why each token was protected. Surfaced in the prediction so a user can see
# that a rule -- not the model -- decided.
PROTECT_REASONS = (
    "email", "url", "ip", "number", "time", "date", "handle", "file",
    "punctuation", "latin",
)


@dataclass(frozen=True)
class Protection:
    index: int
    token: str
    reason: str


def classify_token(token: str, *, protect_latin: bool = False) -> str | None:
    """Return why ``token`` is protected, or ``None`` if the model should decide."""
    t = token.strip()
    if not t:
        return None
    # Strip trailing sentence punctuation before matching, so "3.14," still
    # reads as a number, but keep it attached for the punctuation-only test.
    core = t.rstrip(".,!?؟،؛:;\"')]}")
    if not core:
        return "punctuation"
    if _EMAIL.match(core):
        return "email"
    if _URL.match(core):
        return "url"
    if _IP.match(core):
        return "ip"
    if _TIME.match(core):
        return "time"
    if _DATE.match(core):
        return "date"
    if _FILE.match(core):
        return "file"
    if _HASHTAG.match(core):
        return "handle"
    if _NUMERIC.match(core):
        return "number"
    if _PUNCT_ONLY.match(t):
        return "punctuation"
    # Already-Latin English. Off by default: the corpus deliberately teaches the
    # model to copy such tokens, and protecting them here would mask a real
    # regression in that behaviour. Turn on for maximum safety in production.
    if protect_latin and _LATIN.search(core) and not _ARABIC.search(core):
        return "latin"
    return None


def protect(tokens: list[str], *, protect_latin: bool = False) -> list[Protection]:
    """Find every token a rule can decide, so the model never sees it."""
    out: list[Protection] = []
    for i, tok in enumerate(tokens):
        reason = classify_token(tok, protect_latin=protect_latin)
        if reason is not None:
            out.append(Protection(i, tok, reason))
    return out


def protected_mask(tokens: list[str], *, protect_latin: bool = False) -> list[bool]:
    """``True`` where a token is rule-protected and must be copied verbatim."""
    mask = [False] * len(tokens)
    for p in protect(tokens, protect_latin=protect_latin):
        mask[p.index] = True
    return mask
