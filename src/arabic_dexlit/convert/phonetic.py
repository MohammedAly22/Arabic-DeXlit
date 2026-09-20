"""Phonetic keys: match Arabic spellings to English words by *sound*.

Why this exists
---------------
A lookup table only answers forms it has already seen, and that is hopeless here.
Measured with the project's own transliterator, a single English word produces a
large family of Arabic spellings -- ``meeting`` yields 6, ``deadline`` 14,
``stakeholder`` 20, ``kubernetes`` 26 -- because real ASR is inconsistent about
vowels, emphatics and gemination. A gazetteer would need every variant.

A phonetic key collapses that family to one string. ``ميتينج``, ``ميتنج`` and
``ميطنغ`` all reduce to the same key, and so does ``meeting`` coming from the
English side. Matching on keys therefore generalises to spellings never seen:
the system recognises the *sound*, not the surface form.

This is the candidate generator for out-of-vocabulary spans. It proposes; the
reranker disposes.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

# --- Arabic side -----------------------------------------------------------
# Every Arabic grapheme maps to a coarse phonetic class. Distinctions that ASR
# does not reliably preserve are deliberately collapsed: س/ص both become S, ت/ط
# both T, since which one appears is an artefact of the transcriber.
_AR_CLASS: dict[str, str] = {
    "ب": "B", "پ": "B",                        # ب پ
    "ت": "T", "ط": "T", "ث": "T",         # ت ط ث
    "د": "D", "ض": "D", "ذ": "D",         # د ض ذ
    "س": "S", "ص": "S",                        # س ص
    "ز": "Z", "ظ": "Z", "ژ": "Z",         # ز ظ ژ
    "ك": "K", "ق": "K", "گ": "G",         # ك ق گ
    "ج": "J", "چ": "J",                        # ج چ
    "ش": "C",                                       # ش
    "ف": "F", "ڤ": "F",                        # ف ڤ
    "م": "M", "ن": "N", "ل": "L", "ر": "R",
    "ه": "H", "ح": "H", "خ": "K",         # ه ح خ
    "غ": "G", "ع": "",                         # غ ع (ع is silent here)
    "و": "W", "ي": "Y",
    "ا": "A", "أ": "A", "إ": "A", "آ": "A",
    "ى": "A", "ة": "A", "ء": "", "ئ": "Y", "ؤ": "W",
}

# --- English side ----------------------------------------------------------
# Digraphs first so "sh"/"ch"/"th" are not read as separate letters.
_EN_RULES: list[tuple[str, str]] = [
    ("tch", "J"), ("sch", "C"), ("sh", "C"), ("ch", "J"), ("ph", "F"),
    ("th", "T"), ("gh", "G"), ("ck", "K"), ("qu", "KW"), ("ng", "NG"),
    ("wh", "W"), ("kn", "N"), ("wr", "R"),
    ("a", "A"), ("e", "A"), ("i", "Y"), ("o", "W"), ("u", "W"), ("y", "Y"),
    ("b", "B"), ("c", "K"), ("d", "D"), ("f", "F"), ("g", "G"), ("h", "H"),
    ("j", "J"), ("k", "K"), ("l", "L"), ("m", "M"), ("n", "N"), ("p", "B"),
    ("q", "K"), ("r", "R"), ("s", "S"), ("t", "T"), ("v", "F"), ("w", "W"),
    ("x", "KS"), ("z", "Z"),
]
_EN_MAX = max(len(a) for a, _ in _EN_RULES)
_EN_MAP = dict(_EN_RULES)

_DIACRITICS = re.compile(r"[ً-ْـ]")


def _squeeze(key: str, drop_vowels: bool) -> str:
    """Collapse repeats and optionally drop vowel classes."""
    out: list[str] = []
    for ch in key:
        if drop_vowels and ch in "AWY":
            continue
        if out and out[-1] == ch:
            continue
        out.append(ch)
    return "".join(out)


def arabic_key(word: str, *, drop_vowels: bool = True) -> str:
    """Phonetic key for an Arabic-script word.

    ``ميتينج``, ``ميتنج`` and ``ميطنغ`` all collapse to the same key, which is
    what lets a match generalise across ASR spelling variation.
    """
    text = _DIACRITICS.sub("", word.strip())
    key = "".join(_AR_CLASS.get(c, "") for c in text)
    return _squeeze(key, drop_vowels)


def english_key(word: str, *, drop_vowels: bool = True) -> str:
    """Phonetic key for an English word, in the same space as ``arabic_key``."""
    w = re.sub(r"[^a-z]", "", word.lower())
    out: list[str] = []
    i = 0
    while i < len(w):
        for n in range(_EN_MAX, 0, -1):
            chunk = w[i : i + n]
            if chunk in _EN_MAP:
                out.append(_EN_MAP[chunk])
                i += n
                break
        else:
            i += 1
    return _squeeze("".join(out), drop_vowels)


def _edit_distance(a: str, b: str, cap: int) -> int:
    """Levenshtein distance, abandoned once it provably exceeds ``cap``."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            best = min(best, cur[-1])
        if best > cap:
            return cap + 1
        prev = cur
    return prev[-1]


class PhoneticIndex:
    """Maps phonetic keys to the English words that share them.

    Built from any English word list -- the training lexicon, a domain
    vocabulary, or an external dictionary -- so coverage is a property of the
    word list rather than of anything seen during training. That is what makes
    the converter general: a word never seen in an Arabic spelling can still be
    proposed, provided it is in the vocabulary at all.
    """

    def __init__(self, words: Iterable[str] | None = None) -> None:
        self.by_key: dict[str, list[str]] = defaultdict(list)
        self.by_key_v: dict[str, list[str]] = defaultdict(list)
        self.words: list[str] = []
        if words:
            self.add_many(words)

    def add(self, word: str) -> None:
        w = word.strip()
        if not w:
            return
        self.words.append(w)
        # Two indexes: one vowel-free (recall) and one vowel-aware (precision).
        for key, table in (
            (english_key(w, drop_vowels=True), self.by_key),
            (english_key(w, drop_vowels=False), self.by_key_v),
        ):
            if key and w not in table[key]:
                table[key].append(w)

    def add_many(self, words: Iterable[str]) -> None:
        for w in words:
            self.add(w)

    def __len__(self) -> int:
        return len(self.words)

    def _fuzzy(self, key: str, limit: int, max_dist: int) -> list[str]:
        """Nearest keys by edit distance, for spellings an exact key misses.

        ``اونبوردنج`` keys to ONBRDNJ while ``onboarding`` keys to ONBRDNG -- one
        substitution apart. Exact matching alone therefore loses words the index
        genuinely contains, so near keys are searched too.
        """
        if not key:
            return []
        # Bucket by length first: keys differing by more than max_dist in length
        # cannot be within max_dist, and this avoids scoring the whole index.
        out: list[tuple[int, str]] = []
        for cand_key, words in self.by_key.items():
            if abs(len(cand_key) - len(key)) > max_dist:
                continue
            d = _edit_distance(key, cand_key, max_dist)
            if d <= max_dist:
                for w in words:
                    out.append((d, w))
        out.sort(key=lambda x: (x[0], len(x[1])))
        seen: list[str] = []
        for _, w in out:
            if w not in seen:
                seen.append(w)
                if len(seen) >= limit:
                    break
        return seen

    def lookup(
        self, arabic: str, *, limit: int = 12, fuzzy: bool = True, max_dist: int = 1
    ) -> list[str]:
        """Candidate English words that sound like this Arabic form.

        The vowel-aware index is consulted first because it is more precise;
        the vowel-free index then widens recall for spellings that disagree
        about vowels, which is most of them.
        """
        out: list[str] = []
        for key, table in (
            (arabic_key(arabic, drop_vowels=False), self.by_key_v),
            (arabic_key(arabic, drop_vowels=True), self.by_key),
        ):
            for w in table.get(key, []):
                if w not in out:
                    out.append(w)
                    if len(out) >= limit:
                        return out

        # Exact keys are strict: one substitution is enough to miss a word the
        # index actually holds. Widen only when nothing exact was found.
        if fuzzy and not out:
            out = self._fuzzy(arabic_key(arabic, drop_vowels=True), limit, max_dist)
        return out
