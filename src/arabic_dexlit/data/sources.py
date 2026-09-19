"""Corpus acquisition.

Each source yields ``(target_sentence, dialect_code, source_name)``. The
*target* is Arabic with English in Latin script -- the form the model must
produce. Inputs are manufactured later by :mod:`arabic_dexlit.data.pairing`.

Licensing note
--------------
Only sources that permit redistribution of derivatives feed the published
dataset:

* **SDAIA ArE-CSTD** (CC-BY-SA-4.0) -- redistributable with share-alike, so the
  released corpus inherits CC-BY-SA.
* **Gemini synthetic** -- generated here, no upstream restriction.
* **Casablanca** (CC-BY-**NC-ND**-4.0) -- *no-derivatives*, so transliterating it
  into training pairs and republishing them is not permitted. It is therefore
  available as a held-out evaluation set only, cited rather than redistributed,
  and is excluded from the released corpus by default.

Monolingual Arabic for pass-through examples is drawn from the *non*
code-switched portion of these corpora plus optional external monolingual text;
these teach the model to leave ordinary Arabic alone.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator

_LATIN = re.compile(r"[A-Za-z]")
_SENT_SPLIT = re.compile(r"(?<=[.!?؟۔])\s+|\n+")
_ARABIC = re.compile(r"[؀-ۿ]")

# --- SDAIA ArE-CSTD --------------------------------------------------------
SDAIA_REPO = "SDAIANCAI/Ar-En-Code-Switching-Textual-Dataset"
SDAIA_FILES: dict[str, tuple[str, str]] = {
    # filename -> (dialect code, split hint)
    "MSA_TRAIN.txt": ("msa", "train"),
    "MSA_TEST.txt": ("msa", "test"),
    "EGY_TRAIN.txt": ("egy", "train"),
    "EGY_TEST.txt": ("egy", "test"),
    "SA_TRAIN.txt": ("glf", "train"),
    "SA_TEST.txt": ("glf", "test"),
}
_HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{name}"


def download(repo: str, name: str, dest_dir: Path, *, force: bool = False) -> Path:
    """Fetch one file from a public HF dataset repo, caching on disk."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if dest.exists() and not force and dest.stat().st_size > 0:
        return dest
    url = _HF_RESOLVE.format(repo=repo, name=name)
    with urllib.request.urlopen(url, timeout=300) as r, dest.open("wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    return dest


def iter_sdaia(
    raw_dir: str | Path = "data/raw/sdaia", *, splits: tuple[str, ...] = ("train", "test")
) -> Iterator[tuple[str, str, str]]:
    """Yield SDAIA sentences, downloading on first use."""
    raw_dir = Path(raw_dir)
    for name, (dialect, split) in SDAIA_FILES.items():
        if split not in splits:
            continue
        path = download(SDAIA_REPO, name, raw_dir)
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s:
                    yield s, dialect, "sdaia"


def iter_synthetic(path: str | Path = "data/raw/synthetic.jsonl") -> Iterator[tuple[str, str, str]]:
    """Yield sentences produced by :mod:`arabic_dexlit.data.synth`."""
    path = Path(path)
    if not path.exists():
        return
    # Read as bytes and decode per line: a single damaged line (from an
    # interrupted or concurrent write) must not abort the whole corpus.
    with path.open("rb") as fh:
        for raw in fh:
            try:
                row = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            text = (row.get("text") or "").strip()
            if text:
                yield text, row.get("dialect", "unk"), "gemini"


# Arabic renderings of common English words. A "monolingual" sentence that
# contains one of these is *already* transliterated code-switching, so using it
# as a pass-through example would teach the model to leave transliteration
# alone -- precisely the behaviour we are trying to remove. Screening on a
# borrowed-word list is crude but catches the frequent offenders, and the cost
# of a false positive is only a discarded sentence.
# English words that Arabic speakers most often borrow. The Arabic surface
# forms are *generated* from these by the project's own transliterator, so the
# screen covers the spelling variants ASR really produces without anyone having
# to enumerate them by hand.
_BORROWED_EN = [
    "meeting", "smart", "phone", "internet", "computer", "laptop", "mobile",
    "online", "email", "manager", "project", "team", "offer", "system",
    "application", "program", "okay", "course", "gym", "climate", "change",
    "social", "media", "video", "password", "account", "marketing", "server",
    "data", "intern", "report", "shopping", "weekend", "sandwich", "ticket",
    "message", "camera", "battery", "screen", "file", "link", "update",
    "download", "upload", "network", "website", "business", "office", "doctor",
    "hospital", "delivery", "order", "customer", "service", "support",
    "engineer", "design", "developer", "budget", "presentation", "schedule",
]

# Naturalised borrowings whose Arabic spelling has settled into a conventional
# form the phonetic transliterator does not reproduce (it yields كلايمات, not
# the actual كلايمت). Generation covers the open-ended tail; these cover the
# frequent, fixed spellings.
_BORROWED_FIXED = [
    "الإنترنت", "إنترنت", "انترنت",
    "كلايمت", "تشينج", "تشينج",
    "الويكند", "ويكند",
    "الشوبينج", "شوبينج",
    "ميتينغ", "ميتينج",
    "سمارت", "موبايل", "لابتوب",
    "أونلاين", "اونلاين",
    "إيميل", "ايميل",
    "ساندويتش", "ساندويتشات",
    "سوشيال", "ميديا",
    "تيكنولوجيا",
]

_BORROWED_CACHE: set[str] | None = None


def _borrowed_forms() -> set[str]:
    """Arabic spellings of the loanwords, generated once and cached.

    The transliterator is stochastic, so each word is sampled several times to
    collect its plausible renderings.
    """
    global _BORROWED_CACHE
    if _BORROWED_CACHE is None:
        import random as _random

        from .translit import transliterate_word

        rng = _random.Random(0)
        forms: set[str] = set(_BORROWED_FIXED)
        for word in _BORROWED_EN:
            for _ in range(12):
                f = transliterate_word(word, rng)
                if len(f) >= 4:  # very short forms collide with real Arabic
                    forms.add(f)
        _BORROWED_CACHE = forms
    return _BORROWED_CACHE


def looks_transliterated(sentence: str) -> bool:
    """True if the sentence appears to contain English written in Arabic script.

    A hand-written blocklist does not scale -- the vocabulary of borrowings is
    open-ended, and hits like الشوبينج / الويكند keep appearing. So the check is
    generative instead: the same transliterator that manufactures training
    inputs is run over a list of frequent English loanwords, producing the
    Arabic forms an ASR system would actually emit, and those are matched
    against the sentence. Extending coverage then means adding an English word,
    not enumerating its Arabic spellings.
    """
    return any(form in sentence for form in _borrowed_forms())


def is_mostly_arabic(sentence: str, threshold: float = 0.7) -> bool:
    """True if Arabic letters dominate the sentence's letters.

    Checking merely for *some* Arabic lets text in other scripts through -- a
    Hebrew sentence with one Arabic character was observed passing the naive
    test -- so the share of Arabic among all letters is what is measured.
    """
    letters = [c for c in sentence if c.isalpha()]
    if not letters:
        return False
    arabic = sum(1 for c in letters if "؀" <= c <= "ۿ")
    return arabic / len(letters) >= threshold


def iter_hf_monolingual(
    repo: str,
    config: str,
    split: str,
    text_field: str,
    *,
    dialect: str = "unk",
    source: str | None = None,
    limit: int = 200_000,
    sentence_split: bool = True,
) -> Iterator[tuple[str, str, str]]:
    """Stream monolingual Arabic from a HF dataset via the datasets-server API.

    Uses the paginated ``/rows`` endpoint rather than the ``datasets`` library so
    that pass-through data can be fetched without materialising a multi-GB
    download -- a few hundred thousand sentences is all that is needed, and on
    Colab the disk and time saved matter.
    """
    source = source or repo.split("/")[-1]
    base = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={urllib.parse.quote(repo, safe='')}"
        f"&config={urllib.parse.quote(config, safe='')}"
        f"&split={urllib.parse.quote(split, safe='')}"
    )
    n = 0
    offset = 0
    page = 100
    while n < limit:
        try:
            with urllib.request.urlopen(f"{base}&offset={offset}&length={page}", timeout=120) as r:
                payload = json.load(r)
        except Exception:
            return
        rows = payload.get("rows") or []
        if not rows:
            return
        for item in rows:
            text = (item.get("row") or {}).get(text_field) or ""
            if not text:
                continue
            # Wikipedia rows are whole articles; split into sentences so the
            # pass-through examples resemble ASR utterances in length.
            chunks = _SENT_SPLIT.split(text) if sentence_split else [text]
            for chunk in chunks:
                c = chunk.strip()
                if 4 <= len(c.split()) <= 40:
                    yield c, dialect, source
                    n += 1
                    if n >= limit:
                        return
        offset += page


def iter_monolingual(
    paths: list[str | Path] | None = None,
    *,
    sdaia_dir: str | Path = "data/raw/sdaia",
    limit: int | None = None,
    screen_transliterated: bool = True,
) -> Iterator[tuple[str, str, str]]:
    """Yield pure-Arabic sentences for pass-through training.

    Two filters apply. Sentences with any Latin character are rejected outright.
    Sentences that *look* transliterated are rejected too -- see
    :func:`looks_transliterated` -- because a pass-through example must be
    genuinely monolingual, or it teaches exactly the wrong lesson.
    """
    n = 0
    for path in paths or []:
        p = Path(path)
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s or _LATIN.search(s) or not is_mostly_arabic(s):
                    continue
                if screen_transliterated and looks_transliterated(s):
                    continue
                yield s, "unk", "monolingual"
                n += 1
                if limit and n >= limit:
                    return

    # SDAIA lines that happen to carry no English still make good pass-through
    # examples, and come from the same distribution as the positives.
    for sent, dialect, _ in iter_sdaia(sdaia_dir):
        if _LATIN.search(sent) or not is_mostly_arabic(sent):
            continue
        if screen_transliterated and looks_transliterated(sent):
            continue
        yield sent, dialect, "sdaia-mono"
        n += 1
        if limit and n >= limit:
            return


# Monolingual Arabic used for pass-through training. Wikipedia supplies formal
# MSA under CC-BY-SA (compatible with the released corpus); the ASR transcript
# corpus supplies *dialectal* speech, which matters because the model must leave
# ordinary dialect alone too, not just newspaper Arabic.
MONO_SOURCES: list[dict] = [
    {
        "repo": "wikimedia/wikipedia",
        "config": "20231101.ar",
        "split": "train",
        "text_field": "text",
        "dialect": "msa",
        "source": "wikipedia",
        "sentence_split": True,
    },
    {
        "repo": "oddadmix/dialectal-arabic-lahgtna-v2",
        "config": "default",
        "split": "train",
        "text_field": "transcript_text",
        "dialect": "mix",
        "source": "lahgtna",
        "sentence_split": False,
    },
]


def iter_monolingual_remote(
    limit_per_source: int = 120_000, *, screen_transliterated: bool = True
) -> Iterator[tuple[str, str, str]]:
    """Stream pass-through candidates from the registered monolingual corpora."""
    for spec in MONO_SOURCES:
        for sent, dialect, source in iter_hf_monolingual(
            spec["repo"],
            spec["config"],
            spec["split"],
            spec["text_field"],
            dialect=spec["dialect"],
            source=spec["source"],
            limit=limit_per_source,
            sentence_split=spec["sentence_split"],
        ):
            if _LATIN.search(sent) or not is_mostly_arabic(sent):
                continue
            if screen_transliterated and looks_transliterated(sent):
                continue
            yield sent, dialect, source
