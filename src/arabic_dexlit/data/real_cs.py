"""Harvest *real* human code-switching, not synthesised text.

Why this matters
----------------
The bulk corpus is GPT-4 generated and writes code-switching the way a careful
writer would: ``ال meeting`` with a space, English words neatly separated. Real
speakers do not do that. In ArzEn-MultiGenre -- transcribed human speech and
subtitles -- the same phenomenon appears as::

    بس الsystem down          (article fused directly onto the English word)
    حلو الlook ده
    ده fresh graduates مش لاقيين
    حتى الextra expenses

That fused ``ال`` is precisely the pattern the model handled worst, turning it
into ``I'll``. No amount of synthetic data fixes it, because the generator that
makes the synthetic data does not produce the pattern in the first place.

Density is low -- about 6% of rows contain genuine intra-sentential switching --
so this is a *quality* supplement layered on top of the bulk corpus, never a
replacement for it. Every row here is worth many synthetic ones for teaching the
orthographic habits of actual speech.
"""
from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator

_LATIN = re.compile(r"[A-Za-z]")
_ARABIC = re.compile(r"[؀-ۿ]")
# Bidi and presentation marks litter subtitle corpora and would otherwise end up
# inside training targets.
_BIDI = re.compile(r"[​-‏‪-‮⁦-⁩]")

_ROWS_API = "https://datasets-server.huggingface.co/rows"

# Public datasets carrying genuine human code-switched Arabic.
REAL_CS_SOURCES: list[dict] = [
    {
        "repo": "HeshamHaroon/ArzEn-MultiGenre",
        "config": "default",
        "split": "train",
        "field": "EGY",
        "dialect": "egy",
        "license": "cc-by-4.0",
    },
]


def clean(text: str) -> str:
    """Strip bidi controls and normalise whitespace."""
    return " ".join(_BIDI.sub(" ", text or "").split())


def is_code_switched(text: str, *, min_words: int = 3, max_words: int = 60) -> bool:
    """True when a sentence genuinely mixes both scripts.

    Requiring *both* scripts is what separates code-switching from a monolingual
    English subtitle line, which these corpora also contain.
    """
    t = clean(text)
    if not t:
        return False
    n = len(t.split())
    if not (min_words <= n <= max_words):
        return False
    return bool(_LATIN.search(t) and _ARABIC.search(t))


def iter_hf_rows(
    repo: str,
    config: str,
    split: str,
    field: str,
    *,
    limit: int = 50_000,
    page: int = 100,
) -> Iterator[str]:
    """Stream one text column from a public HF dataset via the rows API."""
    base = (
        f"{_ROWS_API}?dataset={urllib.parse.quote(repo, safe='')}"
        f"&config={urllib.parse.quote(config, safe='')}"
        f"&split={urllib.parse.quote(split, safe='')}"
    )
    offset = 0
    seen = 0
    while seen < limit:
        payload = None
        delay = 3.0
        # The rows API rate-limits (HTTP 429) under sustained use, and a harvest
        # is exactly that. Backing off is the difference between collecting the
        # corpus and silently collecting nothing.
        for attempt in range(6):
            try:
                with urllib.request.urlopen(
                    f"{base}&offset={offset}&length={page}", timeout=120
                ) as r:
                    payload = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(delay + random.random())
                    delay *= 2
                    continue
                return
            except Exception:
                if attempt < 5:
                    time.sleep(delay + random.random())
                    delay *= 2
                    continue
                return
        if payload is None:
            return
        rows = payload.get("rows") or []
        if not rows:
            return
        for item in rows:
            value = (item.get("row") or {}).get(field)
            if value:
                yield value
                seen += 1
                if seen >= limit:
                    return
        offset += page


def harvest(
    out_path: str | Path = "data/raw/real_cs.jsonl",
    *,
    limit_per_source: int = 50_000,
    verbose: bool = True,
) -> int:
    """Collect real code-switched sentences into JSONL. Returns the count kept."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    kept = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for src in REAL_CS_SOURCES:
            got = 0
            for raw in iter_hf_rows(
                src["repo"], src["config"], src["split"], src["field"],
                limit=limit_per_source,
            ):
                text = clean(raw)
                if not is_code_switched(text) or text in seen:
                    continue
                seen.add(text)
                fh.write(
                    json.dumps(
                        {
                            "text": text,
                            "dialect": src["dialect"],
                            "source": "real-cs",
                            "repo": src["repo"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                got += 1
                kept += 1
            if verbose:
                print(f"[real-cs] {src['repo']}: kept {got:,}")
    if verbose:
        print(f"[real-cs] total {kept:,} -> {out_path}")
    return kept


def iter_real_cs(path: str | Path = "data/raw/real_cs.jsonl") -> Iterator[tuple[str, str, str]]:
    """Yield ``(target_sentence, dialect, source)`` for the dataset builder."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("rb") as fh:
        for raw in fh:
            try:
                row = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            text = (row.get("text") or "").strip()
            if text:
                yield text, row.get("dialect", "egy"), "real-cs"
