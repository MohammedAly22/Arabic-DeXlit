"""Synthesise code-switched Arabic with Gemini.

Purpose
-------
Two gaps in the harvested corpora justify generation:

1. **Dialect coverage.** SDAIA ArE-CSTD is MSA / Saudi / Egyptian only. Levantine,
   Iraqi, Maghrebi, Sudanese and Yemeni have no textual code-switching corpus.
2. **Category coverage.** A scale test over 20K SDAIA sentences yielded ~60K
   plain code-switch spans but only 178 acronyms, 9 entities and *zero* emails
   or numbers -- yet acronyms, emails and numbers are explicit requirements.

So generation is *targeted*: it buys the dialects and categories the free data
cannot supply, rather than bulk volume that SDAIA already provides cheaply.

Output is always the **target** form (English in Latin script). The noisy ASR
input is then manufactured deterministically by
:func:`arabic_dexlit.data.pairing.build_example`, so the generator never has to
be trusted to produce correct alignments -- it only has to write natural Arabic.

Robustness
----------
A long run must survive rate limits and interruption, so the writer appends
JSONL after every batch and ``--resume`` skips prompts already recorded. Failed
batches are retried with exponential backoff and then skipped, never fatal.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .dialects import CATEGORY_PROMPTS, DIALECT_BY_CODE, TOPICS, Dialect

DEFAULT_MODEL = "gemini-3.6-flash"
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

_ARABIC = re.compile(r"[؀-ۿ]")
_LATIN = re.compile(r"[A-Za-z]")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def api_key(explicit: str | None = None) -> str:
    """Resolve the API key from an argument or the environment."""
    key = explicit or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit(
            "No Gemini API key. Pass --api-key or set GEMINI_API_KEY."
        )
    return key


def build_prompt(dialect: Dialect, topic: str, category: str, n: int) -> str:
    """Compose one generation request.

    The instructions that protect data quality are the script rules: English
    must stay in Latin script, because the whole pipeline depends on the model
    emitting the *target* form. If the generator transliterated the English
    itself, the manufactured input/output pair would collapse.
    """
    emphasis = CATEGORY_PROMPTS[category]
    return f"""You are building a research dataset of natural Arabic-English code-switching.

DIALECT: {dialect.name}
{dialect.hint}

SPEAKER CONTEXT: {topic}

TASK: Write {n} different sentences of casual, spoken-style {dialect.name} Arabic
that naturally mix in English, as a real bilingual speaker would text or speak.

{emphasis}

STRICT RULES:
- Write the Arabic in Arabic script, in the {dialect.name} dialect (NOT formal MSA,
  unless the dialect IS Modern Standard Arabic).
- Keep every English word in LATIN script, spelled correctly. Never write English
  words using Arabic letters.
- Vary sentence length (5 to 25 words) and vary the English words used.
- Make them sound like real speech, not textbook examples.
- Do not number the sentences or add any commentary.

Return ONLY a JSON array of strings."""


@dataclass
class GenStats:
    requested: int = 0
    returned: int = 0
    kept: int = 0
    batches_ok: int = 0
    batches_failed: int = 0
    by_dialect: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)

    def note(self, dialect: str, category: str, n: int) -> None:
        self.by_dialect[dialect] = self.by_dialect.get(dialect, 0) + n
        self.by_category[category] = self.by_category.get(category, 0) + n


class GeminiClient:
    """Minimal Gemini caller. Uses urllib so the package has no hard HTTP dep."""

    def __init__(
        self,
        key: str,
        model: str = DEFAULT_MODEL,
        *,
        temperature: float = 1.15,
        max_retries: int = 4,
        timeout: int = 180,
    ) -> None:
        self.url = _ENDPOINT.format(model=model, key=key)
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

    def generate(self, prompt: str) -> list[str]:
        """Return the parsed sentence list, or ``[]`` if the batch fails."""
        body = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": self.temperature,
                    "responseMimeType": "application/json",
                },
            }
        ).encode()

        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    self.url, data=body, headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.load(r)
                return _parse(payload)
            except urllib.error.HTTPError as e:
                # 429 = rate limit, 5xx = transient. Both are worth waiting out.
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    time.sleep(delay + random.random())
                    delay *= 2
                    continue
                return []
            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(delay + random.random())
                    delay *= 2
                    continue
                return []
        return []


def _parse(payload: dict) -> list[str]:
    """Pull the JSON array out of a Gemini response, tolerating stray prose."""
    try:
        parts = payload["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError):
        return []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        return []
    text = _FENCE.sub("", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Salvage: take the first bracketed array in the blob.
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    return [s for s in data if isinstance(s, str)]


def acceptable(sent: str, *, min_words: int = 4, max_words: int = 40) -> bool:
    """Quality gate applied before a generated sentence enters the corpus.

    Rejects the three failure modes actually observed from LLM generation:
    no Arabic at all, no English (nothing to learn from), and runaway length.
    """
    s = sent.strip()
    if not s:
        return False
    words = s.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if not _ARABIC.search(s):
        return False
    if not _LATIN.search(s):
        return False
    return True


def plan_batches(
    total: int,
    dialect_codes: list[str],
    categories: list[str],
    per_batch: int,
    seed: int = 0,
) -> list[tuple[str, str, str, int]]:
    """Lay out ``(dialect, topic, category, n)`` jobs covering the request evenly.

    Spreading across dialect x category x topic in a planned grid, rather than
    sampling independently, keeps rare categories from being starved by chance.
    """
    rng = random.Random(seed)
    jobs: list[tuple[str, str, str, int]] = []
    combos = [(d, c) for d in dialect_codes for c in categories]
    if not combos:
        return jobs
    per_combo = max(1, total // (len(combos) * per_batch))
    for d, c in combos:
        for _ in range(per_combo):
            jobs.append((d, rng.choice(TOPICS), c, per_batch))
    rng.shuffle(jobs)
    return jobs


def synthesize(
    out_path: str | Path,
    *,
    key: str,
    total: int = 20000,
    dialect_codes: list[str] | None = None,
    categories: list[str] | None = None,
    per_batch: int = 12,
    workers: int = 8,
    model: str = DEFAULT_MODEL,
    seed: int = 0,
    resume: bool = True,
    verbose: bool = True,
) -> GenStats:
    """Generate sentences and append them to ``out_path`` as JSONL."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dialect_codes = dialect_codes or list(DIALECT_BY_CODE)
    categories = categories or list(CATEGORY_PROMPTS)

    seen: set[str] = set()
    if resume and out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    seen.add(json.loads(line)["text"])
                except Exception:
                    continue
        if verbose:
            print(f"[synth] resuming: {len(seen):,} sentences already on disk")

    jobs = plan_batches(total, dialect_codes, categories, per_batch, seed)
    client = GeminiClient(key, model)
    stats = GenStats(requested=total)

    lock = threading.Lock()
    fh = out_path.open("a", encoding="utf-8")

    def run(job: tuple[str, str, str, int]) -> None:
        code, topic, category, n = job
        dialect = DIALECT_BY_CODE[code]
        sents = client.generate(build_prompt(dialect, topic, category, n))
        if not sents:
            with lock:
                stats.batches_failed += 1
            return
        with lock:
            stats.batches_ok += 1
            stats.returned += len(sents)
            kept = 0
            buf: list[str] = []
            for s in sents:
                s = s.strip()
                if s in seen or not acceptable(s):
                    continue
                seen.add(s)
                buf.append(
                    json.dumps(
                        {
                            "text": s,
                            "dialect": code,
                            "category_hint": category,
                            "source": "gemini",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1
            # One buffered write per batch, inside the lock. Writing line by
            # line let concurrent workers interleave partial buffer flushes,
            # which produced a handful of corrupt (non-UTF-8) lines per run.
            if buf:
                fh.write("".join(buf))
                fh.flush()
            stats.kept += kept
            stats.note(code, category, kept)
            if verbose and stats.batches_ok % 20 == 0:
                print(
                    f"[synth] batches ok={stats.batches_ok} failed={stats.batches_failed} "
                    f"kept={stats.kept:,}",
                    flush=True,
                )

    # Stopping is cooperative rather than a `break` out of as_completed: leaving
    # that loop early lets the pool's shutdown cancel every still-pending future,
    # which silently truncated runs to a fraction of the requested total.
    done = threading.Event()

    def guarded(job: tuple[str, str, str, int]) -> None:
        if done.is_set():
            return
        run(job)
        if stats.kept >= total:
            done.set()

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(guarded, j) for j in jobs]):
                f.result()
    finally:
        fh.close()

    if verbose:
        print(
            f"[synth] done: kept {stats.kept:,} of {stats.returned:,} returned "
            f"({stats.batches_failed} batches failed)"
        )
    return stats
