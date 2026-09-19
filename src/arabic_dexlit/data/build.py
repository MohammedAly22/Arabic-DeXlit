"""Assemble the final train / val / test corpus.

Responsibilities
----------------
1. Pull targets from every enabled source.
2. Manufacture the ASR-style input for each (:mod:`.pairing`).
3. Mix in pure-Arabic pass-through examples at a controlled ratio.
4. Split without leakage.
5. Write JSONL plus a statistics report.

Leakage control
---------------
SDAIA is LLM-generated and contains large families of near-identical sentences
("اليوم عندي meeting مع ال team في المكتب" recurs with small edits). A random
split would scatter such a family across train and test and inflate the reported
score. Splitting is therefore done on a *normalised sentence key* -- Arabic
diacritics stripped, English lowercased, digits and punctuation folded -- so an
entire family lands in exactly one split.

Pass-through ratio
------------------
``passthrough_ratio`` is the single most consequential knob in the build. Too
low and the model learns to always edit something; too high and it learns to
never edit. The default of 0.3 reflects the deployment reality that a large
share of real ASR output carries no code-switching at all.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from .pairing import Example, build_example, passthrough_example

_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_NONWORD = re.compile(r"[^\w\s]", re.UNICODE)
_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")


def normal_key(sentence: str) -> str:
    """Collapse a sentence to a family key for leak-free splitting."""
    s = unicodedata.normalize("NFKC", sentence).lower()
    s = _DIACRITICS.sub("", s)
    # Alef/ya/ta-marbuta variants are orthographic noise, not different words.
    s = s.translate(str.maketrans({"أ": "ا", "إ": "ا",
                                   "آ": "ا", "ى": "ي",
                                   "ة": "ه"}))
    s = _DIGITS.sub("0", s)
    s = _NONWORD.sub(" ", s)
    return _WS.sub(" ", s).strip()


def split_of(key: str, val_frac: float, test_frac: float) -> str:
    """Deterministically assign a family key to a split.

    Hashing rather than shuffling means the assignment is stable across runs and
    across machines, so a rebuilt dataset keeps the same test set.
    """
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "validation"
    return "train"


@dataclass
class BuildStats:
    total_seen: int = 0
    kept: int = 0
    skipped_no_latin: int = 0
    skipped_dupe: int = 0
    passthrough: int = 0
    by_split: Counter = field(default_factory=Counter)
    by_dialect: Counter = field(default_factory=Counter)
    by_source: Counter = field(default_factory=Counter)
    by_category: Counter = field(default_factory=Counter)
    token_count: int = 0
    span_count: int = 0

    def record(self, ex: Example, split: str) -> None:
        self.kept += 1
        self.by_split[split] += 1
        self.by_dialect[ex.dialect] += 1
        self.by_source[ex.source] += 1
        self.token_count += len(ex.src_tokens)
        self.span_count += len(ex.spans)
        if ex.is_passthrough():
            self.passthrough += 1
        for s in ex.spans:
            self.by_category[s["category"]] += 1

    def as_dict(self) -> dict:
        return {
            "total_seen": self.total_seen,
            "kept": self.kept,
            "skipped_no_latin": self.skipped_no_latin,
            "skipped_duplicate": self.skipped_dupe,
            "passthrough_examples": self.passthrough,
            "passthrough_ratio": round(self.passthrough / max(1, self.kept), 4),
            "total_tokens": self.token_count,
            "total_spans": self.span_count,
            "by_split": dict(self.by_split),
            "by_dialect": dict(self.by_dialect),
            "by_source": dict(self.by_source),
            "by_category": dict(self.by_category),
        }


def build_corpus(
    targets: Iterable[tuple[str, str, str]],
    monolingual: Iterable[tuple[str, str, str]] | None = None,
    *,
    out_dir: str | Path = "data/processed",
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    passthrough_ratio: float = 0.30,
    seed: int = 1234,
    max_examples: int | None = None,
    variants: int = 1,
    shuffle_output: bool = True,
    verbose: bool = True,
) -> BuildStats:
    """Build and write the dataset.

    ``variants`` > 1 re-transliterates each target several times. Because the
    transliterator is stochastic, this yields genuinely different inputs for the
    same output -- cheap augmentation that teaches robustness to the spelling
    variation real ASR produces.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    stats = BuildStats()
    seen_keys: set[str] = set()

    handles = {
        s: (out_dir / f"{s}.jsonl").open("w", encoding="utf-8")
        for s in ("train", "validation", "test")
    }

    # A family key is computed from the *target*, but two different targets can
    # collapse to the same key once transliterated -- SDAIA writes English
    # "mall" as مال, which is also an ordinary Arabic word. Those rows would
    # then be split by their own (differing) target keys while looking identical
    # on the input side, i.e. a leak. Remembering the split each *input* key was
    # first assigned to keeps identical inputs together.
    input_split: dict[str, str] = {}

    def emit(ex: Example, key: str) -> None:
        src_key = normal_key(ex.src)
        split = input_split.get(src_key)
        if split is None:
            split = split_of(key, val_frac, test_frac)
            input_split[src_key] = split
        handles[split].write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        stats.record(ex, split)

    try:
        for sent, dialect, source in targets:
            stats.total_seen += 1
            key = normal_key(sent)
            if not key:
                continue
            if key in seen_keys:
                stats.skipped_dupe += 1
                continue
            seen_keys.add(key)

            made = 0
            for v in range(variants):
                ex = build_example(
                    sent, rng, dialect=dialect, source=source
                )
                if ex is None:
                    break
                emit(ex, key)  # same key => all variants share a split
                made += 1
            if made == 0:
                stats.skipped_no_latin += 1

            if verbose and stats.total_seen % 50000 == 0:
                print(f"[build] seen {stats.total_seen:,} kept {stats.kept:,}", flush=True)
            if max_examples and stats.kept >= max_examples:
                break

        # --- pass-through examples ---------------------------------------
        # Added last so the count can be sized against the positives actually
        # produced, keeping the ratio honest regardless of source yield.
        if monolingual is not None and passthrough_ratio > 0:
            want = int(stats.kept * passthrough_ratio / max(1e-9, 1 - passthrough_ratio))
            added = 0
            # Pass-through sentences are tracked in their own key set. They are
            # drawn from the same corpora as the positives, so sharing
            # ``seen_keys`` would reject every one of them as an exact duplicate
            # of the sentence it came from -- which silently produced a corpus
            # with zero pass-through examples.
            pt_keys: set[str] = set()
            for sent, dialect, source in monolingual:
                if added >= want:
                    break
                key = normal_key(sent)
                if not key or key in pt_keys:
                    continue
                pt_keys.add(key)
                ex = passthrough_example(sent, dialect=dialect, source=source)
                if ex is None:
                    continue
                emit(ex, key)
                added += 1
            if verbose:
                print(f"[build] added {added:,} pass-through examples (target {want:,})")
    finally:
        for fh in handles.values():
            fh.close()

    # Pass-through examples are generated after the positives, so on disk they
    # form one contiguous block at the end of each split. Anything that reads a
    # prefix -- a --limit on evaluation, a quick look at the head of the file --
    # would then see no pass-through rows at all and report the guarantee as
    # unmeasured (or, worse, as zero). Shuffle each split once, in place.
    if shuffle_output:
        shuf = random.Random(seed + 1)
        for name in ("train", "validation", "test"):
            path = out_dir / f"{name}.jsonl"
            if not path.exists():
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            shuf.shuffle(lines)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = out_dir / "dataset_stats.json"
    report.write_text(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"[build] wrote {stats.kept:,} examples to {out_dir}")
        print(f"[build] splits: {dict(stats.by_split)}")
    return stats


def load_split(path: str | Path) -> Iterator[dict]:
    """Stream a built split back from disk."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)
