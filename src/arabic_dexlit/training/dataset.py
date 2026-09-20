"""Row loading and subsetting for the end-to-end rewriter.

The model is a single sequence-to-sequence pass over whole sentences, so there
is no span flattening, no tag projection and no per-category batching here --
all of that belonged to the previous pipeline. What remains is reading the
corpus and, when a run needs to be capped, capping it *without* skewing the
balance between sentences that must change and sentences that must not.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    """Load a built split."""
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def balanced_subset(
    rows: list[dict], max_rows: int | None, seed: int = 0
) -> list[dict]:
    """Cap corpus size while preserving the pass-through / edit ratio.

    Sampling uniformly would drift that ratio, and the pass-through share is the
    single most consequential property of this corpus: it is what teaches the
    model to leave ordinary Arabic alone. A capped run must not quietly train on
    a different task than the full one.
    """
    if not max_rows or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    edits = [r for r in rows if r["src"] != r["tgt"]]
    passes = [r for r in rows if r["src"] == r["tgt"]]
    frac = len(passes) / max(1, len(rows))
    n_pass = int(max_rows * frac)
    rng.shuffle(edits)
    rng.shuffle(passes)
    out = edits[: max_rows - n_pass] + passes[:n_pass]
    rng.shuffle(out)
    return out
