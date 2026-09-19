#!/usr/bin/env python
"""Generate synthetic code-switched Arabic with Gemini.

Fills the two gaps the harvested corpora leave: dialects outside MSA/Saudi/
Egyptian, and the acronym / email / entity / number categories that barely occur
in SDAIA. Writes JSONL that ``build_dataset.py`` then consumes.

Examples
--------
    # fill the missing dialects and rare categories (the default recipe)
    python scripts/synthesize_data.py --total 30000 --api-key $GEMINI_API_KEY

    # top up one category only
    python scripts/synthesize_data.py --total 5000 --categories EMAIL NUMBER

The run is resumable: re-running with the same --out appends and skips
sentences already present.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_dexlit.data.dialects import CATEGORY_PROMPTS, DIALECT_BY_CODE  # noqa: E402
from arabic_dexlit.data.synth import DEFAULT_MODEL, api_key, synthesize  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/raw/synthetic.jsonl", help="JSONL output path")
    p.add_argument("--total", type=int, default=30000, help="target number of sentences")
    p.add_argument("--api-key", default=None, help="Gemini key (else $GEMINI_API_KEY)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--dialects", nargs="*", default=None,
        help=f"dialect codes; default all. choices: {', '.join(DIALECT_BY_CODE)}",
    )
    p.add_argument(
        "--categories", nargs="*", default=None,
        help=f"categories to emphasise; default all. choices: {', '.join(CATEGORY_PROMPTS)}",
    )
    p.add_argument("--per-batch", type=int, default=12, help="sentences per API call")
    p.add_argument("--workers", type=int, default=12, help="concurrent API calls")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-resume", action="store_true", help="ignore existing output")
    args = p.parse_args()

    for code in args.dialects or []:
        if code not in DIALECT_BY_CODE:
            p.error(f"unknown dialect {code!r}; choose from {', '.join(DIALECT_BY_CODE)}")
    for cat in args.categories or []:
        if cat not in CATEGORY_PROMPTS:
            p.error(f"unknown category {cat!r}; choose from {', '.join(CATEGORY_PROMPTS)}")

    stats = synthesize(
        args.out,
        key=api_key(args.api_key),
        total=args.total,
        dialect_codes=args.dialects,
        categories=args.categories,
        per_batch=args.per_batch,
        workers=args.workers,
        model=args.model,
        seed=args.seed,
        resume=not args.no_resume,
    )
    print("\n=== synthesis summary ===")
    print(f"kept      : {stats.kept:,}")
    print(f"batches   : {stats.batches_ok} ok / {stats.batches_failed} failed")
    print(f"by dialect: {stats.by_dialect}")
    print(f"by category: {stats.by_category}")
    print(f"written to: {args.out}")


if __name__ == "__main__":
    main()
