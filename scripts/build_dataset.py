#!/usr/bin/env python
"""Build the ArabicDeXlit train / validation / test corpus.

Downloads the SDAIA code-switching corpus, folds in any synthetic data produced
by ``synthesize_data.py``, manufactures the ASR-style inputs, mixes in
pass-through examples, and writes leak-free splits.

Examples
--------
    python scripts/build_dataset.py                      # full build
    python scripts/build_dataset.py --max-examples 5000  # quick smoke build
    python scripts/build_dataset.py --variants 2         # 2x augmentation
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_dexlit.data.build import build_corpus  # noqa: E402
from arabic_dexlit.data.real_cs import harvest as harvest_real_cs  # noqa: E402
from arabic_dexlit.data.real_cs import iter_real_cs  # noqa: E402
from arabic_dexlit.data.vocab_inject import iter_vocab_sentences  # noqa: E402
from arabic_dexlit.data.sources import (  # noqa: E402
    iter_monolingual,
    iter_monolingual_remote,
    iter_sdaia,
    iter_synthetic,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--raw-dir", default="data/raw/sdaia")
    p.add_argument("--synthetic", default="data/raw/synthetic.jsonl")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument(
        "--passthrough-ratio", type=float, default=0.30,
        help="fraction of the corpus that is pure Arabic (must pass through unchanged)",
    )
    p.add_argument(
        "--variants", type=int, default=1,
        help="re-transliterate each target N times (stochastic augmentation)",
    )
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-sdaia", action="store_true")
    p.add_argument("--no-synthetic", action="store_true")
    p.add_argument(
        "--vocab-sentences", type=int, default=80000,
        help="locally synthesised sentences covering real code-switching "
             "vocabulary (intern, deadline, sprint...), which the harvested "
             "corpus barely contains; 0 disables",
    )
    p.add_argument(
        "--real-cs", default="data/raw/real_cs.jsonl",
        help="human code-switched sentences (ArzEn); teaches the fused-article "
             "pattern (الsystem) that synthetic data never produces",
    )
    p.add_argument(
        "--fetch-real-cs", action="store_true",
        help="harvest the real code-switching corpus before building",
    )
    p.add_argument(
        "--real-cs-repeat", type=int, default=8,
        help="oversample real sentences; they are few but worth many synthetic ones",
    )
    p.add_argument(
        "--mono-cache", default="data/raw/monolingual.txt",
        help="local cache of monolingual Arabic used for pass-through examples",
    )
    p.add_argument(
        "--fetch-mono", type=int, default=150000,
        help="monolingual Arabic to download for pass-through examples; these "
             "are what teach the model to leave ordinary text alone, so the "
             "default fetches them rather than silently building without",
    )
    args = p.parse_args()

    streams = []
    if not args.no_sdaia:
        streams.append(iter_sdaia(args.raw_dir))
    if not args.no_synthetic and Path(args.synthetic).exists():
        streams.append(iter_synthetic(args.synthetic))
    if args.vocab_sentences:
        streams.append(iter_vocab_sentences(args.vocab_sentences, seed=args.seed))
    if args.fetch_real_cs:
        harvest_real_cs(args.real_cs)
    if Path(args.real_cs).exists() and args.real_cs_repeat:
        # Deliberately oversampled. There are only a few hundred of these, but
        # they carry orthographic habits (الsystem, fused articles) that the
        # generator cannot produce, so each is worth many synthetic rows.
        real = list(iter_real_cs(args.real_cs))
        streams.append(iter(real * args.real_cs_repeat))
        print(f"[real-cs] {len(real):,} sentences x{args.real_cs_repeat}")
    if not streams:
        p.error("no sources enabled")

    # SDAIA is ~99% code-switched by construction, so it cannot supply enough
    # monolingual Arabic on its own -- a build relying on it alone produced a
    # 0.6% pass-through ratio against a 30% target. External monolingual text
    # is cached locally and used as the primary source.
    mono_cache = Path(args.mono_cache)
    if args.fetch_mono:
        mono_cache.parent.mkdir(parents=True, exist_ok=True)
        have = sum(1 for _ in mono_cache.open(encoding="utf-8")) if mono_cache.exists() else 0
        if have < args.fetch_mono:
            print(f"[mono] fetching {args.fetch_mono - have:,} more sentences ...")
            with mono_cache.open("a", encoding="utf-8") as fh:
                for i, (sent, _, _) in enumerate(
                    iter_monolingual_remote(args.fetch_mono)
                ):
                    fh.write(sent.replace("\n", " ") + "\n")
                    if i and i % 20000 == 0:
                        print(f"[mono]   {i:,}", flush=True)
                        fh.flush()
        print(f"[mono] cache: {mono_cache}")

    stats = build_corpus(
        itertools.chain(*streams),
        monolingual=iter_monolingual(
            paths=[mono_cache] if mono_cache.exists() else None,
            sdaia_dir=args.raw_dir,
        ),
        out_dir=args.out_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        passthrough_ratio=args.passthrough_ratio,
        seed=args.seed,
        max_examples=args.max_examples,
        variants=args.variants,
    )

    print("\n=== dataset summary ===")
    print(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
