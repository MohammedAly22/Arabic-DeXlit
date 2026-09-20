#!/usr/bin/env python
"""Evaluate the end-to-end rewriter on a held-out split.

Read the report in this order:

1. **unnecessary_modification_rate** -- of the tokens that should have been left
   alone, how many were changed. For a drop-in ASR post-processor this decides
   whether the model is deployable at all: a missed conversion is recoverable by
   a reader, a corrupted word is not.
2. **passthrough_accuracy** -- pure-Arabic sentences returned byte-identical.
3. **sentence_exact_match** -- the whole output string correct.

Sentence exact-match alone is misleading here, because a large share of spans
are trivial (already-Latin text that only needs copying). A model can score well
on it while quietly damaging ordinary Arabic, which is exactly what the first
two metrics catch.

Example
-------
    python scripts/evaluate.py --model-dir outputs/dexlit-s2s \
        --data data/processed/test.jsonl --out outputs/dexlit-s2s/test_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from arabic_dexlit.model.seq2seq import Seq2SeqConfig, generate  # noqa: E402
from arabic_dexlit.schema import OUTSIDE  # noqa: E402
from arabic_dexlit.training.metrics import EndToEndMetrics  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-dir", required=True, help="directory holding seq2seq/")
    p.add_argument("--data", default="data/processed/test.jsonl")
    p.add_argument("--out", default=None, help="write the JSON report here")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--show", type=int, default=10, help="print this many errors")
    args = p.parse_args()

    rows = []
    with Path(args.data).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break
    print(f"[eval] {len(rows):,} examples from {args.data}")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_path = Path(args.model_dir) / "seq2seq"
    if not model_path.exists():
        model_path = Path(args.model_dir)
    cfg_path = Path(args.model_dir) / "seq2seq_config.json"
    cfg = (
        Seq2SeqConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
        if cfg_path.exists()
        else Seq2SeqConfig()
    )
    cfg.num_beams = args.num_beams

    tok = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(args.device).eval()

    metrics = EndToEndMetrics()
    by_dialect_total: Counter = Counter()
    by_dialect_ok: Counter = Counter()
    by_cat_total: Counter = Counter()
    by_cat_ok: Counter = Counter()
    pass_total = pass_ok = 0
    shown: list[dict] = []

    t0 = time.time()
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i : i + args.batch_size]
        preds = generate(model, tok, [r["src"] for r in chunk], cfg, device=args.device)

        for r, pred in zip(chunk, preds):
            exact = pred.strip() == r["tgt"].strip()
            metrics.update(
                pred_tokens=pred.split(),
                gold_tokens=r["tgt"].split(),
                src_tokens=r["src_tokens"],
                gold_tags=r["tags"],
            )

            # A pass-through row must come back byte-identical, not merely close.
            if all(t == OUTSIDE for t in r["tags"]):
                pass_total += 1
                pass_ok += int(pred.strip() == r["src"].strip())

            d = r.get("dialect", "unk")
            by_dialect_total[d] += 1
            by_dialect_ok[d] += exact
            for s in r.get("spans", []):
                by_cat_total[s["category"]] += 1
                by_cat_ok[s["category"]] += exact

            if len(shown) < args.show and not exact:
                shown.append({"src": r["src"], "pred": pred, "gold": r["tgt"]})

        if i and i % (args.batch_size * 20) == 0:
            print(f"[eval]   {i:,}/{len(rows):,}", flush=True)

    elapsed = time.time() - t0
    report = {
        "n_examples": len(rows),
        **metrics.compute(),
        # ``None`` rather than 0.0 when the slice holds no pass-through rows:
        # reporting 0.0 for "not measured" reads as total failure of the very
        # guarantee this metric exists to check.
        "passthrough_accuracy": (pass_ok / pass_total) if pass_total else None,
        "passthrough_examples": pass_total,
        "by_category_sentence_accuracy": {
            c: by_cat_ok[c] / n for c, n in by_cat_total.items() if n
        },
        "by_dialect_sentence_accuracy": {
            d: by_dialect_ok[d] / n for d, n in by_dialect_total.items() if n
        },
        "seconds": round(elapsed, 1),
        "ms_per_sentence": round(1000 * elapsed / max(1, len(rows)), 2),
    }

    print("\n=== report ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if shown:
        print("\n=== example errors ===")
        for e in shown:
            print(f"IN  : {e['src']}")
            print(f"PRED: {e['pred']}")
            print(f"GOLD: {e['gold']}\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
