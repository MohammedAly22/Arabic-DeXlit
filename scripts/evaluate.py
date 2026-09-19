#!/usr/bin/env python
"""Evaluate a trained ArabicDeXlit model on a held-out split.

Reports the end-to-end numbers that decide whether the model is usable:

* **passthrough_accuracy** -- of the sentences that must come back untouched,
  how many did. The safety metric; read it first.
* **sentence_exact_match** -- the whole output string is correct.
* per-category span F1 and converter exact-match, to show *which* kind of span
  (acronym, email, entity...) is weak.

Example
-------
    python scripts/evaluate.py --model-dir outputs/dexlit \
        --data data/processed/test.jsonl --out outputs/dexlit/test_report.json
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

from arabic_dexlit.inference.pipeline import DeXlitPipeline  # noqa: E402
from arabic_dexlit.schema import OUTSIDE  # noqa: E402
from arabic_dexlit.training.metrics import DetectorMetrics  # noqa: E402
from arabic_dexlit.schema import TAG2ID  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data", default="data/processed/test.jsonl")
    p.add_argument("--out", default=None, help="write the JSON report here")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--copy-threshold", type=float, default=None,
        help="force tokens the copy gate is confident about back to O",
    )
    p.add_argument("--show", type=int, default=10, help="print this many examples")
    args = p.parse_args()

    rows = []
    with Path(args.data).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break
    print(f"[eval] {len(rows):,} examples from {args.data}")

    pipe = DeXlitPipeline.from_pretrained(
        args.model_dir, device=args.device, copy_threshold=args.copy_threshold
    )

    det = DetectorMetrics()
    sent_exact = 0
    pass_total = pass_ok = 0
    edit_total = edit_ok = 0
    by_cat_total: Counter = Counter()
    by_cat_ok: Counter = Counter()
    by_dialect_total: Counter = Counter()
    by_dialect_ok: Counter = Counter()
    shown: list[dict] = []

    t0 = time.time()
    for row in rows:
        gold_text = row["tgt"]
        out = pipe.predict(row["src"])

        exact = out.text.strip() == gold_text.strip()
        sent_exact += exact

        is_pass = all(t == OUTSIDE for t in row["tags"])
        if is_pass:
            pass_total += 1
            # For a pass-through row the bar is byte-identity, not "close".
            pass_ok += int(out.text == row["src"])
        else:
            edit_total += 1
            edit_ok += exact

        dialect = row.get("dialect", "unk")
        by_dialect_total[dialect] += 1
        by_dialect_ok[dialect] += exact
        for s in row.get("spans", []):
            by_cat_total[s["category"]] += 1
        if exact:
            for s in row.get("spans", []):
                by_cat_ok[s["category"]] += 1

        gold_ids = [TAG2ID.get(t, 0) for t in row["tags"]]
        pred_ids = [TAG2ID.get(t, 0) for t in out.tags[: len(gold_ids)]]
        pred_ids += [0] * (len(gold_ids) - len(pred_ids))
        det.update([pred_ids], [gold_ids])

        if len(shown) < args.show and not exact and not is_pass:
            shown.append({"src": row["src"], "pred": out.text, "gold": gold_text})

    elapsed = time.time() - t0
    report = {
        "n_examples": len(rows),
        "sentence_exact_match": sent_exact / max(1, len(rows)),
        # ``None`` rather than 0.0 when the slice contains no pass-through rows:
        # reporting a hard 0.0 for "not measured" reads as a total failure of the
        # very guarantee this metric exists to check.
        "passthrough_accuracy": (pass_ok / pass_total) if pass_total else None,
        "passthrough_examples": pass_total,
        "edit_exact_match": (edit_ok / edit_total) if edit_total else None,
        "edit_examples": edit_total,
        "detector": det.compute(),
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
