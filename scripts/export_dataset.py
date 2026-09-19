#!/usr/bin/env python
"""Convert the built JSONL splits to Parquet and push them to the Hugging Face Hub.

Parquet rather than raw JSONL because it is several times smaller, loads far
faster with ``datasets``, and is what powers the Hub's dataset viewer. The
schema is declared explicitly so every split types identically -- inferred
schemas can disagree between splits when a column (say, an empty ``spans``
list) happens to be absent from a sample.

Examples
--------
    # convert only
    python scripts/export_dataset.py --out-dir data/hf

    # convert and push
    python scripts/export_dataset.py --push --repo-id USER/ArabicDeXlit-Corpus
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ("train", "validation", "test")

# Declared up front so all three splits share one schema.
SCHEMA = pa.schema(
    [
        pa.field("src", pa.string()),
        pa.field("tgt", pa.string()),
        pa.field("src_tokens", pa.list_(pa.string())),
        pa.field("tags", pa.list_(pa.string())),
        pa.field(
            "spans",
            pa.list_(
                pa.struct(
                    [
                        pa.field("start", pa.int32()),
                        pa.field("end", pa.int32()),
                        pa.field("category", pa.string()),
                        pa.field("target", pa.string()),
                    ]
                )
            ),
        ),
        pa.field("dialect", pa.string()),
        pa.field("source", pa.string()),
        # Derived, but worth materialising: it is the column most users will
        # filter on ("give me only the sentences that must not change").
        pa.field("is_passthrough", pa.bool_()),
        pa.field("num_spans", pa.int32()),
    ]
)


def convert(src: Path, dest: Path, *, chunk: int = 50_000) -> int:
    """Stream one JSONL split into a Parquet file without loading it all."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(dest, SCHEMA, compression="zstd")
    rows: list[dict] = []
    total = 0

    def flush() -> None:
        nonlocal rows
        if not rows:
            return
        writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
        rows = []

    try:
        with src.open("rb") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                try:
                    r = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                spans = r.get("spans") or []
                rows.append(
                    {
                        "src": r["src"],
                        "tgt": r["tgt"],
                        "src_tokens": r["src_tokens"],
                        "tags": r["tags"],
                        "spans": [
                            {
                                "start": int(s["start"]),
                                "end": int(s["end"]),
                                "category": s["category"],
                                "target": s["target"],
                            }
                            for s in spans
                        ],
                        "dialect": r.get("dialect", "unk"),
                        "source": r.get("source", "unk"),
                        "is_passthrough": all(t == "O" for t in r["tags"]),
                        "num_spans": len(spans),
                    }
                )
                total += 1
                if len(rows) >= chunk:
                    flush()
        flush()
    finally:
        writer.close()
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out-dir", default="data/hf")
    p.add_argument("--push", action="store_true", help="upload to the Hub")
    p.add_argument("--repo-id", default=None, help="e.g. USER/ArabicDeXlit-Corpus")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        src = data_dir / f"{split}.jsonl"
        if not src.exists():
            sys.exit(f"missing {src} -- run scripts/build_dataset.py first")
        dest = out_dir / f"{split}.parquet"
        n = convert(src, dest)
        mb_in = src.stat().st_size / 1e6
        mb_out = dest.stat().st_size / 1e6
        print(f"[export] {split:11s} {n:>8,} rows  {mb_in:7.1f}MB -> {mb_out:6.1f}MB parquet")

    for extra in ("README.md", "dataset_stats.json"):
        srcf = data_dir / extra
        if srcf.exists():
            (out_dir / extra).write_bytes(srcf.read_bytes())

    if args.push:
        if not args.repo_id:
            sys.exit("--push needs --repo-id")
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)
        print(f"[push] uploading to {args.repo_id} ...")
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Add ArabicDeXlit corpus (parquet)",
        )
        print(f"[push] https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
