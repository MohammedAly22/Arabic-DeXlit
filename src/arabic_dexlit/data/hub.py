"""Fetch the prebuilt corpus from the Hugging Face Hub.

Building the dataset from scratch takes 10-20 minutes and re-downloads several
source corpora. The released corpus is already built, so on Colab the usual path
is to pull it instead -- roughly 68 MB of Parquet, a few seconds.

The training scripts read JSONL, so the Parquet splits are materialised to the
same layout ``scripts/build_dataset.py`` produces. That keeps exactly one data
format in the training code regardless of where the corpus came from.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_REPO = "mohammedaly22/ArabicDeXlit-Corpus"
SPLITS = ("train", "validation", "test")


def download_corpus(
    out_dir: str | Path = "data/processed",
    *,
    repo_id: str = DEFAULT_REPO,
    splits: tuple[str, ...] = SPLITS,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Download the corpus and write ``<out_dir>/<split>.jsonl``.

    Returns the output directory. Existing splits are left alone unless
    ``force`` is set, so re-running a notebook cell is cheap.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        dest = out_dir / f"{split}.jsonl"
        if dest.exists() and not force and dest.stat().st_size > 0:
            if verbose:
                print(f"[hub] {split}: already present, skipping")
            continue

        path = hf_hub_download(
            repo_id=repo_id, filename=f"{split}.parquet", repo_type="dataset"
        )
        table = pq.read_table(path)
        n = 0
        with dest.open("w", encoding="utf-8") as fh:
            # Batched rather than one big to_pylist(): the train split is ~428K
            # rows and materialising it whole wastes memory on a Colab VM.
            for batch in table.to_batches(max_chunksize=20_000):
                for row in batch.to_pylist():
                    fh.write(
                        json.dumps(
                            {
                                "src": row["src"],
                                "tgt": row["tgt"],
                                "src_tokens": row["src_tokens"],
                                "tags": row["tags"],
                                "spans": row["spans"],
                                "dialect": row["dialect"],
                                "source": row["source"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n += 1
        if verbose:
            print(f"[hub] {split}: {n:,} examples -> {dest}")

    try:
        stats = hf_hub_download(
            repo_id=repo_id, filename="dataset_stats.json", repo_type="dataset"
        )
        (out_dir / "dataset_stats.json").write_bytes(Path(stats).read_bytes())
    except Exception:
        pass  # optional metadata; its absence must not fail the download

    return out_dir


def corpus_summary(data_dir: str | Path = "data/processed") -> dict:
    """Count rows and pass-through examples in a materialised corpus."""
    data_dir = Path(data_dir)
    out: dict = {}
    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = passthrough = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rows += 1
                if all(t == "O" for t in json.loads(line)["tags"]):
                    passthrough += 1
        out[split] = {
            "rows": rows,
            "passthrough": passthrough,
            "passthrough_pct": round(100 * passthrough / max(1, rows), 1),
        }
    return out
