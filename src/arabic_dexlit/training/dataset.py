"""Torch datasets and collators for both stages.

The subtle part is **label projection**. The detector's labels are one tag per
*word*, but the encoder works in sub-word pieces. Each word's tag is therefore
attached to its **first** sub-word, and every continuation piece is set to
``IGNORE_INDEX`` so it contributes no loss. At inference the same rule reads the
prediction back off the first piece. Getting this wrong is the classic silent
bug in token classification -- the loss still falls, but the labels mean nothing
-- so ``word_ids`` from the fast tokenizer is used rather than any hand-rolled
alignment.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from ..model.converter import (
    BOS_ID,
    CATEGORY_IDS,
    EOS_ID,
    PAD_ID,
    SPAN_CLOSE,
    SPAN_OPEN,
    encode_chars,
)
from ..model.detector import script_of
from ..schema import IGNORE_INDEX, OUTSIDE_ID, TAG2ID


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class DetectorDataset(Dataset):
    """Word-tagged examples, encoded for a sub-word transformer."""

    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        max_length: int = 128,
    ) -> None:
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        words: list[str] = row["src_tokens"]
        tags: list[str] = row["tags"]

        enc = self.tok(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = enc.word_ids()

        labels: list[int] = []
        scripts: list[int] = []
        copy: list[int] = []
        prev = None
        for wid in word_ids:
            if wid is None:  # [CLS] / [SEP] / padding
                labels.append(IGNORE_INDEX)
                scripts.append(0)
                copy.append(0)
            elif wid != prev:  # first sub-word of a word: carries the label
                tag_id = TAG2ID.get(tags[wid], OUTSIDE_ID)
                labels.append(tag_id)
                scripts.append(script_of(words[wid]))
                copy.append(1 if tag_id == OUTSIDE_ID else 0)
            else:  # continuation piece: no loss
                labels.append(IGNORE_INDEX)
                scripts.append(script_of(words[wid]))
                copy.append(0)
            prev = wid

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "script_ids": scripts,
            "labels": labels,
            "copy_labels": copy,
        }


class DetectorCollator:
    """Picklable collator.

    A ``lambda`` or closure cannot be sent to DataLoader worker processes under
    the spawn start method (Windows, and macOS by default), which fails the run
    with a pickling error. A small class keeps ``num_workers > 0`` portable.
    """

    def __init__(self, pad_id: int = 0) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        return collate_detector(batch, self.pad_id)


def collate_detector(batch: list[dict], pad_id: int = 0) -> dict:
    """Pad a detector batch to its longest member."""
    n = max(len(b["input_ids"]) for b in batch)

    def pad(key: str, fill: int) -> torch.Tensor:
        return torch.tensor(
            [b[key] + [fill] * (n - len(b[key])) for b in batch], dtype=torch.long
        )

    return {
        "input_ids": pad("input_ids", pad_id),
        "attention_mask": pad("attention_mask", 0),
        "script_ids": pad("script_ids", 0),
        "labels": pad("labels", IGNORE_INDEX),
        "copy_labels": pad("copy_labels", 0),
    }


def build_context(
    tokens: list[str], start: int, end: int, window: int = 3
) -> str:
    """Wrap a span in its surrounding words, with explicit boundary markers.

    ``["عندي","ميتينج","مع","ال","مانجر"], 1, 2`` ->
    ``"عندي ‹ ميتينج › مع ال مانجر"``

    The guillemets mark exactly which tokens must be converted; everything else
    is there only to disambiguate. This is what lets the model learn that a bare
    Arabic ال in context is the article (leave it) rather than a transliteration
    of "la", which it cannot possibly know from the span alone.
    """
    if window <= 0:
        return " ".join(tokens[start:end])
    left = tokens[max(0, start - window) : start]
    right = tokens[end : end + window]
    span = tokens[start:end]
    return " ".join([*left, SPAN_OPEN, *span, SPAN_CLOSE, *right])


class ConverterDataset(Dataset):
    """Span-level pairs: Arabic-script characters -> English characters.

    Built by flattening every span out of the detector corpus, so the two stages
    train on exactly the same data, just viewed differently.
    """

    def __init__(
        self,
        rows: Iterable[dict],
        *,
        max_src_len: int = 48,
        max_tgt_len: int = 40,
        dedupe: bool = True,
        lexicon=None,
        context_window: int = 3,
    ) -> None:
        self.context_window = context_window
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.lexicon = lexicon
        self.pairs: list[tuple[str, str, int]] = []
        seen: set[tuple[str, str, str]] = set()
        for row in rows:
            toks = row["src_tokens"]
            for span in row.get("spans", []):
                start, end = span["start"], span["end"]
                src = " ".join(toks[start:end])
                tgt = span["target"]
                cat = span["category"]
                if not src or not tgt:
                    continue
                # Surrounding words, marked off so the model can tell context
                # from the span it must convert. Without this the converter sees
                # a span in isolation and cannot tell Arabic ال ("the") from a
                # transliterated English word -- the observed failure was
                # ال -> "la", و -> "we", ان -> "na".
                ctx = build_context(toks, start, end, self.context_window)
                key = (ctx, tgt, cat)
                if dedupe and key in seen:
                    continue
                seen.add(key)
                self.pairs.append((ctx, tgt, CATEGORY_IDS.get(cat, 0)))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        src, tgt, cat = self.pairs[i]
        src_ids = encode_chars(src, self.max_src_len)
        tgt_ids = encode_chars(tgt, self.max_tgt_len)
        item = {
            "src": src_ids,
            "category": cat,
            # Teacher forcing: the decoder reads <bos>+y and predicts y+<eos>.
            "tgt_in": [BOS_ID] + tgt_ids[:-1],
            "tgt_out": tgt_ids,
        }
        if self.lexicon is not None:
            # 0 (OOV) is a real class here, not a padding value: the word head
            # must learn to abstain on targets outside the lexicon.
            item["word_id"] = self.lexicon.get(tgt)
        return item

    def targets(self) -> list[str]:
        """All gold targets, for building a lexicon."""
        return [t for _, t, _ in self.pairs]


def collate_converter(batch: list[dict]) -> dict:
    ns = max(len(b["src"]) for b in batch)
    nt = max(len(b["tgt_in"]) for b in batch)

    def pad(key: str, n: int) -> torch.Tensor:
        return torch.tensor(
            [b[key] + [PAD_ID] * (n - len(b[key])) for b in batch], dtype=torch.long
        )

    out = {
        "src": pad("src", ns),
        "category": torch.tensor([b["category"] for b in batch], dtype=torch.long),
        "tgt_in": pad("tgt_in", nt),
        "tgt_out": pad("tgt_out", nt),
    }
    if "word_id" in batch[0]:
        out["word_ids"] = torch.tensor([b["word_id"] for b in batch], dtype=torch.long)
    return out


def balanced_subset(
    rows: list[dict], max_rows: int | None, seed: int = 0
) -> list[dict]:
    """Cap corpus size while preserving the pass-through / edit balance."""
    if not max_rows or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    edits = [r for r in rows if any(t != "O" for t in r["tags"])]
    passes = [r for r in rows if all(t == "O" for t in r["tags"])]
    frac = len(passes) / max(1, len(rows))
    n_pass = int(max_rows * frac)
    rng.shuffle(edits)
    rng.shuffle(passes)
    out = edits[: max_rows - n_pass] + passes[:n_pass]
    rng.shuffle(out)
    return out
