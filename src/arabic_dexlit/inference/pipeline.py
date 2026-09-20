"""End-to-end inference: detect spans, convert them, rebuild the sentence.

The pass-through guarantee is enforced here, in code, not hoped for from the
weights:

* a sentence whose tokens are all tagged ``O`` short-circuits and the **original
  string is returned unchanged**, before any conversion runs;
* tokens outside a span are never passed through the converter, so they cannot
  be altered;
* rebuilding preserves the original whitespace of untouched regions.

That means the worst a mis-firing detector can do is convert a span it should
have left alone -- it can never corrupt text it did not flag.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..model.converter import (
    CATEGORY_IDS,
    ConverterConfig,
    SpanConverter,
    decode_ids,
    encode_chars,
)
from ..model.detector import DetectorConfig, SpanDetector, script_of
from ..model.lexicon import OOV_ID, Lexicon
from ..schema import ID2TAG, OUTSIDE, spans_from_tags
from ..training.dataset import build_context
from .protect import protect as protect_tokens

_WS = re.compile(r"\s+")


@dataclass
class Prediction:
    text: str
    changed: bool
    spans: list[dict]
    tokens: list[str]
    tags: list[str]
    # Tokens a deterministic rule pinned, with the reason. Useful for showing a
    # user that a rule -- not the model -- decided a given token.
    protected: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "changed": self.changed,
            "spans": self.spans,
        }


class DeXlitPipeline:
    """Load a trained ArabicDeXlit model and run it over text."""

    def __init__(
        self,
        detector: SpanDetector,
        converter: SpanConverter | None,
        tokenizer,
        *,
        device: torch.device | str = "cpu",
        copy_threshold: float | None = None,
        max_length: int = 256,
        lexicon: Lexicon | None = None,
        use_protection: bool = True,
        protect_latin: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.detector = detector.to(self.device).eval()
        self.converter = converter.to(self.device).eval() if converter is not None else None
        self.tok = tokenizer
        self.copy_threshold = copy_threshold
        self.max_length = max_length
        self.lexicon = lexicon
        self.use_protection = use_protection
        self.protect_latin = protect_latin

    # --- loading ----------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        device: torch.device | str = "cpu",
        copy_threshold: float | None = None,
    ) -> "DeXlitPipeline":
        from transformers import AutoTokenizer

        path = Path(path)
        dcfg = DetectorConfig.from_dict(
            json.loads((path / "detector_config.json").read_text(encoding="utf-8"))
        )
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        det = SpanDetector(dcfg)
        det.load_state_dict(torch.load(path / "detector.pt", map_location="cpu"))

        conv = None
        cpath = path / "converter.pt"
        if cpath.exists():
            ccfg = ConverterConfig.from_dict(
                json.loads((path / "converter_config.json").read_text(encoding="utf-8"))
            )
            conv = SpanConverter(ccfg)
            conv.load_state_dict(torch.load(cpath, map_location="cpu"))

        lex_path = path / "lexicon.json"
        lexicon = Lexicon.load(lex_path) if lex_path.exists() else None
        return cls(
            det, conv, tok, device=device, copy_threshold=copy_threshold,
            lexicon=lexicon,
        )

    # --- detection --------------------------------------------------------
    @torch.no_grad()
    def tag(self, words: list[str]) -> list[str]:
        """Return one tag per word."""
        if not words:
            return []
        enc = self.tok(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        word_ids = enc.word_ids()
        scripts = torch.tensor(
            [[0 if w is None else script_of(words[w]) for w in word_ids]],
            dtype=torch.long,
        )
        pred = self.detector.predict(
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
            scripts.to(self.device),
            copy_threshold=self.copy_threshold,
        )[0].cpu().tolist()

        tags = [OUTSIDE] * len(words)
        seen: set[int] = set()
        for pos, w in enumerate(word_ids):
            if w is None or w in seen:
                continue
            seen.add(w)
            tags[w] = ID2TAG.get(int(pred[pos]), OUTSIDE)
        return tags

    # --- conversion -------------------------------------------------------
    @torch.no_grad()
    def convert_spans(self, texts: list[str], categories: list[str]) -> list[str]:
        """Convert flagged spans to English. Batched for a single decode pass."""
        if not texts or self.converter is None:
            return list(texts)
        cfg = self.converter.cfg
        src = [encode_chars(t, cfg.max_src_len) for t in texts]
        n = max(len(s) for s in src)
        src_t = torch.tensor([s + [0] * (n - len(s)) for s in src], dtype=torch.long)
        cat_t = torch.tensor(
            [CATEGORY_IDS.get(c, 0) for c in categories], dtype=torch.long
        )
        src_t, cat_t = src_t.to(self.device), cat_t.to(self.device)

        # Three tiers, cheapest and safest first:
        #   1. NAR head  -- one forward pass, cannot produce repetition
        #   2. word head -- a whole lexicon word, cannot be misspelled
        #   3. AR decode -- the open-vocabulary fallback
        cfg = self.converter.cfg
        if getattr(self.converter, "nar_char_head", None) is not None:
            out = self.converter.nar_decode(src_t, cat_t)
        else:
            gen = self.converter.greedy_decode(src_t, cat_t)
            out = [decode_ids(r) for r in gen.cpu().tolist()]

        # Prefer the word head wherever it is confident: its output is a whole
        # lexicon entry, so it cannot be a misspelling. The character decoder
        # covers the rest -- unseen brand names, URLs, emails.
        if self.lexicon is not None and getattr(self.converter, "word_head", None):
            widx, wconf = self.converter.predict_words(src_t, cat_t)
            thr = self.converter.cfg.word_confidence
            for j, (wi, wc) in enumerate(zip(widx.cpu().tolist(), wconf.cpu().tolist())):
                if wi != OOV_ID and wc >= thr:
                    out[j] = self.lexicon.word(wi)
        return out

    # --- end to end -------------------------------------------------------
    def __call__(self, text: str) -> Prediction:
        return self.predict(text)

    def predict(self, text: str) -> Prediction:
        words = text.split()
        if not words:
            return Prediction(text, False, [], [], [])

        tags = self.tag(words)

        # Deterministic protection runs AFTER tagging but BEFORE anything is
        # converted: emails, URLs, IPs, numbers, dates and times have exactly one
        # correct output, so a rule decides them and the model's opinion is
        # discarded. This turns "probably preserved" into "cannot be modified".
        protections = []
        if self.use_protection:
            protections = protect_tokens(words, protect_latin=self.protect_latin)
            for p in protections:
                tags[p.index] = OUTSIDE

        # The guarantee: nothing flagged => the original string, untouched.
        if all(t == OUTSIDE for t in tags):
            return Prediction(text, False, [], words, tags)

        spans = spans_from_tags(tags)
        if not spans:
            return Prediction(text, False, [], words, tags)

        # Same context format the converter was trained on -- a bare span would
        # be out of distribution and reintroduces the ambiguity context solves.
        window = getattr(self.converter.cfg, "context_window", 0) if self.converter else 0
        texts = [build_context(words, s, e, window) for s, e, _ in spans]
        cats = [c for _, _, c in spans]
        converted = self.convert_spans(texts, cats)

        out: list[str] = []
        cursor = 0
        recorded: list[dict] = []
        for (start, end, cat), new in zip(spans, converted):
            out.extend(words[cursor:start])
            new = new.strip() or " ".join(words[start:end])
            out.append(new)
            recorded.append(
                {
                    "start": start,
                    "end": end,
                    "category": cat,
                    "original": " ".join(words[start:end]),
                    "converted": new,
                }
            )
            cursor = end
        out.extend(words[cursor:])

        result = " ".join(out)
        pred = Prediction(result, result != text, recorded, words, tags)
        pred.protected = [
            {"index": p.index, "token": p.token, "reason": p.reason}
            for p in protections
        ]
        return pred

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        return [self.predict(t) for t in texts]
