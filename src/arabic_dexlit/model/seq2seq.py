"""End-to-end sentence rewriter built on ByT5.

The problem with the previous design
------------------------------------
Stage 2 was split across five components -- parsers, a gazetteer, an acronym
inventory, phonetic retrieval, a span converter -- each seeing only a fragment
of the sentence. That produced failures no single component could have avoided:

* ``سي بلاس بلاس`` -> ``Scele leles``  (a span converter cannot know this is C++)
* ``ايفالويشن``    -> ``finish``       (retrieval matched the wrong word)
* ``لتوبيك``       -> ``toticeoe``     (the ``ل`` prefix broke the phonetic key)
* ``ال``           -> ``I'll``         (the article converted as if it were English)
* ``البريزنتيشن``  -> unchanged        (the detector missed it entirely)

Every one of these is a *context* failure. The information needed to get them
right is in the sentence; it simply never reached the component making the
decision.

Why ByT5
--------
Byte-level, so the vocabulary is 256 symbols and **nothing is ever out of
vocabulary**. That matters concretely here:

* ``C++`` is three bytes, not an unknown token;
* ``الsystem`` needs no word segmentation to be understood;
* a brand name never seen in training is still representable exactly;
* Arabic orthographic variation (``ميتينج`` / ``ميتنج`` / ``ميطنغ``) is a small
  edit in byte space rather than three unrelated vocabulary entries.

Published comparisons find byte-level ByT5 substantially ahead of subword mT5 on
transliteration and other spelling-sensitive tasks, which is exactly this task.
The cost is sequence length -- Arabic is two bytes per character -- but measured
on this corpus the 99th percentile is 393 bytes, so a 512-byte budget covers
effectively everything.

The decoder's cross-attention is the mechanism the whole design rests on: when
emitting ``Presentation`` it attends to ``البريزنتيشن``, and when emitting
``the`` it attends to ``ال``. Nothing is routed, matched or looked up.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Seq2SeqConfig:
    """Everything that defines the rewriter, saved next to the weights."""

    # ByT5 is the default because this task is spelling-sensitive and
    # open-vocabulary. mT5 or AraT5 can be substituted for a subword baseline.
    model_name: str = "google/byt5-base"

    # Byte budgets. Arabic costs ~2 bytes per character, so these are roughly
    # 256 characters. p99 of the corpus is 393 bytes.
    max_source_length: int = 512
    max_target_length: int = 512

    # A short natural-language prefix, which is how T5-family models are told
    # which task they are performing.
    task_prefix: str = "restore code-switching: "

    label_smoothing: float = 0.1
    dropout_rate: float = 0.1

    # Generation. Beam search matters more here than for most seq2seq work:
    # the difference between "Presentation" and "Presantation" is a single byte
    # deep inside the sequence, where greedy decoding cannot recover.
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0   # byte-level: n-gram blocking would corrupt words

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "Seq2SeqConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def load_model_and_tokenizer(cfg: Seq2SeqConfig):
    """Load the pretrained encoder-decoder and its tokenizer."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    if hasattr(model.config, "dropout_rate"):
        model.config.dropout_rate = cfg.dropout_rate
    return model, tok


def encode_example(
    tokenizer,
    source: str,
    target: str | None,
    cfg: Seq2SeqConfig,
) -> dict:
    """Encode one (source, target) pair for training or inference."""
    model_inputs = tokenizer(
        cfg.task_prefix + source,
        max_length=cfg.max_source_length,
        truncation=True,
    )
    if target is not None:
        labels = tokenizer(
            target, max_length=cfg.max_target_length, truncation=True
        )["input_ids"]
        # -100 marks padding as ignored by the loss.
        model_inputs["labels"] = labels
    return model_inputs


@torch.no_grad()
def generate(
    model,
    tokenizer,
    sentences: list[str],
    cfg: Seq2SeqConfig,
    device: torch.device | str = "cpu",
) -> list[str]:
    """Rewrite whole sentences. No routing, no rules, no post-processing."""
    if not sentences:
        return []
    batch = tokenizer(
        [cfg.task_prefix + s for s in sentences],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_source_length,
    ).to(device)

    out = model.generate(
        **batch,
        max_length=cfg.max_target_length,
        num_beams=cfg.num_beams,
        length_penalty=cfg.length_penalty,
        no_repeat_ngram_size=cfg.no_repeat_ngram_size,
        early_stopping=cfg.num_beams > 1,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)
