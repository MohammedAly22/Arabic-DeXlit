"""Stage 1: the span detector.

What it does
------------
Tags every input token: ``O`` (leave it alone) or ``B-/I-<category>`` (this
token is part of a transliterated span that must be converted).

Why tagging rather than generation
----------------------------------
A seq2seq model rewrites the whole sentence, which makes two of this project's
hard requirements impossible to honour:

* **Pass-through.** A decoder is free to alter any token, so "pure Arabic must
  come back unchanged" can only ever be *approximately* learned. Here, ``O``
  means the token is copied verbatim, so an all-``O`` sentence is returned byte
  for byte. The guarantee is structural.
* **Latency.** A decoder runs over every token of every sentence. This runs one
  encoder pass and then converts only the few spans that were flagged -- and for
  monolingual Arabic input, stage 2 never runs at all.

Architectural additions over a plain token classifier
-----------------------------------------------------
Two pieces are specific to this task rather than generic NER:

1. **Script-feature injection.** Whether a character is Arabic, Latin, a digit or
   punctuation is perfectly known at inference time -- it is a property of the
   string, not something to infer. Feeding it in as an explicit embedding frees
   the encoder from rediscovering it and sharply helps on the already-Latin and
   digit cases, which must be left alone.
2. **A copy-gate head.** A scalar per token predicting "is this token untouched",
   trained jointly with the tag head. It gives a calibrated, directly
   thresholdable pass-through signal, and at inference it can veto spurious
   edits -- the conservative direction for a model that must not corrupt text.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from ..schema import IGNORE_INDEX, NUM_TAGS

# --- script features -------------------------------------------------------
# Computed from the raw token string, never learned. Cheap and unambiguous.
SCRIPT_ARABIC = 0
SCRIPT_LATIN = 1
SCRIPT_DIGIT = 2
SCRIPT_PUNCT = 3
SCRIPT_MIXED = 4
SCRIPT_OTHER = 5
NUM_SCRIPTS = 6


def script_of(token: str) -> int:
    """Classify a token's script. Pure string inspection -- no model involved."""
    has_ar = has_lat = has_dig = False
    has_other = False
    for ch in token:
        o = ord(ch)
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:
            has_ar = True
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            has_lat = True
        elif ch.isdigit():
            has_dig = True
        elif ch.isalnum():
            has_other = True
    kinds = sum((has_ar, has_lat, has_dig))
    if kinds > 1:
        return SCRIPT_MIXED
    if has_ar:
        return SCRIPT_ARABIC
    if has_lat:
        return SCRIPT_LATIN
    if has_dig:
        return SCRIPT_DIGIT
    if has_other:
        return SCRIPT_OTHER
    return SCRIPT_PUNCT


@dataclass
class DetectorConfig:
    """Everything that defines a detector, saved next to the weights."""

    encoder_name: str = "UBC-NLP/MARBERTv2"
    num_tags: int = NUM_TAGS
    dropout: float = 0.1
    script_embed_dim: int = 32
    use_script_features: bool = True
    use_copy_gate: bool = True
    copy_gate_weight: float = 0.3
    # Class weight applied to non-O tags. Spans are a small minority of tokens,
    # so without this the model can score well by predicting O everywhere.
    positive_weight: float = 2.0
    # Per-category weighting on top of that. The corpus is 98.9% plain CS, with
    # ACRONYM at 0.66%, EMAIL 0.32% and ENTITY 0.14%; a uniform positive weight
    # leaves the rare categories badly under-trained, which showed up as an
    # ENTITY ("Orange Innovation Egypt") being mis-tagged EMAIL and converted to
    # "info@gening". Weights are computed from the data at build time.
    category_weights: dict | None = None
    freeze_encoder_layers: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "DetectorConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class SpanDetector(nn.Module):
    """Token tagger with script features and a copy gate."""

    def __init__(self, cfg: DetectorConfig, encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        if encoder is not None:
            self.encoder = encoder
            hidden = self.encoder.config.hidden_size
        else:
            enc_cfg = AutoConfig.from_pretrained(cfg.encoder_name)
            self.encoder = AutoModel.from_pretrained(cfg.encoder_name, config=enc_cfg)
            hidden = enc_cfg.hidden_size

        if cfg.freeze_encoder_layers:
            self._freeze_bottom(cfg.freeze_encoder_layers)

        feat = hidden
        if cfg.use_script_features:
            self.script_embed = nn.Embedding(NUM_SCRIPTS, cfg.script_embed_dim)
            feat += cfg.script_embed_dim

        self.dropout = nn.Dropout(cfg.dropout)
        self.tag_head = nn.Linear(feat, cfg.num_tags)
        if cfg.use_copy_gate:
            self.copy_head = nn.Linear(feat, 1)

        # Weight non-O classes up so rare spans are not drowned out by O, then
        # scale each category by its own rarity.
        w = torch.ones(cfg.num_tags)
        w[1:] = cfg.positive_weight
        if cfg.category_weights:
            from ..schema import ID2TAG, category_of

            for i in range(1, cfg.num_tags):
                cat = category_of(ID2TAG[i])
                if cat and cat in cfg.category_weights:
                    w[i] = cfg.positive_weight * float(cfg.category_weights[cat])
        self.register_buffer("class_weight", w)

    def _freeze_bottom(self, n: int) -> None:
        """Freeze embeddings and the lowest ``n`` layers -- faster, less overfit."""
        emb = getattr(self.encoder, "embeddings", None)
        if emb is not None:
            for p in emb.parameters():
                p.requires_grad = False
        layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
        if layers is not None:
            for layer in layers[:n]:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        script_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        copy_labels: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> dict:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        h = out.last_hidden_state

        if self.cfg.use_script_features:
            if script_ids is None:
                script_ids = torch.zeros_like(input_ids)
            h = torch.cat([h, self.script_embed(script_ids)], dim=-1)

        h = self.dropout(h)
        logits = self.tag_head(h)

        result: dict = {"logits": logits}
        if output_attentions:
            result["attentions"] = out.attentions
        if self.cfg.use_copy_gate:
            result["copy_logits"] = self.copy_head(h).squeeze(-1)

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.num_tags),
                labels.view(-1),
                weight=self.class_weight,
                ignore_index=IGNORE_INDEX,
            )
            result["tag_loss"] = loss.detach()
            if self.cfg.use_copy_gate and copy_labels is not None:
                valid = labels.view(-1) != IGNORE_INDEX
                cl = F.binary_cross_entropy_with_logits(
                    result["copy_logits"].view(-1)[valid],
                    copy_labels.view(-1)[valid].float(),
                )
                result["copy_loss"] = cl.detach()
                loss = loss + self.cfg.copy_gate_weight * cl
            result["loss"] = loss
        return result

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        script_ids: torch.Tensor | None = None,
        copy_threshold: float | None = None,
    ) -> torch.Tensor:
        """Predicted tag ids, optionally vetoed by the copy gate.

        With ``copy_threshold`` set, a token the gate is confident about keeping
        is forced to ``O`` even if the tag head wanted to edit it. Erring toward
        leaving text alone is the safe failure mode for this model.
        """
        out = self.forward(input_ids, attention_mask, script_ids)
        pred = out["logits"].argmax(-1)
        if copy_threshold is not None and "copy_logits" in out:
            keep = torch.sigmoid(out["copy_logits"]) > copy_threshold
            pred = torch.where(keep, torch.zeros_like(pred), pred)
        return pred
