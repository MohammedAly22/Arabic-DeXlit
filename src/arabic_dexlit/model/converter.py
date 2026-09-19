"""Stage 2: the character-level span converter.

Runs **only** on spans the detector flagged, mapping Arabic characters to Latin
ones: ``انترن`` -> ``intern``, ``ايه اي`` -> ``AI``.

Why character level
-------------------
The mapping is phonetic, not lexical. A word-level model could only emit English
words it saw in training; a character model generalises to unseen ones, which
matters because the long tail here is company names, product names and jargon
that no fixed vocabulary covers. The alphabet is tiny (~80 symbols), so the
model stays small enough to add negligible latency.

Why a separate model from the detector
--------------------------------------
Cost. Conversion is autoregressive, and running it over an entire sentence would
surrender the latency advantage that motivates the whole design. Confining it to
flagged spans means a typical sentence decodes a handful of short strings, and a
monolingual Arabic sentence decodes nothing at all.

The category token
------------------
The decoder is conditioned on the detector's category, because the same input
characters have different correct outputs depending on it: ``ايه اي`` is ``AI``
as an ACRONYM but would be ``eh ay`` as plain CS. Passing the category makes the
conversion policy explicit instead of something the model must guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import CATEGORIES

# --- vocabulary ------------------------------------------------------------
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

# Arabic letters actually produced by ASR, plus the Latin output alphabet.
_ARABIC_CHARS = "ابتثجحخدذرزسشصضطظعغفقكلمنهوىيئءأإآةپچڤگژ"
_ARABIC_MARKS = "ًٌٍَُِّْـ"
_LATIN_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DIGITS = "0123456789"
_PUNCT = " .,!?-_@/:'&+#%()"


def build_vocab() -> tuple[dict[str, int], dict[int, str]]:
    """Fixed, deterministic vocabulary -- checkpoints stay compatible."""
    chars = (
        list(_ARABIC_CHARS) + list(_ARABIC_MARKS) + list(_LATIN_CHARS)
        + list(_DIGITS) + list(_PUNCT)
    )
    seen: list[str] = []
    for c in chars:
        if c not in seen:
            seen.append(c)
    stoi = {t: i for i, t in enumerate(SPECIALS)}
    for c in seen:
        stoi.setdefault(c, len(stoi))
    return stoi, {i: t for t, i in stoi.items()}


STOI, ITOS = build_vocab()
VOCAB_SIZE = len(STOI)
CATEGORY_IDS = {c: i for i, c in enumerate(CATEGORIES)}


def encode_chars(text: str, max_len: int, *, add_eos: bool = True) -> list[int]:
    ids = [STOI.get(c, UNK_ID) for c in text[: max_len - 1]]
    if add_eos:
        ids.append(EOS_ID)
    return ids


def decode_ids(ids: list[int]) -> str:
    out = []
    for i in ids:
        if i in (PAD_ID, BOS_ID):
            continue
        if i == EOS_ID:
            break
        out.append(ITOS.get(i, ""))
    return "".join(out)


@dataclass
class ConverterConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 768
    dropout: float = 0.1
    max_src_len: int = 48
    max_tgt_len: int = 40
    num_categories: int = field(default=len(CATEGORIES))

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "ConverterConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positions. Spans are short, so nothing fancier pays."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SpanConverter(nn.Module):
    """Tiny char-level seq2seq transformer, conditioned on span category."""

    def __init__(self, cfg: ConverterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.cat_embed = nn.Embedding(cfg.num_categories, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, max(cfg.max_src_len, cfg.max_tgt_len) + 8)
        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.scale = cfg.d_model ** 0.5

    def _encode(self, src: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        x = self.embed(src) * self.scale
        # The category is prepended as a pseudo-token so every encoder position
        # can attend to it, rather than being added into position 0 only.
        x = torch.cat([self.cat_embed(cat).unsqueeze(1), x], dim=1)
        return self.pos(x)

    def forward(
        self,
        src: torch.Tensor,
        cat: torch.Tensor,
        tgt_in: torch.Tensor,
        tgt_out: torch.Tensor | None = None,
    ) -> dict:
        src_pad = torch.cat(
            [torch.zeros(src.size(0), 1, dtype=torch.bool, device=src.device),
             src == PAD_ID],
            dim=1,
        )
        mem_in = self._encode(src, cat)
        tgt_emb = self.pos(self.embed(tgt_in) * self.scale)
        causal = nn.Transformer.generate_square_subsequent_mask(
            tgt_in.size(1), device=tgt_in.device
        )
        h = self.transformer(
            mem_in,
            tgt_emb,
            tgt_mask=causal,
            src_key_padding_mask=src_pad,
            memory_key_padding_mask=src_pad,
            tgt_key_padding_mask=(tgt_in == PAD_ID),
        )
        logits = self.out(h)
        res = {"logits": logits}
        if tgt_out is not None:
            res["loss"] = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                tgt_out.reshape(-1),
                ignore_index=PAD_ID,
                label_smoothing=0.1,
            )
        return res

    @torch.no_grad()
    def greedy_decode(
        self, src: torch.Tensor, cat: torch.Tensor, max_len: int | None = None
    ) -> torch.Tensor:
        """Greedy decoding. Spans are short and near-deterministic, so beam
        search buys accuracy that does not justify the latency here.

        ``max_len`` is additionally bounded relative to the input: a transliterated
        span is never much longer than its English form, so an undertrained (or
        confused) decoder that fails to emit EOS is cut off early instead of
        running to the full length limit on every span.
        """
        budget = int(src.size(1) * 1.5) + 4
        max_len = min(max_len or self.cfg.max_tgt_len, budget)
        b = src.size(0)
        src_pad = torch.cat(
            [torch.zeros(b, 1, dtype=torch.bool, device=src.device), src == PAD_ID], dim=1
        )
        memory = self.transformer.encoder(
            self._encode(src, cat), src_key_padding_mask=src_pad
        )
        ys = torch.full((b, 1), BOS_ID, dtype=torch.long, device=src.device)
        finished = torch.zeros(b, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            tgt_emb = self.pos(self.embed(ys) * self.scale)
            causal = nn.Transformer.generate_square_subsequent_mask(
                ys.size(1), device=ys.device
            )
            h = self.transformer.decoder(
                tgt_emb, memory, tgt_mask=causal, memory_key_padding_mask=src_pad
            )
            nxt = self.out(h[:, -1]).argmax(-1)
            nxt = torch.where(finished, torch.full_like(nxt, PAD_ID), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            finished |= nxt == EOS_ID
            if bool(finished.all()):
                break
        return ys[:, 1:]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
