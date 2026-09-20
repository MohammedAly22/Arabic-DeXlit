"""Dual-conditioned sequence-to-sequence rewriter.

What this is
------------
A full-sentence encoder-decoder whose decoder is conditioned on **two** inputs at
once: the source text, and the detector's tag sequence. The tags are not a hard
routing switch here -- they are a learned signal telling the decoder where to
copy and where to convert, which it can weigh against the text itself.

Why it beats span routing
-------------------------
Routing each detected span through an isolated converter loses the relationships
*between* spans. ``اورانج انوفيشن ايجيبت`` converted span-by-span produced
``Orange Inovation Egyph`` plus a stray ``Egypt``, because no component ever saw
the three pieces as one name. A decoder that attends over the whole sentence can.

The copy problem, and how it is solved
--------------------------------------
Measured on this corpus, **84% of output tokens appear verbatim in the input**.
A plain decoder must learn to reproduce those from scratch, and every one it
gets wrong is text corrupted that nobody asked it to touch -- the exact failure
the pass-through guarantee exists to prevent.

So generation is an explicit mixture:

    P(token) = p_gen * P_vocab(token) + (1 - p_gen) * P_copy(token)

``p_gen`` is predicted per step. ``P_copy`` is the decoder's attention over the
source, so copying is a *pointer* into the input rather than a reconstruction
from a vocabulary. At ``p_gen -> 0`` the model reproduces its input exactly,
which makes pass-through the model's easiest behaviour rather than its hardest.

Conditioning on tags
--------------------
Each source position gets its tag embedding added to its token embedding before
encoding, so every encoder state carries "what the detector thought of this
token". The decoder then attends over tag-aware memory. A gold-tag channel is
used during training and the detector's predictions at inference; dropping tags
at random during training (``tag_dropout``) stops the decoder trusting them
blindly, so a detector mistake degrades the output instead of dictating it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import NUM_TAGS

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


@dataclass
class DualSeq2SeqConfig:
    """Everything that defines the rewriter, saved next to the weights."""

    vocab_size: int = 32000
    num_tags: int = NUM_TAGS
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_src_len: int = 48
    max_tgt_len: int = 48

    # The copy mechanism. Off, this is an ordinary seq2seq and 84% of the task
    # becomes memorisation; on, copying is a pointer into the source.
    use_copy: bool = True
    # Tags are dropped this often during training so the decoder cannot treat
    # the detector as infallible -- it must still read the text.
    tag_dropout: float = 0.1
    label_smoothing: float = 0.1

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "DualSeq2SeqConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class SinusoidalPositions(nn.Module):
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


class DualConditionedRewriter(nn.Module):
    """Encoder-decoder conditioned jointly on source tokens and detector tags."""

    def __init__(self, cfg: DualSeq2SeqConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD_ID)
        # Tag embeddings are *added* to token embeddings rather than concatenated,
        # so every encoder state is tag-aware without widening the model.
        self.tag_embed = nn.Embedding(cfg.num_tags + 1, cfg.d_model, padding_idx=cfg.num_tags)
        self.no_tag_id = cfg.num_tags  # used when tags are dropped or unavailable

        self.pos = SinusoidalPositions(cfg.d_model, max(cfg.max_src_len, cfg.max_tgt_len) + 8)
        self.drop = nn.Dropout(cfg.dropout)

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

        if cfg.use_copy:
            # Cross-attention used purely as a pointer distribution over source
            # positions, kept separate from the decoder's own attention so the
            # copy scores stay interpretable.
            self.copy_attn = nn.MultiheadAttention(
                cfg.d_model, num_heads=1, dropout=cfg.dropout, batch_first=True
            )
            # p_gen: generate from vocabulary, or copy from the source?
            self.gen_gate = nn.Linear(cfg.d_model * 2, 1)

        self.scale = cfg.d_model ** 0.5

    # --- encoding ---------------------------------------------------------
    def encode(
        self,
        src: torch.Tensor,
        tags: torch.Tensor | None = None,
        *,
        training_dropout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode source tokens together with their tags."""
        pad_mask = src == PAD_ID
        x = self.token_embed(src) * self.scale

        if tags is None:
            tags = torch.full_like(src, self.no_tag_id)
        elif training_dropout and self.cfg.tag_dropout > 0:
            drop = torch.rand_like(src, dtype=torch.float) < self.cfg.tag_dropout
            tags = torch.where(drop, torch.full_like(tags, self.no_tag_id), tags)
        x = x + self.tag_embed(tags)

        memory = self.transformer.encoder(
            self.drop(self.pos(x)), src_key_padding_mask=pad_mask
        )
        return memory, pad_mask

    # --- decoding ---------------------------------------------------------
    def _decode_step(
        self,
        memory: torch.Tensor,
        pad_mask: torch.Tensor,
        tgt_in: torch.Tensor,
        src: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Log-probabilities over the vocabulary, plus the copy attention.

        The attention is returned because a copied token may be ``<unk>`` in the
        vocabulary: the only faithful rendering is the *source word at the
        position that was pointed to*, which the ids alone cannot express.
        """
        y = self.token_embed(tgt_in) * self.scale
        causal = nn.Transformer.generate_square_subsequent_mask(
            tgt_in.size(1), device=tgt_in.device
        )
        h = self.transformer.decoder(
            self.drop(self.pos(y)),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=(tgt_in == PAD_ID),
            memory_key_padding_mask=pad_mask,
        )
        gen_logits = self.out(h)

        if not self.cfg.use_copy:
            return F.log_softmax(gen_logits, dim=-1), None

        # Pointer distribution over source positions.
        ctx, attn = self.copy_attn(
            h, memory, memory, key_padding_mask=pad_mask, need_weights=True
        )
        p_gen = torch.sigmoid(self.gen_gate(torch.cat([h, ctx], dim=-1)))

        gen_p = F.softmax(gen_logits, dim=-1) * p_gen
        # Scatter attention mass onto the vocabulary ids actually present in the
        # source: copying is choosing an input position, not a vocabulary entry.
        copy_p = torch.zeros_like(gen_p)
        src_ids = src.unsqueeze(1).expand(-1, attn.size(1), -1)
        copy_p.scatter_add_(2, src_ids, attn * (1.0 - p_gen))

        return torch.log(gen_p + copy_p + 1e-10), attn * (1.0 - p_gen)

    def forward(
        self,
        src: torch.Tensor,
        tags: torch.Tensor | None,
        tgt_in: torch.Tensor,
        tgt_out: torch.Tensor | None = None,
    ) -> dict:
        memory, pad_mask = self.encode(src, tags, training_dropout=self.training)
        logprobs, copy_attn = self._decode_step(memory, pad_mask, tgt_in, src)

        res: dict = {"logprobs": logprobs}
        if copy_attn is not None:
            res["copy_attn"] = copy_attn
        if tgt_out is not None:
            # NLL rather than cross_entropy: the copy mixture already produced a
            # normalised distribution, so applying another softmax would be wrong.
            flat = logprobs.reshape(-1, self.cfg.vocab_size)
            gold = tgt_out.reshape(-1)
            keep = gold != PAD_ID

            nll = -flat.gather(1, gold.clamp(min=0).unsqueeze(1)).squeeze(1)
            if self.cfg.label_smoothing > 0:
                # Smoothing is applied by hand because F.nll_loss does not take
                # it -- and cross_entropy, which does, would re-normalise a
                # distribution that is already a probability.
                eps = self.cfg.label_smoothing
                smooth = -flat.mean(dim=1)
                nll = (1.0 - eps) * nll + eps * smooth
            res["loss"] = nll[keep].mean() if bool(keep.any()) else nll.sum() * 0.0
        return res

    # --- inference --------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        tags: torch.Tensor | None = None,
        *,
        max_len: int | None = None,
        return_copy: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Greedy decoding. Output length is bounded relative to the input, since
        a rewrite is never much longer than what it rewrites.

        With ``return_copy``, also returns the source position each step pointed
        at (-1 where the step generated rather than copied), so an out-of-
        vocabulary copy can be rendered as the original word.
        """
        max_len = min(max_len or self.cfg.max_tgt_len, src.size(1) + 8)
        memory, pad_mask = self.encode(src, tags)

        b = src.size(0)
        ys = torch.full((b, 1), BOS_ID, dtype=torch.long, device=src.device)
        done = torch.zeros(b, dtype=torch.bool, device=src.device)
        copy_src: list[torch.Tensor] = []
        for _ in range(max_len):
            lp, attn = self._decode_step(memory, pad_mask, ys, src)
            nxt = lp[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
            if attn is not None:
                weight, pos = attn[:, -1].max(-1)
                # Only treat it as a copy when the pointer carries real mass;
                # otherwise the step genuinely generated.
                copy_src.append(torch.where(weight > 0.5, pos, torch.full_like(pos, -1)))
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done |= nxt == EOS_ID
            if bool(done.all()):
                break
        out = ys[:, 1:]
        if return_copy:
            cs = (
                torch.stack(copy_src, dim=1)
                if copy_src
                else torch.full_like(out, -1)
            )
            return out, cs[:, : out.size(1)]
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
