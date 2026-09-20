"""Stage 2: the span converter -- a hybrid word + character model.

Runs **only** on spans the detector flagged, turning Arabic script back into
English: ``انترن`` -> ``intern``, ``ايه اي`` -> ``AI``.

Two heads over one shared encoder
---------------------------------
Measured on this corpus, the task is *nearly closed-vocabulary*: 1,245,593
training spans reduce to only 15,326 unique English targets, the top 1,000
covering 93.3% of test spans and the full lexicon 99.4%. Only 0.6% is genuinely
unseen, and that tail is mostly emails and URLs, which are compositional.

Free character generation was therefore doing difficult, error-prone work in the
99.4% of cases where a lookup is exact. The observed symptom was a decoder
producing ``devers`` and ``deversion`` for ``فريندس`` (gold ``friends``):
plausible English orthography, wrong word, because a from-scratch character
decoder has no lexical prior.

* **Word head** -- a softmax over the lexicon, read off mean-pooled encoder
  memory. When confident it emits a *whole dictionary word*, so a misspelling is
  impossible, and it runs no autoregressive loop at all.
* **Character head** -- the decoder below, used wherever the word head abstains.
  This is what keeps the vocabulary open for brand names, jargon and URLs.

``OOV`` is a trained class rather than a masked one, so the head learns to
*abstain* instead of forcing a wrong lexicon entry onto a word it has never seen.
Set ``use_word_head=False`` to fall back to pure character decoding.

Why character level for the fallback
------------------------------------
The mapping is phonetic, not lexical, so a character model generalises to words
no vocabulary contains. The alphabet is tiny (~130 symbols), keeping the model
small enough that latency stays negligible.

Why a separate model from the detector
--------------------------------------
Cost. Conversion is autoregressive, and running it over an entire sentence would
surrender the latency advantage that motivates the whole design. Confining it to
flagged spans means a typical sentence decodes a handful of short strings, and a
monolingual Arabic sentence decodes nothing at all.

The category token
------------------
The model is conditioned on the detector's category, because the same characters
convert differently depending on it: ``ايه اي`` is ``AI`` as an ACRONYM but would
be ``eh ay`` as plain CS.
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

# Delimiters that mark the span to convert inside its surrounding context.
# Single characters (not multi-char tags) so the character tokenizer handles
# them naturally, and guillemets because they never occur in Arabic ASR output.
SPAN_OPEN, SPAN_CLOSE = "‹", "›"

# Arabic letters actually produced by ASR, plus the Latin output alphabet.
_ARABIC_CHARS = "ابتثجحخدذرزسشصضطظعغفقكلمنهوىيئءأإآةپچڤگژ"
_ARABIC_MARKS = "ًٌٍَُِّْـ"
_LATIN_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DIGITS = "0123456789"
_PUNCT = " .,!?-_@/:'&+#%()" + SPAN_OPEN + SPAN_CLOSE


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
    # Capacity. The first trained model (d=256, 3+3 layers, ~9M params) plateaued
    # at 33% exact match with errors like "meting"/"frends" -- near-misses that
    # indicate too little capacity to settle on exact spellings, not a broken
    # objective. This is ~53M: six times larger, still far smaller than a
    # pretrained byte model, and it runs only on flagged spans.
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    # Source length now has to hold the context window, not just the span.
    max_src_len: int = 96
    max_tgt_len: int = 40
    # Words of surrounding context given to the converter on each side of the
    # span. 0 reproduces the old isolated-span behaviour.
    context_window: int = 3
    num_categories: int = field(default=len(CATEGORIES))
    # --- hybrid word head ---------------------------------------------------
    # Measured on the corpus: 1.25M spans reduce to 15,326 unique targets, the
    # top 1,000 covering 93% of test spans. A closed-vocabulary head answers
    # those exactly -- a whole dictionary word, so misspelling is impossible --
    # while the character decoder keeps the open vocabulary that brand names,
    # unseen jargon and URLs require.
    use_word_head: bool = True
    lexicon_size: int = 0          # filled in when the lexicon is built
    word_loss_weight: float = 1.0
    # Below this probability the word head abstains and the characters decide.
    # 0.90 is deliberately conservative: a wrong lexicon word is a worse failure
    # than a slightly misspelled character decode, because it is confidently
    # wrong. Lower it only if evaluation shows the head firing too rarely.
    word_confidence: float = 0.90
    # Label smoothing on the word head. Over a vocabulary of ~15K, a hard
    # one-hot target makes the head slow to become confident enough to fire at
    # all; a little smoothing calibrates it without hurting accuracy.
    word_label_smoothing: float = 0.05
    # --- non-autoregressive character head ---------------------------------
    # The autoregressive decoder produced runaway repetition when unsure
    # ("I'llllllllllllll", "commmmmmunits") -- a decode-loop pathology, not a
    # knowledge gap. This head predicts the output length and every character
    # position in ONE forward pass, so repetition is structurally impossible
    # and CPU inference needs no loop. 97.5% of targets are <= 10 characters,
    # so a fixed position budget covers the task comfortably.
    use_nar_head: bool = True
    nar_max_len: int = 24
    nar_loss_weight: float = 1.0

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

        # Word head: mean-pooled encoder state -> softmax over the lexicon.
        # It reads the encoder only (no decoding), so it costs one matmul and
        # removes the autoregressive loop entirely for the spans it answers.
        if cfg.use_word_head and cfg.lexicon_size > 0:
            self.word_head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, cfg.lexicon_size),
            )
        else:
            self.word_head = None

        # Non-autoregressive head: one length classifier plus a per-position
        # character classifier, both read off pooled / projected encoder state.
        if cfg.use_nar_head:
            self.nar_len_head = nn.Linear(cfg.d_model, cfg.nar_max_len + 1)
            self.nar_char_head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, cfg.nar_max_len * cfg.vocab_size),
            )
        else:
            self.nar_len_head = None
            self.nar_char_head = None

    def encode_memory(
        self, src: torch.Tensor, cat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the encoder once, returning memory and its padding mask."""
        src_pad = torch.cat(
            [torch.zeros(src.size(0), 1, dtype=torch.bool, device=src.device),
             src == PAD_ID],
            dim=1,
        )
        memory = self.transformer.encoder(
            self._encode(src, cat), src_key_padding_mask=src_pad
        )
        return memory, src_pad

    def _pool(self, memory: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        """Masked mean over encoder memory."""
        keep = (~src_pad).unsqueeze(-1).to(memory.dtype)
        return (memory * keep).sum(1) / keep.sum(1).clamp(min=1.0)

    def nar_logits(
        self, memory: torch.Tensor, src_pad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(length_logits, char_logits)`` for parallel decoding.

        char_logits is ``(batch, nar_max_len, vocab)`` -- every output position
        predicted at once, so no position can copy its predecessor and run away.
        """
        pooled = self._pool(memory, src_pad)
        length = self.nar_len_head(pooled)
        chars = self.nar_char_head(pooled).view(
            pooled.size(0), self.cfg.nar_max_len, self.cfg.vocab_size
        )
        return length, chars

    def word_logits(self, memory: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        """Lexicon logits from masked-mean-pooled encoder memory."""
        if self.word_head is None:
            raise RuntimeError("word head is disabled on this model")
        return self.word_head(self._pool(memory, src_pad))

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
        word_ids: torch.Tensor | None = None,
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

        # The word head reads the encoder half of the same forward pass, so
        # training both heads costs one extra matmul, not a second model.
        if self.word_head is not None or self.nar_char_head is not None:
            memory = self.transformer.encoder(
                mem_in, src_key_padding_mask=src_pad
            )
            if self.word_head is not None:
                res["word_logits"] = self.word_logits(memory, src_pad)
            if self.nar_char_head is not None:
                nl, nc = self.nar_logits(memory, src_pad)
                res["nar_len_logits"], res["nar_char_logits"] = nl, nc

        if tgt_out is not None:
            char_loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                tgt_out.reshape(-1),
                ignore_index=PAD_ID,
                label_smoothing=0.1,
            )
            res["char_loss"] = char_loss.detach()
            loss = char_loss
            if self.nar_char_head is not None and tgt_out is not None:
                # Target characters, minus EOS/PAD, laid out at fixed positions.
                b, L = tgt_out.size(0), self.cfg.nar_max_len
                flat = torch.full((b, L), PAD_ID, dtype=torch.long, device=tgt_out.device)
                lengths = torch.zeros(b, dtype=torch.long, device=tgt_out.device)
                for j in range(b):
                    row = [t for t in tgt_out[j].tolist() if t not in (PAD_ID, EOS_ID, BOS_ID)]
                    row = row[:L]
                    lengths[j] = len(row)
                    if row:
                        flat[j, : len(row)] = torch.tensor(row, device=tgt_out.device)
                len_loss = F.cross_entropy(res["nar_len_logits"], lengths)
                char_loss_nar = F.cross_entropy(
                    res["nar_char_logits"].reshape(-1, self.cfg.vocab_size),
                    flat.reshape(-1),
                    ignore_index=PAD_ID,
                )
                nar = len_loss + char_loss_nar
                res["nar_loss"] = nar.detach()
                loss = loss + self.cfg.nar_loss_weight * nar

            if self.word_head is not None and word_ids is not None:
                # OOV spans are kept in this loss (not ignored): the head must
                # learn to *predict* OOV so it abstains rather than guessing a
                # wrong lexicon word for a brand name it has never seen.
                word_loss = F.cross_entropy(
                    res["word_logits"], word_ids,
                    label_smoothing=self.cfg.word_label_smoothing,
                )
                res["word_loss"] = word_loss.detach()
                loss = loss + self.cfg.word_loss_weight * word_loss
            res["loss"] = loss
        return res

    @torch.no_grad()
    def nar_decode(self, src: torch.Tensor, cat: torch.Tensor) -> list[str]:
        """Decode every span in ONE forward pass -- no autoregressive loop.

        Because each position is predicted independently from the encoder state,
        a position cannot condition on (and therefore cannot repeat) the one
        before it. The runaway outputs the autoregressive decoder produced when
        unsure -- "I'llllllllllllll", "commmmmmunits" -- are impossible here.
        """
        memory, src_pad = self.encode_memory(src, cat)
        len_logits, char_logits = self.nar_logits(memory, src_pad)
        lengths = len_logits.argmax(-1).clamp(0, self.cfg.nar_max_len)
        chars = char_logits.argmax(-1)
        out: list[str] = []
        for row, n in zip(chars.cpu().tolist(), lengths.cpu().tolist()):
            out.append("".join(ITOS.get(c, "") for c in row[:n] if c > UNK_ID))
        return out

    @torch.no_grad()
    def predict_words(
        self, src: torch.Tensor, cat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Word-head prediction: ``(word_id, probability)`` per span.

        No decoding loop -- one encoder pass and a softmax.
        """
        memory, src_pad = self.encode_memory(src, cat)
        probs = self.word_logits(memory, src_pad).softmax(-1)
        conf, idx = probs.max(-1)
        return idx, conf

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
