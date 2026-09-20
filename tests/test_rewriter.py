"""Tests for the dual-conditioned rewriter.

The test that matters most is ``test_tags_change_the_output``: the whole premise
is that the decoder conditions on BOTH text and tags, and a model that silently
ignored the tag channel would still pass every shape and gradient check.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_dexlit.model.dual_seq2seq import (  # noqa: E402
    BOS_ID,
    DualConditionedRewriter,
    DualSeq2SeqConfig,
)
from arabic_dexlit.model.word_vocab import UNK, WordVocab  # noqa: E402
from arabic_dexlit.schema import TAG2ID  # noqa: E402

M = "\u0645\u064a\u062a\u064a\u0646\u062c"
X = "\u0639\u0646\u062f\u064a"
Y = "\u0645\u0647\u0645"

ROWS = [
    {"src_tokens": [X, M, Y], "tags": ["O", "B-CS", "O"], "tgt": f"{X} meeting {Y}"},
    {"src_tokens": [X, M, Y], "tags": ["O", "O", "O"], "tgt": f"{X} {M} {Y}"},
]


def _tiny(vocab_size: int) -> DualConditionedRewriter:
    cfg = DualSeq2SeqConfig(
        vocab_size=vocab_size, d_model=64, nhead=2, num_encoder_layers=1,
        num_decoder_layers=1, dim_feedforward=128, dropout=0.0,
        tag_dropout=0.0, max_src_len=10, max_tgt_len=10,
    )
    return DualConditionedRewriter(cfg)


def _batch(rows, vocab, cfg):
    S = [vocab.encode(r["src_tokens"], cfg.max_src_len) for r in rows]
    T = [vocab.encode(r["tgt"].split(), cfg.max_tgt_len) for r in rows]
    G = [[TAG2ID[t] for t in r["tags"]][: len(s)] for r, s in zip(rows, S)]
    n, ml = max(map(len, S)), max(map(len, T))
    src = torch.tensor([s + [0] * (n - len(s)) for s in S])
    tags = torch.tensor([g + [cfg.num_tags] * (n - len(g)) for g in G])
    tgt = torch.tensor([t + [0] * (ml - len(t)) for t in T])
    tin = torch.cat([torch.full((len(rows), 1), BOS_ID), tgt[:, :-1]], 1)
    return src, tags, tin, tgt


def test_forward_shapes_and_gradients():
    vocab = WordVocab.build(ROWS, min_count=1)
    m = _tiny(len(vocab))
    src, tags, tin, tout = _batch(ROWS, vocab, m.cfg)
    out = m(src, tags, tin, tout)
    assert out["logprobs"].shape == (2, tout.size(1), len(vocab))
    assert "copy_attn" in out          # copy mechanism is active
    out["loss"].backward()
    assert any(p.grad is not None for p in m.parameters())


def test_logprobs_are_normalised():
    """The copy mixture must stay a probability distribution."""
    vocab = WordVocab.build(ROWS, min_count=1)
    m = _tiny(len(vocab)).eval()
    src, tags, tin, _ = _batch(ROWS, vocab, m.cfg)
    lp = m(src, tags, tin)["logprobs"]
    total = lp.exp().sum(-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-3), total


def test_tags_change_the_output():
    """The same source with different tags must produce different output.

    This is the architecture's whole claim. A model ignoring the tag channel
    cannot get both of these right, since the inputs are character-identical.
    """
    vocab = WordVocab.build(ROWS, min_count=1)
    m = _tiny(len(vocab))
    src, tags, tin, tout = _batch(ROWS, vocab, m.cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    m.train()
    for _ in range(400):
        out = m(src, tags, tin, tout)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()

    m.eval()
    gen = m.generate(src, tags)
    preds = [" ".join(vocab.decode(r)) for r in gen.cpu().tolist()]
    assert preds[0] == ROWS[0]["tgt"], preds[0]   # tagged B-CS -> converted
    assert preds[1] == ROWS[1]["tgt"], preds[1]   # tagged O    -> unchanged
    assert preds[0] != preds[1]


def test_copy_recovers_out_of_vocabulary_words():
    """An OOV copy must render as the source word, not <unk>."""
    vocab = WordVocab(["a", "b"])           # deliberately tiny
    rare = "\u0627\u0646\u062a\u064a\u0644\u0627\u0646\u0633"
    assert vocab.encode([rare], 8)[0] == 3  # it really is <unk>
    # With a copy pointer at position 1, decode must yield the source word.
    got = vocab.decode([3], src_tokens=["x", rare, "y"], copy_positions=[1])
    assert got == [rare], got
    # Without the pointer there is nothing to recover from.
    assert vocab.decode([3]) == [UNK]


def test_generate_respects_length_bound():
    vocab = WordVocab.build(ROWS, min_count=1)
    m = _tiny(len(vocab)).eval()
    src, tags, _, _ = _batch(ROWS, vocab, m.cfg)
    gen = m.generate(src, tags)
    assert gen.size(1) <= src.size(1) + 8


def test_runs_without_tags():
    """Tags are optional: the model must still produce output without them."""
    vocab = WordVocab.build(ROWS, min_count=1)
    m = _tiny(len(vocab)).eval()
    src, _, _, _ = _batch(ROWS, vocab, m.cfg)
    assert m.generate(src, None).shape[0] == 2


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("\nall green" if not failed else f"\n{failed} failing")
    sys.exit(1 if failed else 0)
