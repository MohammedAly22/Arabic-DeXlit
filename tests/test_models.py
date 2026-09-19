"""Shape and behaviour tests for both model stages.

These use a tiny randomly-initialised encoder rather than downloading MARBERTv2,
so the suite runs offline and in seconds while still exercising the real code
paths: label projection, script features, the copy gate, and greedy decoding.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_dexlit.model.converter import (  # noqa: E402
    BOS_ID,
    ConverterConfig,
    SpanConverter,
    decode_ids,
    encode_chars,
)
from arabic_dexlit.model.detector import (  # noqa: E402
    DetectorConfig,
    SpanDetector,
    script_of,
)
from arabic_dexlit.model.lexicon import OOV, OOV_ID, Lexicon  # noqa: E402
from arabic_dexlit.schema import IGNORE_INDEX, NUM_TAGS  # noqa: E402


def _tiny_detector() -> SpanDetector:
    """A SpanDetector on a 2-layer random BERT -- no network access needed."""
    from transformers import BertConfig, BertModel

    enc = BertModel(
        BertConfig(
            vocab_size=200, hidden_size=64, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=128, max_position_embeddings=64,
        )
    )
    return SpanDetector(DetectorConfig(num_tags=NUM_TAGS, script_embed_dim=8), encoder=enc)


def test_script_features_classify_correctly():
    assert script_of("انترن") == 0   # Arabic
    assert script_of("intern") == 1                             # Latin
    assert script_of("2024") == 2                               # digits
    assert script_of("...") == 4 or script_of("...") == 3       # punctuation
    assert script_of("AI2") == 4                                # mixed


def test_detector_forward_and_loss():
    m = _tiny_detector()
    ids = torch.randint(5, 199, (2, 12))
    mask = torch.ones_like(ids)
    scripts = torch.zeros_like(ids)
    labels = torch.randint(0, NUM_TAGS, (2, 12))
    labels[:, 0] = IGNORE_INDEX  # [CLS] carries no label
    out = m(ids, mask, scripts, labels=labels, copy_labels=(labels == 0).long())
    assert out["logits"].shape == (2, 12, NUM_TAGS)
    assert out["loss"].requires_grad
    out["loss"].backward()  # gradients must flow
    assert any(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_copy_gate_can_veto_edits():
    """With a threshold of 0, every token is forced to O -- the safe extreme."""
    m = _tiny_detector().eval()
    ids = torch.randint(5, 199, (1, 8))
    mask = torch.ones_like(ids)
    pred = m.predict(ids, mask, torch.zeros_like(ids), copy_threshold=0.0)
    assert torch.all(pred == 0)


def test_detector_runs_without_script_features():
    cfg = DetectorConfig(num_tags=NUM_TAGS, use_script_features=False, use_copy_gate=False)
    from transformers import BertConfig, BertModel

    enc = BertModel(
        BertConfig(vocab_size=200, hidden_size=64, num_hidden_layers=2,
                   num_attention_heads=2, intermediate_size=128,
                   max_position_embeddings=64)
    )
    m = SpanDetector(cfg, encoder=enc)
    out = m(torch.randint(5, 199, (2, 10)), torch.ones(2, 10, dtype=torch.long))
    assert out["logits"].shape == (2, 10, NUM_TAGS)
    assert "copy_logits" not in out


def test_converter_forward_and_decode():
    cfg = ConverterConfig(d_model=64, nhead=2, num_encoder_layers=1,
                          num_decoder_layers=1, dim_feedforward=128)
    m = SpanConverter(cfg)
    src = torch.randint(4, 50, (3, 10))
    cat = torch.randint(0, cfg.num_categories, (3,))
    tgt_in = torch.randint(4, 50, (3, 7))
    tgt_out = torch.randint(4, 50, (3, 7))
    out = m(src, cat, tgt_in, tgt_out)
    assert out["logits"].shape == (3, 7, cfg.vocab_size)
    out["loss"].backward()

    gen = m.greedy_decode(src, cat, max_len=6)
    assert gen.shape[0] == 3 and gen.shape[1] <= 6


def test_char_roundtrip():
    """Encoding then decoding must preserve the string."""
    for text in ["intern", "AI", "ahmed@gmail.com", "Orange Innovation"]:
        ids = encode_chars(text, 48)
        assert decode_ids(ids) == text


def test_lexicon_roundtrip_and_oov():
    """OOV must be id 0 and unseen words must map to it, so the head can abstain."""
    lex = Lexicon.from_spans(["meeting", "meeting", "offer", "offer", "rare"], min_count=2)
    assert lex.itos[OOV_ID] == OOV
    assert lex.get("meeting") != OOV_ID
    assert lex.get("Meeting") == lex.get("meeting")   # case-insensitive fallback
    assert lex.get("rare") == OOV_ID                  # dropped by min_count
    assert lex.get("Kubernetes") == OOV_ID            # never seen


def test_lexicon_save_load(tmp_path=None):
    import tempfile, pathlib
    lex = Lexicon.from_spans(["a", "a", "b", "b"], min_count=2)
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "lex.json"
        lex.save(p)
        back = Lexicon.load(p)
    assert back.itos == lex.itos
    assert back.get("a") == lex.get("a")


def test_word_head_trains_and_predicts():
    """The word head must learn a closed-set mapping and expose confidences."""
    lex = Lexicon.from_spans(["intern", "intern", "offer", "offer"], min_count=2)
    cfg = ConverterConfig(d_model=64, nhead=2, num_encoder_layers=1,
                          num_decoder_layers=1, dim_feedforward=128,
                          use_word_head=True, lexicon_size=len(lex))
    m = SpanConverter(cfg)
    assert m.word_head is not None

    src = torch.randint(4, 50, (4, 8))
    cat = torch.zeros(4, dtype=torch.long)
    tgt_in = torch.randint(4, 50, (4, 6))
    tgt_out = torch.randint(4, 50, (4, 6))
    wid = torch.tensor([1, 2, 1, 2])

    out = m(src, cat, tgt_in, tgt_out, word_ids=wid)
    assert "word_logits" in out and "word_loss" in out and "char_loss" in out
    assert out["word_logits"].shape == (4, len(lex))
    out["loss"].backward()

    idx, conf = m.predict_words(src, cat)
    assert idx.shape == (4,) and conf.shape == (4,)
    assert bool(((conf >= 0) & (conf <= 1)).all())


def test_word_head_can_be_disabled():
    """With the head off the model must still train as a pure char converter."""
    cfg = ConverterConfig(d_model=64, nhead=2, num_encoder_layers=1,
                          num_decoder_layers=1, dim_feedforward=128,
                          use_word_head=False)
    m = SpanConverter(cfg)
    assert m.word_head is None
    out = m(torch.randint(4, 50, (2, 8)), torch.zeros(2, dtype=torch.long),
            torch.randint(4, 50, (2, 5)), torch.randint(4, 50, (2, 5)))
    assert "word_logits" not in out
    assert out["loss"].requires_grad


def test_converter_stays_small_enough():
    """Stage 2 must stay far below a pretrained byte model (~300M).

    It was raised from ~9M to ~53M after the first model plateaued at 33% exact
    match, but the low-latency design depends on it not growing without limit.
    """
    m = SpanConverter(ConverterConfig())
    assert m.num_parameters() < 80_000_000, m.num_parameters()


def test_context_window_marks_the_span():
    """Context must mark which tokens to convert, and survive encoding."""
    from arabic_dexlit.model.converter import SPAN_CLOSE, SPAN_OPEN
    from arabic_dexlit.training.dataset import build_context

    toks = ["a", "b", "c", "d", "e"]
    ctx = build_context(toks, 2, 3, window=1)
    assert ctx == f"b {SPAN_OPEN} c {SPAN_CLOSE} d"
    # window=0 reproduces the old isolated-span behaviour exactly
    assert build_context(toks, 2, 3, window=0) == "c"
    # markers must be real vocabulary, not <unk>
    from arabic_dexlit.model.converter import STOI
    assert SPAN_OPEN in STOI and SPAN_CLOSE in STOI


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
