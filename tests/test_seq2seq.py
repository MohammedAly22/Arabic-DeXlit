"""Tests for the end-to-end rewriter.

The byte-level claims are the ones worth pinning: if `C++` or a fused `الsystem`
does not survive a tokenizer round-trip, the model cannot possibly produce them,
and that failure would be invisible in a loss curve.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_dexlit.model.seq2seq import Seq2SeqConfig, encode_example  # noqa: E402
from arabic_dexlit.training.dataset import balanced_subset  # noqa: E402
from arabic_dexlit.training.metrics import EndToEndMetrics  # noqa: E402


def _tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("google/byt5-small")


def test_bytes_round_trip_on_the_hard_cases():
    """Nothing is out of vocabulary -- including the cases that broke the old pipeline."""
    tok = _tok()
    for text in [
        "C++",
        "\u0627\u0644system",                      # fused article + English
        "\u0627\u0644\u0628\u0631\u064a\u0632\u0646\u062a\u064a\u0634\u0646",   # البريزنتيشن
        "\u0644\u062a\u0648\u0628\u064a\u0643",    # لتوبيك -- prefix that broke phonetic keys
        "Orange Innovation Egypt",
        "ahmed@gmail.com",
    ]:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        assert tok.decode(ids) == text, text


def test_encode_example_adds_prefix_and_labels():
    tok = _tok()
    cfg = Seq2SeqConfig(max_source_length=64, max_target_length=64)
    enc = encode_example(tok, "\u0639\u0646\u062f\u064a \u0645\u064a\u062a\u064a\u0646\u062c", "\u0639\u0646\u062f\u064a Meeting", cfg)
    assert "input_ids" in enc and "labels" in enc
    # The task prefix must actually reach the encoder.
    assert cfg.task_prefix.strip() in tok.decode(enc["input_ids"])


def test_encode_example_without_target_has_no_labels():
    """Inference-time encoding must not fabricate labels."""
    tok = _tok()
    enc = encode_example(tok, "\u0639\u0646\u062f\u064a \u0645\u064a\u062a\u064a\u0646\u062c", None, Seq2SeqConfig())
    assert "labels" not in enc


def test_config_round_trips():
    cfg = Seq2SeqConfig(model_name="google/byt5-small", num_beams=2)
    back = Seq2SeqConfig.from_dict({**cfg.to_dict(), "unrelated_key": 1})
    assert back.model_name == "google/byt5-small"
    assert back.num_beams == 2


def test_balanced_subset_preserves_passthrough_ratio():
    """A capped run must train on the same task as the full one."""
    rows = [{"src": "a", "tgt": "b"}] * 70 + [{"src": "c", "tgt": "c"}] * 30
    out = balanced_subset(rows, 50, seed=0)
    assert len(out) == 50
    passes = sum(1 for r in out if r["src"] == r["tgt"])
    assert 10 <= passes <= 20, passes     # ~30% preserved


def test_umr_counts_only_tokens_that_should_not_change():
    m = EndToEndMetrics()
    m.update(
        pred_tokens=["a", "WRONG", "c"],
        gold_tokens=["a", "b", "c"],
        src_tokens=["a", "b", "c"],
        gold_tags=["O", "O", "O"],
    )
    res = m.compute()
    assert abs(res["unnecessary_modification_rate"] - 1 / 3) < 1e-9


def test_umr_is_zero_when_nothing_is_touched():
    m = EndToEndMetrics()
    m.update(["a", "b"], ["a", "b"], ["a", "b"], ["O", "O"])
    assert m.compute()["unnecessary_modification_rate"] == 0.0


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
