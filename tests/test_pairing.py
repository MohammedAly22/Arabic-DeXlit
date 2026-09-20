"""Correctness tests for the data pipeline.

The invariants here are the ones that, if violated, would silently poison
training rather than raise an error -- so they are asserted explicitly.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_dexlit.data.pairing import build_example, passthrough_example  # noqa: E402
from arabic_dexlit.schema import OUTSIDE, TAG2ID, spans_from_tags  # noqa: E402


def test_tags_align_with_tokens():
    """One tag per input token, always. Misalignment would corrupt every label."""
    rng = random.Random(0)
    corpus = [
        "أنا رايح ال gym بعد ال work",
        "عندي meeting مهم مع ال team",
        "ابعت على ahmed@gmail.com",
        "بشتغل في AI و NLP",
    ]
    for sent in corpus:
        ex = build_example(sent, rng)
        assert ex is not None
        assert len(ex.src_tokens) == len(ex.tags)


def test_spans_recover_from_tags():
    """Decoded spans must match the spans recorded at generation time."""
    rng = random.Random(1)
    ex = build_example(
        "أنا عندي meeting مع ال manager النهارده",
        rng, copy_prob=0.0, fuse_al_prob=0.0,
    )
    decoded = spans_from_tags(ex.tags)
    assert len(decoded) == len(ex.spans)
    for (start, end, cat), rec in zip(decoded, ex.spans):
        assert start == rec["start"]
        assert end == rec["end"]
        assert cat == rec["category"]


def test_outside_tokens_are_copied_verbatim():
    """Every O-tagged token must survive byte-identically into the target."""
    rng = random.Random(2)
    ex = build_example(
        "أنا عندي meeting مهم جدا",
        rng,
    )
    kept = [t for t, g in zip(ex.src_tokens, ex.tags) if g == OUTSIDE]
    for tok in kept:
        assert tok in ex.tgt_tokens


def test_pure_arabic_is_identity():
    """The headline guarantee: pure Arabic in, byte-identical Arabic out."""
    sent = "أول شغل ليا بعد التخرج كنت مبسوط"
    ex = passthrough_example(sent)
    assert ex is not None
    assert ex.is_passthrough()
    assert ex.src == ex.tgt == sent
    assert set(ex.tags) == {OUTSIDE}


def test_sentence_with_no_latin_yields_none():
    """build_example is for conversion examples only; pure Arabic returns None."""
    rng = random.Random(3)
    assert build_example("أنا رايح البيت", rng) is None


def test_acronym_becomes_multi_token_span():
    """'AI' is spoken as two tokens, so it must form one 2-token ACRONYM span."""
    rng = random.Random(4)
    # copy_prob off: this test is about how an acronym is *spoken*, so the span
    # must actually be transliterated rather than copied through.
    ex = build_example("بشتغل في AI", rng, copy_prob=0.0)
    assert ex is not None
    acr = [s for s in ex.spans if s["category"] == "ACRONYM"]
    assert acr and acr[0]["target"] == "AI"
    assert acr[0]["end"] - acr[0]["start"] >= 2


def test_all_tags_are_in_schema():
    """Any tag the generator emits must be encodable by the model."""
    rng = random.Random(5)
    sents = [
        "عندي meeting مع Orange Innovation Egypt",
        "ابعت على a@b.com دلوقتي",
        "السنة 2024 كانت كويسة",
    ]
    for s in sents:
        ex = build_example(s, rng)
        if ex is None:
            continue
        for t in ex.tags:
            assert t in TAG2ID, t


def test_copy_case_leaves_latin_input_untouched():
    """Already-correct English must be learnable as a COPY, not a rewrite.

    With no such examples the model mangled Latin input it should have kept --
    "review" came out as "so", "deployment" as "He".
    """
    rng = random.Random(11)
    ex = build_example("هانعمل review للكود", rng, copy_prob=1.0)
    assert ex is not None
    assert "review" in ex.src          # input keeps the Latin form
    assert "review" in ex.tgt          # and so does the target
    assert any(s["target"] == "review" for s in ex.spans)


def test_fused_definite_article_is_generated():
    """السيرفر (fused) must occur, not only "ال سيرفر" (spaced).

    SDAIA always spaces the article, so the model converted سيرفر but failed on
    السيرفر -- the fused form real Egyptian ASR actually produces.
    """
    rng = random.Random(5)
    al = "ال"
    found = False
    for _ in range(60):
        ex = build_example(
            "عندي ال manager بكرة",
            rng, copy_prob=0.0, fuse_al_prob=1.0,
        )
        if ex is None:
            continue
        for s in ex.spans:
            src = " ".join(ex.src_tokens[s["start"]:s["end"]])
            if src.startswith(al) and len(src) > 3:
                found = True
                # the article must NOT leak into the English target
                assert s["target"] == "manager", s["target"]
    assert found, "fused article never generated"


def test_multiword_entities_are_grouped():
    """Multi-word targets were 0.14% of spans; they must be common enough."""
    rng = random.Random(6)
    multi = 0
    for _ in range(40):
        ex = build_example(
            "كنت في Orange Innovation Egypt بكرة",
            rng, copy_prob=0.0,
        )
        if ex and any(len(s["target"].split()) > 1 for s in ex.spans):
            multi += 1
    assert multi > 10, multi


def test_generation_is_deterministic_given_seed():
    """Same seed => same data, so dataset builds are reproducible."""
    a = build_example("عندي meeting بكره", random.Random(7))
    b = build_example("عندي meeting بكره", random.Random(7))
    assert a.src == b.src and a.tags == b.tags


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print("\nall green" if not fails else f"\n{fails} failing")
    sys.exit(1 if fails else 0)
