"""Tests for the generality stack: phonetic keys, retrieval and reranking.

The claim being tested is that the system answers words it has *never seen in
Arabic form*, by matching sound rather than surface. A lookup table cannot do
this, which is the whole reason these components exist.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_dexlit.convert.phonetic import (  # noqa: E402
    PhoneticIndex,
    arabic_key,
    english_key,
)
from arabic_dexlit.convert.rerank import (  # noqa: E402
    CandidateReranker,
    ContextModel,
    phonetic_agreement,
)
from arabic_dexlit.convert.router import ConversionRouter  # noqa: E402


def test_spelling_variants_share_one_key():
    """The point of a phonetic key: ASR variation collapses to one string."""
    variants = ["\u0645\u064a\u062a\u064a\u0646\u062c", "\u0645\u064a\u062a\u0646\u062c", "\u0645\u064a\u0637\u0646\u062c"]
    keys = {arabic_key(v) for v in variants}
    assert len(keys) == 1, keys


def test_arabic_and_english_keys_meet():
    """Both scripts must land in the same key space or nothing matches."""
    assert arabic_key("\u0633\u062a\u0627\u0643\u0647\u0648\u0644\u062f\u0631") == english_key("stakeholder")
    assert arabic_key("\u0643\u0648\u0628\u0631\u0646\u064a\u062a\u0633") == english_key("kubernetes")


def test_phonetic_index_recalls_unseen_spellings():
    """A word is retrievable from an Arabic spelling the index never stored."""
    idx = PhoneticIndex(["kubernetes", "stakeholder", "meeting", "friends"])
    assert "kubernetes" in idx.lookup("\u0643\u0648\u0628\u064a\u0631\u0646\u064a\u062a\u064a\u0633")
    assert "stakeholder" in idx.lookup("\u0633\u062a\u0627\u0643\u064a\u0647\u0648\u0644\u062f\u064a\u0631")


def test_fuzzy_lookup_tolerates_one_substitution():
    """Exact keys are brittle; near keys must still be found."""
    idx = PhoneticIndex(["onboarding"])
    assert idx.lookup("\u0627\u0648\u0646\u0628\u0648\u0631\u062f\u0646\u062c") == ["onboarding"]


def test_phonetic_agreement_separates_homophones():
    span = "\u0641\u0631\u064a\u0646\u062f\u0633"           # فريندس
    assert phonetic_agreement(span, "friends") > phonetic_agreement(span, "French")


def test_context_decides_between_plausible_candidates():
    """Context is what picks 'meeting' over an equally-phonetic alternative."""
    # Two competing words with DIFFERENT contexts. With only one word in the
    # corpus every context term has marginal probability 1.0, the log-ratio
    # cancels, and both candidates score a meaningless 0.5.
    rows = [{
        "src_tokens": ["\u0639\u0646\u062f\u064a", "x", "\u0645\u0647\u0645", "\u0628\u0643\u0631\u0629"],
        "spans": [{"start": 1, "end": 2, "category": "CS", "target": "meeting"}],
    }] * 20 + [{
        "src_tokens": ["\u0644\u0628\u0633\u062a", "y", "\u0641\u064a", "\u0627\u0644\u0634\u062a\u0627"],
        "spans": [{"start": 1, "end": 2, "category": "CS", "target": "mitten"}],
    }] * 20
    ctx = ContextModel().fit(rows)
    rr = CandidateReranker(
        context_model=ctx, lexicon_counts=Counter({"meeting": 20, "mitten": 20})
    )
    ranked = rr.rank(
        "\u0645\u064a\u062a\u064a\u0646\u062c",
        [("meeting", "phonetic", 0.0), ("mitten", "phonetic", 0.0)],
        ["\u0639\u0646\u062f\u064a", "\u0645\u0647\u0645", "\u0628\u0643\u0631\u0629"],
    )
    assert ranked[0].text == "meeting"
    assert ranked[0].parts["context"] > ranked[1].parts["context"]


def test_rule_candidates_do_not_win_on_circular_evidence():
    """A rule output scores 1.0 phonetically by construction -- it must not count.

    The rule table and the phonetic key share a model, so agreement between them
    proves nothing. Left uncorrected, a known misspelling outranks the real word.
    """
    rr = CandidateReranker()
    ranked = rr.rank(
        "\u0641\u0631\u064a\u0646\u062f\u0633",
        [("frinds", "rule", 0.0), ("friends", "phonetic", 0.0)],
        [],
    )
    assert ranked[0].text == "friends"


def test_already_latin_spans_are_copied_not_converted():
    """34.7% of spans arrive in Latin already; converting them is pure damage."""
    r = ConversionRouter(phonetic_index=PhoneticIndex(["meeting", "meting"]))
    out = r.convert([(["meeting"], "CS", [])])[0]
    assert out.text == "meeting"
    assert out.converted is False and out.source == "copy"


def test_router_falls_back_to_copy_when_unconvinced():
    r = ConversionRouter(threshold=0.99)   # nothing can clear this
    out = r.convert([(["\u0645\u064a\u062a\u064a\u0646\u062c"], "CS", [])])[0]
    assert out.converted is False


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
