"""Tests for the protection layer, acronym inventory and end-to-end metrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_dexlit.inference.protect import (  # noqa: E402
    classify_token,
    protect,
    protected_mask,
)
from arabic_dexlit.model.acronyms import AcronymInventory, SEED_ACRONYMS  # noqa: E402
from arabic_dexlit.training.metrics import EndToEndMetrics  # noqa: E402


def test_protection_recognises_structured_tokens():
    """These have one correct output; a model must never get to decide."""
    cases = {
        "test@gmail.com": "email",
        "https://example.com/x": "url",
        "www.google.com": "url",
        "192.168.1.1": "ip",
        "3.14": "number",
        "2026": "number",
        "30%": "number",
        "5:30": "time",
        "12/05/2026": "date",
        "@user": "handle",
        "report.pdf": "file",
    }
    for token, reason in cases.items():
        assert classify_token(token) == reason, token


def test_protection_leaves_convertible_tokens_alone():
    """Arabic-script spans must still reach the detector."""
    for token in ["\u0645\u064a\u062a\u064a\u0646\u062c", "\u0627\u0646\u062a\u0631\u0646", "\u0628\u0643\u0631\u0629"]:
        assert classify_token(token) is None


def test_latin_protection_is_opt_in():
    """Already-Latin English is taught via the corpus by default, not pinned."""
    assert classify_token("meeting") is None
    assert classify_token("meeting", protect_latin=True) == "latin"


def test_protected_mask_marks_the_right_positions():
    toks = ["\u0627\u0628\u0639\u062a", "\u0639\u0644\u0649", "a@b.com", "\u0628\u0643\u0631\u0629", "2026"]
    mask = protected_mask(toks)
    assert mask == [False, False, True, False, True]
    assert [p.reason for p in protect(toks)] == ["email", "number"]


def test_acronym_inventory_is_closed_and_cased():
    """An acronym span can only ever produce a real, correctly-cased acronym."""
    inv = AcronymInventory(SEED_ACRONYMS)
    assert inv.get("AI") is not None
    assert inv.at(inv.get("AI")) == "AI"
    # case-insensitive lookup resolves to the canonical casing
    assert inv.at(inv.get("ai")) == "AI"
    assert inv.get("NOTANACRONYM") is None


def test_acronym_inventory_merges_corpus_and_seed():
    inv = AcronymInventory.from_corpus(["ZZZ", "ZZZ", "AI"], min_count=2)
    assert inv.get("ZZZ") is not None      # learned from the corpus
    assert inv.get("PM") is not None       # from the seed list
    assert inv.get("QQQ") is None          # below min_count


def test_unnecessary_modification_rate():
    """UMR counts tokens that were changed but should not have been."""
    m = EndToEndMetrics()
    # Three O tokens; the prediction corrupts one of them.
    m.update(
        pred_tokens=["a", "WRONG", "c"],
        gold_tokens=["a", "b", "c"],
        src_tokens=["a", "b", "c"],
        gold_tags=["O", "O", "O"],
    )
    res = m.compute()
    assert abs(res["unnecessary_modification_rate"] - 1 / 3) < 1e-9
    assert abs(res["protected_token_accuracy"] - 2 / 3) < 1e-9
    assert res["sentence_exact_match"] == 0.0


def test_umr_is_zero_when_nothing_is_touched():
    m = EndToEndMetrics()
    m.update(["a", "b"], ["a", "b"], ["a", "b"], ["O", "O"])
    res = m.compute()
    assert res["unnecessary_modification_rate"] == 0.0
    assert res["sentence_exact_match"] == 1.0


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
