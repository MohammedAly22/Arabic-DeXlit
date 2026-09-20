"""Tests for the hybrid converter: parsers, rules, gazetteer and routing.

The organising claim is that structured categories are *parsed* (exact, and not
a model's opinion) while ambiguous ones are neural with copy-by-default. These
tests pin both halves, and especially the decline paths -- a parser that guesses
when it should abstain is worse than no parser.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_dexlit.convert.gazetteer import Gazetteer  # noqa: E402
from arabic_dexlit.convert.parsers import (  # noqa: E402
    parse_email,
    parse_number,
    parse_span,
    parse_time,
    parse_url,
)
from arabic_dexlit.convert.router import ConversionRouter  # noqa: E402
from arabic_dexlit.convert.translit_rules import arabic_to_latin  # noqa: E402
from arabic_dexlit.model.acronyms import AcronymInventory, SEED_ACRONYMS  # noqa: E402


def test_number_parser_is_exact_or_declines():
    assert parse_number("\u062a\u0648 \u0632\u064a\u0631\u0648 \u062a\u0648 \u0641\u0648\u0631".split()) == "2024"
    assert parse_number("\u0648\u0627\u0646 \u0646\u0627\u064a\u0646 \u0646\u0627\u064a\u0646 \u0641\u0627\u064a\u0641".split()) == "1995"
    # A token that is not a digit word must abort the whole parse, not be
    # silently dropped -- a half-understood number is worse than none.
    assert parse_number(["\u062a\u0648", "\u0643\u0644\u0627\u0645"]) is None
    assert parse_number([]) is None


def test_email_parser_builds_real_addresses():
    got = parse_email("\u0627\u062d\u0645\u062f \u0627\u062a \u062c\u064a\u0645\u064a\u0644 \u062f\u0648\u062a \u0643\u0648\u0645".split())
    assert got is not None and "@" in got and got.endswith(".com")
    # Already-valid input is returned untouched.
    assert parse_email(["info@globaltech.com"]) == "info@globaltech.com"
    # Ordinary Arabic is not an address; the parser must decline.
    assert parse_email("\u0643\u0644\u0627\u0645 \u0639\u0631\u0628\u064a".split()) is None


def test_url_parser_handles_spoken_addresses():
    got = parse_url("\u062f\u0628\u0644\u064a\u0648 \u062f\u0628\u0644\u064a\u0648 \u062f\u0628\u0644\u064a\u0648 \u062f\u0648\u062a \u062c\u0648\u062c\u0644 \u062f\u0648\u062a \u0643\u0648\u0645".split())
    assert got == "www.google.com"


def test_email_category_falls_back_to_url():
    """The corpus labels spoken web addresses EMAIL; requiring @ rejected them."""
    toks = "\u062f\u0628\u0644\u064a\u0648 \u062f\u0628\u0644\u064a\u0648 \u062f\u0628\u0644\u064a\u0648 \u062f\u0648\u062a \u062c\u0648\u062c\u0644 \u062f\u0648\u062a \u0643\u0648\u0645".split()
    assert parse_email(toks) is None          # no @, so email declines
    assert parse_span(toks, "EMAIL") == "www.google.com"   # router recovers it


def test_time_parser_rejects_impossible_clock_times():
    assert parse_time("\u0641\u0627\u064a\u0641 \u062b\u0631\u064a \u0632\u064a\u0631\u0648".split()) == "5:30"
    # 99:99 is not a time.
    assert parse_time("\u0646\u0627\u064a\u0646 \u0646\u0627\u064a\u0646 \u0646\u0627\u064a\u0646 \u0646\u0627\u064a\u0646".split()) is None


def test_rule_transliteration_is_a_skeleton_not_an_answer():
    """The rules give phonetics, not orthography -- that is why a model follows."""
    assert arabic_to_latin("\u0641\u0631\u064a\u0646\u062f\u0633") == "frinds"   # gold: friends
    assert arabic_to_latin("\u0645\u064a\u062a\u064a\u0646\u062c") == "mitinj"   # gold: meeting
    assert arabic_to_latin("") == ""


def test_gazetteer_resolves_entities():
    rows = [{
        "src_tokens": ["\u0645\u0643\u0631\u0648\u0633\u0648\u0641\u062a", "\u062a\u064a\u0645\u0633"],
        "spans": [{"start": 0, "end": 2, "category": "ENTITY", "target": "Microsoft Teams"}],
    }]
    g = Gazetteer.from_corpus(rows)
    assert g.get("\u0645\u0643\u0631\u0648\u0633\u0648\u0641\u062a \u062a\u064a\u0645\u0633") == "Microsoft Teams"
    assert g.get("\u062c\u0648\u062c\u0644") == "Google"      # from the seed list
    assert g.get("\u0643\u0644\u0627\u0645 \u0645\u0634 \u0645\u0639\u0631\u0648\u0641") is None


def test_router_prefers_deterministic_paths():
    acr = AcronymInventory(SEED_ACRONYMS)
    g = Gazetteer({"\u0627\u0648\u0631\u0627\u0646\u062c": "Orange"})
    r = ConversionRouter(acronyms=acr, gazetteer=g.mapping)

    out = r.convert([
        ("\u062a\u0648 \u0632\u064a\u0631\u0648 \u062a\u0648 \u0641\u0648\u0631".split(), "NUMBER", []),
        ("\u0627\u064a\u0647 \u0627\u064a".split(), "ACRONYM", []),
        (["\u0627\u0648\u0631\u0627\u0646\u062c"], "ENTITY", []),
    ])
    assert out[0].text == "2024" and out[0].source == "parser"
    assert out[1].text == "AI" and out[1].source == "acronym"
    assert out[2].text == "Orange" and out[2].source == "gazetteer"


def test_router_copies_when_nothing_is_confident():
    """Copy-by-default: a missed conversion beats a corrupted token."""
    r = ConversionRouter(neural_fn=None)
    span = ["\u0645\u064a\u062a\u064a\u0646\u062c"]
    out = r.convert([(span, "CS", [])])[0]
    assert out.text == "\u0645\u064a\u062a\u064a\u0646\u062c"
    assert out.source == "copy" and out.converted is False


def test_router_respects_the_confidence_threshold():
    """A low-confidence neural answer must not replace the original."""
    def weak(batch, cats, ctxs):
        return [("meeting", 0.20)] * len(batch)

    def strong(batch, cats, ctxs):
        return [("meeting", 0.95)] * len(batch)

    span = ["\u0645\u064a\u062a\u064a\u0646\u062c"]
    low = ConversionRouter(neural_fn=weak, threshold=0.6).convert([(span, "CS", [])])[0]
    assert low.converted is False and low.text == span[0]

    high = ConversionRouter(neural_fn=strong, threshold=0.6).convert([(span, "CS", [])])[0]
    assert high.converted is True and high.text == "meeting"


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
