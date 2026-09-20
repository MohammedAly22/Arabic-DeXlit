"""Evaluation metrics.

Beyond ordinary tagging scores, two metrics here are specific to this project's
promises and should be read first on any eval report:

* **passthrough_accuracy** -- of the sentences that must be returned untouched,
  the fraction actually left untouched. This is the safety metric. A model with
  excellent F1 and poor pass-through accuracy is unusable, because it corrupts
  ordinary Arabic text.
* **false_edit_rate** -- the share of ``O`` tokens the model wanted to edit. It
  is the token-level view of the same risk and is the number to watch when
  tuning the copy-gate threshold.

Span scores are computed on *exact* boundary-and-category match: a span that
starts one token early is wrong, because it would feed the converter the wrong
characters.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..schema import ID2TAG, IGNORE_INDEX, OUTSIDE, OUTSIDE_ID, spans_from_tags


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@dataclass
class DetectorMetrics:
    """Accumulates over batches; call :meth:`compute` at the end of an epoch."""

    token_correct: int = 0
    token_total: int = 0
    o_total: int = 0
    o_wrongly_edited: int = 0
    span_tp: int = 0
    span_fp: int = 0
    span_fn: int = 0
    per_cat: Counter = field(default_factory=Counter)
    per_cat_gold: Counter = field(default_factory=Counter)
    per_cat_pred: Counter = field(default_factory=Counter)
    passthrough_total: int = 0
    passthrough_clean: int = 0

    def update(self, pred_ids: list[list[int]], gold_ids: list[list[int]]) -> None:
        for pred, gold in zip(pred_ids, gold_ids):
            p_tags, g_tags = [], []
            for pi, gi in zip(pred, gold):
                if gi == IGNORE_INDEX:
                    continue
                self.token_total += 1
                self.token_correct += int(pi == gi)
                if gi == OUTSIDE_ID:
                    self.o_total += 1
                    if pi != OUTSIDE_ID:
                        self.o_wrongly_edited += 1
                p_tags.append(ID2TAG.get(int(pi), OUTSIDE))
                g_tags.append(ID2TAG.get(int(gi), OUTSIDE))

            if not g_tags:
                continue

            # Sentence-level pass-through: an all-O gold sentence must stay all-O.
            if all(t == OUTSIDE for t in g_tags):
                self.passthrough_total += 1
                if all(t == OUTSIDE for t in p_tags):
                    self.passthrough_clean += 1

            gold_spans = set(spans_from_tags(g_tags))
            pred_spans = set(spans_from_tags(p_tags))
            self.span_tp += len(gold_spans & pred_spans)
            self.span_fp += len(pred_spans - gold_spans)
            self.span_fn += len(gold_spans - pred_spans)
            for s in gold_spans:
                self.per_cat_gold[s[2]] += 1
            for s in pred_spans:
                self.per_cat_pred[s[2]] += 1
            for s in gold_spans & pred_spans:
                self.per_cat[s[2]] += 1

    def compute(self) -> dict[str, float]:
        p, r, f = _prf(self.span_tp, self.span_fp, self.span_fn)
        out = {
            "token_accuracy": self.token_correct / max(1, self.token_total),
            "span_precision": p,
            "span_recall": r,
            "span_f1": f,
            "false_edit_rate": self.o_wrongly_edited / max(1, self.o_total),
            "passthrough_accuracy": (
                self.passthrough_clean / self.passthrough_total
                if self.passthrough_total
                else 1.0
            ),
        }
        for cat, tp in self.per_cat_gold.items():
            hit = self.per_cat.get(cat, 0)
            cp, cr, cf = _prf(hit, self.per_cat_pred.get(cat, 0) - hit, tp - hit)
            out[f"f1_{cat}"] = cf
            out[f"recall_{cat}"] = cr
        return out


@dataclass
class ConverterMetrics:
    """Exact-match is the metric that matters: a span is right or it is not."""

    exact: int = 0
    total: int = 0
    char_correct: int = 0
    char_total: int = 0
    per_cat_exact: Counter = field(default_factory=Counter)
    per_cat_total: Counter = field(default_factory=Counter)

    def update(self, preds: list[str], golds: list[str], cats: list[str] | None = None) -> None:
        for i, (p, g) in enumerate(zip(preds, golds)):
            self.total += 1
            ok = p.strip().lower() == g.strip().lower()
            self.exact += int(ok)
            self.char_total += max(len(p), len(g))
            self.char_correct += sum(a == b for a, b in zip(p, g))
            if cats:
                c = cats[i]
                self.per_cat_total[c] += 1
                self.per_cat_exact[c] += int(ok)

    def compute(self) -> dict[str, float]:
        out = {
            "exact_match": self.exact / max(1, self.total),
            "char_accuracy": self.char_correct / max(1, self.char_total),
        }
        for c, n in self.per_cat_total.items():
            out[f"exact_{c}"] = self.per_cat_exact[c] / max(1, n)
        return out


def sentence_exact_match(preds: list[str], golds: list[str]) -> float:
    """End-to-end score: the whole output string must match exactly."""
    if not golds:
        return 0.0
    return sum(p.strip() == g.strip() for p, g in zip(preds, golds)) / len(golds)


# --- end-to-end sentence metrics -------------------------------------------
# Sentence BLEU is close to useless here: "5 PM" vs "5 pm" differs in one
# character but BLEU treats the whole sentence as damaged, while a model that
# quietly corrupts an untouched Arabic word can still score well. The metrics
# below separate the two things that actually matter -- did it convert what it
# should, and did it leave alone what it should.


def _char_error_rate(pred: str, gold: str) -> float:
    """Levenshtein distance normalised by reference length."""
    if not gold:
        return 0.0 if not pred else 1.0
    prev = list(range(len(gold) + 1))
    for i, pc in enumerate(pred, 1):
        cur = [i]
        for j, gc in enumerate(gold, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (pc != gc)))
        prev = cur
    return prev[-1] / len(gold)


@dataclass
class EndToEndMetrics:
    """Sentence-level report for a full pipeline run."""

    sentences: int = 0
    sentence_exact: int = 0

    # Unnecessary Modification Rate: of the tokens that should have been left
    # alone, how many were changed. For a drop-in ASR post-processor this is the
    # single most important number -- it is the rate at which the model damages
    # text it was never asked to touch.
    protected_total: int = 0
    protected_changed: int = 0

    # Of the tokens that genuinely needed conversion, how many came out right.
    convert_total: int = 0
    convert_correct: int = 0

    per_cat_total: Counter = field(default_factory=Counter)
    per_cat_correct: Counter = field(default_factory=Counter)

    cer_sum: float = 0.0
    cer_n: int = 0

    def update(
        self,
        pred_tokens: list[str],
        gold_tokens: list[str],
        src_tokens: list[str],
        gold_tags: list[str],
        spans: list[dict] | None = None,
        pred_spans: list[dict] | None = None,
    ) -> None:
        self.sentences += 1
        self.sentence_exact += int(
            " ".join(pred_tokens).strip() == " ".join(gold_tokens).strip()
        )

        # Token-level protection: an O-tagged source token must survive intact.
        pred_set = Counter(pred_tokens)
        for tok, tag in zip(src_tokens, gold_tags):
            if tag != OUTSIDE:
                continue
            self.protected_total += 1
            if pred_set[tok] > 0:
                pred_set[tok] -= 1
            else:
                self.protected_changed += 1

        # Span conversion accuracy, overall and per category.
        for sp in spans or []:
            tgt = sp.get("target", "")
            cat = sp.get("category", "CS")
            self.convert_total += 1
            self.per_cat_total[cat] += 1
            hit = any(tgt == p.get("converted") for p in (pred_spans or []))
            self.convert_correct += int(hit)
            self.per_cat_correct[cat] += int(hit)
            if pred_spans:
                best = min(
                    (_char_error_rate(p.get("converted", ""), tgt) for p in pred_spans),
                    default=1.0,
                )
                self.cer_sum += best
                self.cer_n += 1

    def compute(self) -> dict[str, float]:
        out = {
            "sentence_exact_match": self.sentence_exact / max(1, self.sentences),
            # The headline safety number. Lower is better; 0.0 is the goal.
            "unnecessary_modification_rate": (
                self.protected_changed / max(1, self.protected_total)
            ),
            "protected_token_accuracy": 1.0
            - self.protected_changed / max(1, self.protected_total),
            "conversion_accuracy": self.convert_correct / max(1, self.convert_total),
            "span_cer": self.cer_sum / max(1, self.cer_n),
            "n_sentences": float(self.sentences),
        }
        for cat, n in self.per_cat_total.items():
            out[f"accuracy_{cat}"] = self.per_cat_correct[cat] / max(1, n)
        return out
