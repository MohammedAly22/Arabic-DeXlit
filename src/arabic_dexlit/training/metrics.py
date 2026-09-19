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
