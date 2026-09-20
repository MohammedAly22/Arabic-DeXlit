"""Entity gazetteer: Arabic surface form -> canonical English name.

Entity conversion is *resolution*, not transliteration. ``مكروسوفت تيمس`` is
``Microsoft Teams`` because that product exists and is spelled that way -- no
character mapping recovers the capital T, and a phonetic model asked to produce
it will sooner or later emit ``Microsoft Teems``.

Measured on the corpus, this is close to a deterministic table: 3,386 distinct
Arabic forms, of which only **7** map to more than one English target. So a
lookup answers the overwhelming majority exactly, and the neural converter is
only needed for names never seen before.

Ambiguous forms keep their most frequent reading, with the alternatives
retained so a ranker can reconsider them in context.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

# Seed entries for names that matter but may be rare in any given corpus.
SEED_ENTITIES: dict[str, str] = {
    "جوجل": "Google",
    "مايكروسوفت": "Microsoft",
    "مكروسوفت": "Microsoft",
    "امازون": "Amazon",
    "ابل": "Apple",
    "فيسبوك": "Facebook",
    "انستجرام": "Instagram",
    "يوتيوب": "YouTube",
    "واتساب": "WhatsApp",
    "لينكدان": "LinkedIn",
    "زوم": "Zoom",
    "سلاك": "Slack",
    "جيتهب": "GitHub",
    "دوكر": "Docker",
    "اورانج": "Orange",
    "فودافون": "Vodafone",
    "اتصالات": "Etisalat",
    "شات جي بي تي": "ChatGPT",
    "اوبر": "Uber",
    "نتفليكس": "Netflix",
}


class Gazetteer:
    """Arabic form -> canonical English, with alternatives kept for ranking."""

    def __init__(self, mapping: dict[str, str], alternatives: dict[str, list[str]] | None = None) -> None:
        self.mapping = {k.strip(): v for k, v in mapping.items() if k and v}
        self.alternatives = alternatives or {}
        # Normalised index so minor orthographic variation still resolves.
        self._norm = {self._key(k): v for k, v in self.mapping.items()}

    @staticmethod
    def _key(text: str) -> str:
        t = text.strip().lower()
        for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"),
                     ("ى", "ي"), ("ة", "ه")):
            t = t.replace(a, b)
        return " ".join(t.split())

    def __len__(self) -> int:
        return len(self.mapping)

    def get(self, arabic: str) -> str | None:
        hit = self.mapping.get(arabic.strip())
        if hit:
            return hit
        return self._norm.get(self._key(arabic))

    def candidates(self, arabic: str) -> list[str]:
        """Every reading recorded for a form, best first."""
        best = self.get(arabic)
        alts = self.alternatives.get(arabic.strip(), [])
        out = ([best] if best else []) + [a for a in alts if a != best]
        return out

    # --- construction -----------------------------------------------------
    @classmethod
    def from_corpus(
        cls,
        rows: Iterable[dict],
        *,
        categories: tuple[str, ...] = ("ENTITY",),
        min_count: int = 1,
        include_seed: bool = True,
    ) -> "Gazetteer":
        """Learn the table from labelled spans."""
        observed: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            toks = r["src_tokens"]
            for s in r.get("spans", []):
                if s["category"] not in categories:
                    continue
                form = " ".join(toks[s["start"] : s["end"]]).strip()
                if form:
                    observed[form][s["target"]] += 1

        mapping: dict[str, str] = dict(SEED_ENTITIES) if include_seed else {}
        alternatives: dict[str, list[str]] = {}
        for form, counts in observed.items():
            ranked = counts.most_common()
            if ranked[0][1] < min_count:
                continue
            mapping[form] = ranked[0][0]
            if len(ranked) > 1:
                alternatives[form] = [t for t, _ in ranked[1:]]
        return cls(mapping, alternatives)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"mapping": self.mapping, "alternatives": self.alternatives},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Gazetteer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["mapping"], d.get("alternatives", {}))

    def coverage(self, rows: Iterable[dict], categories: tuple[str, ...] = ("ENTITY",)) -> float:
        total = hit = 0
        for r in rows:
            toks = r["src_tokens"]
            for s in r.get("spans", []):
                if s["category"] not in categories:
                    continue
                total += 1
                form = " ".join(toks[s["start"] : s["end"]])
                hit += self.get(form) == s["target"]
        return hit / max(1, total)
