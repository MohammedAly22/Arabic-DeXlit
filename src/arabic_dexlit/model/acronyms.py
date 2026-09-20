"""Constrained decoding for acronym spans.

An acronym has a *closed* answer set. ``بي ام`` is ``PM`` -- it is not ``pm``,
``BM``, or ``P.M.``, and a free-form character decoder that can emit any of
those is solving a harder problem than the task actually poses.

So acronym spans are scored against a fixed inventory instead of generated.
Two consequences matter in production:

* the output is always a real acronym, correctly cased;
* scoring a few hundred candidates is a single matrix operation, with no decode
  loop at all.

The inventory is seeded with the acronyms people actually say aloud in Arabic
code-switching, then extended from the training corpus so it tracks the data
rather than this file.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

# Seed inventory: spoken letter-by-letter in real speech. Grouped only for
# readability -- the model sees one flat list.
SEED_ACRONYMS: list[str] = [
    # time and everyday
    "AM", "PM", "OK", "TV", "PC", "SMS", "DVD", "CD", "GPS", "ATM", "ID",
    "VIP", "DIY", "FAQ", "ASAP", "RSVP", "CV", "PhD", "MSc", "BSc", "MBA",
    # tech
    "AI", "ML", "NLP", "API", "CPU", "GPU", "RAM", "SSD", "HDD", "USB",
    "VPN", "SQL", "HTML", "CSS", "JS", "PDF", "URL", "HTTP", "HTTPS", "IP",
    "OS", "UI", "UX", "QA", "IT", "DB", "SDK", "IDE", "CLI", "SaaS", "AWS",
    "GCP", "CI", "CD", "ML", "LLM", "GPT", "OCR", "ASR", "TTS", "NER",
    "REST", "JSON", "XML", "CSV", "RAG", "MVP", "POC", "SLA", "SSO", "OTP",
    # business
    "HR", "CEO", "CTO", "CFO", "COO", "CMO", "KPI", "ROI", "B2B", "B2C",
    "PR", "R&D", "PM", "PO", "OKR", "QoQ", "YoY", "EOD", "EOW", "WFH",
    # medical / academic / other
    "ICU", "ER", "MRI", "CT", "BP", "IV", "DNA", "COVID", "WHO", "UN",
    "USA", "UK", "UAE", "EU", "GMT", "UTC", "FYI", "TBD", "TBA", "N/A",
]


class AcronymInventory:
    """A closed candidate set for acronym spans."""

    def __init__(self, acronyms: Iterable[str]) -> None:
        seen: list[str] = []
        for a in acronyms:
            a = (a or "").strip()
            if a and a not in seen:
                seen.append(a)
        self.items: list[str] = seen
        self.index: dict[str, int] = {a: i for i, a in enumerate(self.items)}
        # Case-insensitive lookup, so "Ai" resolves to the canonical "AI".
        self._ci: dict[str, int] = {}
        for a, i in self.index.items():
            self._ci.setdefault(a.lower(), i)

    def __len__(self) -> int:
        return len(self.items)

    def get(self, acronym: str) -> int | None:
        i = self.index.get(acronym)
        if i is not None:
            return i
        return self._ci.get(acronym.lower())

    def at(self, idx: int) -> str:
        return self.items[idx] if 0 <= idx < len(self.items) else ""

    @classmethod
    def from_corpus(
        cls,
        targets: Iterable[str],
        *,
        min_count: int = 2,
        max_size: int = 2000,
        include_seed: bool = True,
    ) -> "AcronymInventory":
        """Build from the ACRONYM targets seen in training, plus the seed list.

        The seed covers acronyms a corpus may under-represent; the corpus covers
        what this seed cannot anticipate. Both are needed.
        """
        counts = Counter(t for t in targets if t)
        learned = [a for a, c in counts.most_common(max_size) if c >= min_count]
        return cls((SEED_ACRONYMS + learned) if include_seed else learned)

    # --- persistence ---
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"acronyms": self.items}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "AcronymInventory":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["acronyms"])

    def coverage(self, targets: Iterable[str]) -> float:
        seen = list(targets)
        if not seen:
            return 0.0
        return sum(1 for t in seen if self.get(t) is not None) / len(seen)
