"""Synthesise target sentences that carry the real code-switching vocabulary.

The harvested corpus is template-generated and its "English" is mostly function
words. Measured on it, ``intern`` occurs **zero** times in 1.2M spans, along with
``Speech``, ``healthtech`` and ``stakeholder``; ``repo`` occurs once. A model
cannot learn a word the data does not contain, whatever its architecture.

LLM generation fixes this best, but it needs quota and it needs to be online.
This module is the offline complement: it places the vocabulary from
:mod:`arabic_dexlit.data.vocab` into natural Arabic carrier sentences drawn from
several dialects, producing *target*-form output that the normal pipeline then
transliterates. It is deliberately templated -- the goal is lexical coverage,
not stylistic variety, and the stochastic transliterator supplies the surface
variation that matters for this task.

Use it alongside the harvested and generated corpora, never instead of them.
"""
from __future__ import annotations

import random
from typing import Iterator

from .vocab import CODE_SWITCH_VOCAB, SPOKEN_ACRONYMS, VOCAB_DOMAINS

# Carrier sentences with a single {} slot, per dialect. Written to sound like
# speech rather than prose, since that is what ASR transcribes.
CARRIERS: dict[str, list[str]] = {
    "egy": [
        "أنا شغال على ال {} دلوقتي",
        "عندي {} مهم النهارده",
        "هو قاللي على ال {} امبارح",
        "مش عارف اعمل ايه في ال {}",
        "لازم نخلص ال {} قبل بكرة",
        "ال {} ده كان كويس اوي",
        "بعتلي ال {} على الميل",
        "احنا بنستنى ال {} من امبارح",
    ],
    "glf": [
        "أنا أشتغل على ال {} الحين",
        "عندي {} مهم اليوم",
        "ويش رايك في ال {} هذا",
        "لازم نخلص ال {} قبل بكرة",
        "ال {} مرة زين",
        "أرسلت لك ال {} على الإيميل",
    ],
    "lev": [
        "عم بشتغل عال {} هلق",
        "عندي {} مهم اليوم",
        "شو رأيك بال {} هاد",
        "لازم نخلص ال {} قبل بكرا",
        "ال {} كتير منيح",
        "بعتلك ال {} عالإيميل",
    ],
    "msa": [
        "أعمل الآن على ال {}",
        "لدي {} مهم اليوم",
        "يجب أن ننهي ال {} قبل الغد",
        "كان ال {} ممتازاً",
        "أرسلت لك ال {} عبر البريد",
    ],
    "irq": [
        "أني دأشتغل على ال {} هسه",
        "عندي {} مهم اليوم",
        "شنو رأيك بال {} هذا",
        "ال {} كلش زين",
    ],
    "mgr": [
        "كنخدم دابا على ال {}",
        "عندي {} مهم اليوم",
        "شنو رأيك ف ال {}",
        "ال {} بزاف مزيان",
    ],
}

# Two-slot carriers, so a sentence can hold more than one span.
DOUBLE_CARRIERS: dict[str, list[str]] = {
    "egy": [
        "ال {} بتاع ال {} خلص",
        "كلمت ال {} بخصوص ال {}",
        "عندي {} مع ال {} بكرة",
    ],
    "glf": [
        "ال {} حق ال {} خلص",
        "كلمت ال {} بخصوص ال {}",
    ],
    "lev": [
        "ال {} تبع ال {} خلص",
        "حكيت مع ال {} بخصوص ال {}",
    ],
    "msa": [
        "انتهى ال {} الخاص بال {}",
        "تحدثت مع ال {} بشأن ال {}",
    ],
}


def iter_vocab_sentences(
    n: int = 60000,
    *,
    seed: int = 0,
    acronym_share: float = 0.15,
    double_share: float = 0.25,
) -> Iterator[tuple[str, str, str]]:
    """Yield ``(target_sentence, dialect, source)`` covering the vocabulary.

    Sampling walks the vocabulary in shuffled passes rather than drawing
    independently, so every word is seen a similar number of times instead of
    coverage being left to chance.
    """
    rng = random.Random(seed)
    dialects = list(CARRIERS)

    pool = list(CODE_SWITCH_VOCAB)
    rng.shuffle(pool)
    cursor = 0

    made = 0
    while made < n:
        dialect = rng.choice(dialects)

        if rng.random() < acronym_share:
            word = rng.choice(SPOKEN_ACRONYMS)
        else:
            if cursor >= len(pool):
                rng.shuffle(pool)
                cursor = 0
            word = pool[cursor]
            cursor += 1

        doubles = DOUBLE_CARRIERS.get(dialect)
        if doubles and rng.random() < double_share:
            if cursor >= len(pool):
                rng.shuffle(pool)
                cursor = 0
            second = pool[cursor]
            cursor += 1
            sentence = rng.choice(doubles).format(word, second)
        else:
            sentence = rng.choice(CARRIERS[dialect]).format(word)

        yield sentence, dialect, "vocab"
        made += 1


def vocabulary_report(sentences: list[str]) -> dict[str, int]:
    """Count how often each vocabulary word appears -- a coverage check."""
    counts: dict[str, int] = {}
    joined = " ".join(sentences)
    for word in CODE_SWITCH_VOCAB + SPOKEN_ACRONYMS:
        counts[word] = joined.count(word)
    return counts
