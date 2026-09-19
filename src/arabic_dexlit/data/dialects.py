"""Dialect registry and generation topics.

ArabicDeXlit is meant to work across Arabic broadly, not just MSA or Egyptian.
The SDAIA corpus covers MSA / Saudi / Egyptian only, so the remaining dialect
groups are filled in by synthesis (see :mod:`arabic_dexlit.data.synth`).

Each entry carries a *prompt hint* rather than a bare label: naming the
countries and a few signature particles gives the generator enough grounding to
produce recognisably dialectal text instead of MSA wearing a dialect tag.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dialect:
    code: str
    name: str
    hint: str
    # Whether SDAIA already covers it; used to decide synthesis budget.
    covered_by_corpus: bool = False


DIALECTS: list[Dialect] = [
    Dialect(
        "msa", "Modern Standard Arabic",
        "Formal written Modern Standard Arabic, as used in news and business writing.",
        covered_by_corpus=True,
    ),
    Dialect(
        "egy", "Egyptian",
        "Cairene Egyptian Arabic. Uses ده/دي, بقا, عاوز, دلوقتي, مش. "
        "Future with هـ, continuous with بـ.",
        covered_by_corpus=True,
    ),
    Dialect(
        "glf", "Gulf (Saudi/Emirati/Kuwaiti/Qatari)",
        "Gulf Arabic. Uses ويش, شلون, مال, حق, ابي, زين, مرة (= very). "
        "Speakers from Riyadh, Jeddah, Dubai, Kuwait.",
        covered_by_corpus=True,  # Saudi only; other Gulf states still need synthesis
    ),
    Dialect(
        "lev", "Levantine (Syrian/Lebanese/Jordanian/Palestinian)",
        "Levantine Arabic. Uses هيك, كتير, عم بـ for continuous, رح for future, "
        "شو, ليش, منيح. Speakers from Beirut, Damascus, Amman, Ramallah.",
    ),
    Dialect(
        "irq", "Iraqi",
        "Iraqi Arabic. Uses شنو, هسه, اكو/ماكو, زين, دـ for continuous, راح for future. "
        "Speakers from Baghdad, Basra, Mosul.",
    ),
    Dialect(
        "mgr", "Maghrebi (Moroccan/Algerian/Tunisian)",
        "Maghrebi Darija. Uses بزاف, ديال, واخا, غادي for future, شنو/شنوة, دابا. "
        "Note: Maghrebi speakers code-switch heavily with FRENCH as well as English.",
    ),
    Dialect(
        "sdn", "Sudanese",
        "Sudanese Arabic. Uses شنو, داير, زول, كداير. Speakers from Khartoum.",
    ),
    Dialect(
        "yem", "Yemeni",
        "Yemeni Arabic. Uses بش, ذي, كده, ماش. Speakers from Sanaa, Aden.",
    ),
]

DIALECT_BY_CODE: dict[str, Dialect] = {d.code: d for d in DIALECTS}

# Dialects needing the most synthetic data (no corpus coverage at all).
UNCOVERED = [d for d in DIALECTS if not d.covered_by_corpus]


# --- generation topics -----------------------------------------------------
# Code-switching is not uniform across subject matter: it clusters in domains
# where the technical vocabulary is English. Sampling topics explicitly keeps
# the corpus from collapsing onto a single register (all office small-talk).
TOPICS: list[str] = [
    "a software engineer talking about work, deadlines and deployments",
    "a university student discussing courses, exams and assignments",
    "a doctor or nurse discussing patients, shifts and medications",
    "someone describing a job interview and salary negotiation",
    "friends planning a trip, booking flights and hotels",
    "someone complaining about internet, phone or delivery services",
    "a startup founder pitching to investors",
    "someone talking about gym, fitness and diet",
    "a parent discussing their children's school and activities",
    "coworkers arranging a meeting and sharing files by email",
    "someone shopping online and discussing prices and payment",
    "a designer discussing branding, logos and social media",
    "someone discussing cars, driving and maintenance",
    "a teacher describing online classes and student grades",
    "someone talking about a wedding or family occasion",
    "a marketing employee discussing campaigns and analytics",
    "someone describing a problem with their laptop or phone",
    "an accountant discussing invoices, budgets and reports",
    "someone talking about football, matches and players",
    "a customer-support agent handling a complaint",
]

# Category emphases. These exist because the harvested corpus is overwhelmingly
# plain word-level code-switching -- a scale test on 20K SDAIA sentences found
# ~60K CS spans but only 178 acronyms, 9 entities and zero emails or numbers.
# Synthesis is steered to fill exactly those holes.
CATEGORY_PROMPTS: dict[str, str] = {
    "CS": (
        "Mix in ordinary English words and short phrases (nouns, verbs, "
        "adjectives) the way bilingual speakers naturally do."
    ),
    "ACRONYM": (
        "Each sentence MUST contain at least one English acronym said letter by "
        "letter, such as AI, HR, CEO, CV, API, IT, PhD, USB, GPS, ATM, KPI, UI, "
        "VPN, SQL, PDF, OTP, ID."
    ),
    "ENTITY": (
        "Each sentence MUST contain at least one multi-word English company, "
        "product, university or place name, such as Orange Innovation Egypt, "
        "Applied Innovation Center, Cairo University, Google Cloud, Microsoft "
        "Teams, Vodafone Business."
    ),
    "EMAIL": (
        "Each sentence MUST contain a realistic email address or website URL, "
        "such as ahmed.hassan@gmail.com, info@company.com or www.example.com."
    ),
    "NUMBER": (
        "Each sentence MUST contain numbers written as digits: years, prices, "
        "phone numbers, percentages or quantities, such as 2024, 1500, 30%."
    ),
}
