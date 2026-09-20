"""The English vocabulary Arabic speakers actually code-switch into.

Why this file exists
--------------------
The harvested corpus is GPT-4 generated from a narrow set of templates. Measured
on it, the most frequent "code-switched" targets are English *function* words --
``the``, ``to``, ``I``, ``a``, ``and`` -- and one sentence opening repeats 568
times. Meanwhile the words real speakers actually switch into were missing
outright: ``intern`` appeared **zero** times in 1.2M spans, as did ``Speech``,
``healthtech`` and ``stakeholder``; ``repo`` appeared once.

That is why a model scoring 0.947 on validation still failed on a real sentence:
it had mastered a template, not the language. No architecture fixes a word the
data never contains.

This list drives targeted generation (:mod:`arabic_dexlit.data.synth`) so the
corpus covers the vocabulary of actual bilingual speech: workplace, tech,
medical, academic and everyday borrowings, including the ones that are almost
never written in Arabic script but are constantly *spoken*.
"""
from __future__ import annotations

# Grouped by domain purely for maintenance; generation flattens them.
VOCAB_DOMAINS: dict[str, list[str]] = {
    "workplace": [
        "intern", "internship", "offer", "manager", "meeting", "deadline",
        "feedback", "review", "presentation", "report", "schedule", "budget",
        "client", "stakeholder", "onboarding", "promotion", "resign", "notice",
        "contract", "freelance", "remote", "remotely", "hybrid", "shift",
        "overtime", "salary", "raise", "bonus", "appraisal", "probation",
        "handover", "workshop", "training", "conference", "networking",
        "startup", "founder", "investor", "pitch", "funding", "equity",
        "roadmap", "milestone", "sprint", "standup", "retrospective", "backlog",
        "task", "ticket", "priority", "blocker", "follow-up", "escalate",
    ],
    "tech": [
        "software", "hardware", "developer", "engineer", "frontend", "backend",
        "fullstack", "database", "server", "cloud", "deployment", "deploy",
        "pipeline", "repository", "repo", "commit", "merge", "branch", "pull",
        "push", "staging", "production", "bug", "debug", "patch", "release",
        "version", "update", "upgrade", "framework", "library", "package",
        "dependency", "config", "environment", "container", "docker",
        "kubernetes", "microservice", "endpoint", "request", "response",
        "latency", "throughput", "cache", "queue", "load", "scaling",
        "dashboard", "analytics", "metrics", "logging", "monitoring", "alert",
        "machine learning", "deep learning", "model", "training", "dataset",
        "inference", "embedding", "transformer", "fine-tuning", "prompt",
        "token", "benchmark", "accuracy", "healthtech", "fintech", "edtech",
        "Speech", "speech", "vision", "robotics", "annotation", "labeling",
        "corpus", "pipeline", "checkpoint", "hyperparameter", "overfitting",
    ],
    "medical": [
        "doctor", "nurse", "patient", "clinic", "hospital", "emergency",
        "surgery", "operation", "diagnosis", "prescription", "medication",
        "dose", "treatment", "therapy", "checkup", "appointment", "referral",
        "scan", "lab", "results", "symptoms", "recovery", "insurance",
        "consultant", "specialist", "resident", "ward", "discharge",
    ],
    "academic": [
        "university", "faculty", "professor", "lecture", "seminar", "course",
        "semester", "credit", "assignment", "thesis", "research", "paper",
        "citation", "supervisor", "scholarship", "grade", "exam", "midterm",
        "final", "graduate", "undergraduate", "campus", "library", "degree",
        "abstract", "methodology", "conclusion", "peer review", "submission",
    ],
    "everyday": [
        "mall", "gym", "cafe", "restaurant", "delivery", "order", "booking",
        "reservation", "flight", "hotel", "ticket", "passport", "visa",
        "weekend", "holiday", "vacation", "traffic", "parking", "grocery",
        "shopping", "discount", "offer", "receipt", "refund", "warranty",
        "subscription", "account", "password", "profile", "notification",
        "message", "call", "video", "photo", "story", "post", "share",
        "follow", "like", "comment", "stream", "playlist", "episode",
    ],
    "entities": [
        "Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix", "Uber",
        "LinkedIn", "WhatsApp", "Instagram", "YouTube", "Zoom", "Slack",
        "Notion", "Figma", "GitHub", "Docker", "PyTorch", "TensorFlow",
        "Orange Innovation Egypt", "Applied Innovation Center",
        "Vodafone Business", "Microsoft Teams", "Google Cloud",
        "Amazon Web Services", "Cairo University", "German University in Cairo",
        "Valeo", "Siemens", "Schneider Electric", "Dell Technologies",
    ],
}

# Flat list used by the generator.
CODE_SWITCH_VOCAB: list[str] = [w for words in VOCAB_DOMAINS.values() for w in words]

# Acronyms spoken letter by letter. Kept here too so generation can target them
# directly rather than hoping they appear by chance.
SPOKEN_ACRONYMS: list[str] = [
    "AI", "ML", "NLP", "API", "CPU", "GPU", "RAM", "USB", "VPN", "SQL", "PDF",
    "HR", "CEO", "CTO", "CFO", "COO", "KPI", "ROI", "PM", "AM", "OK", "TV",
    "PhD", "MBA", "CV", "IT", "QA", "UI", "UX", "OS", "SDK", "IDE", "LLM",
    "OCR", "ASR", "TTS", "MVP", "POC", "SLA", "OTP", "ICU", "ER", "MRI",
    "DNA", "GPS", "ATM", "ID", "VIP", "FAQ", "ASAP", "EOD", "WFH", "B2B",
]


def domain_of(word: str) -> str | None:
    """Which domain a word belongs to, for balanced sampling."""
    for domain, words in VOCAB_DOMAINS.items():
        if word in words:
            return domain
    return None


def vocab_prompt_block(words: list[str], limit: int = 25) -> str:
    """Render a word list for a generation prompt."""
    return ", ".join(words[:limit])
