"""Diagnostic plots logged to Weights & Biases during training.

These target the two questions asked of this model -- *which words does it
change?* and *what is it looking at?* -- rather than generic loss curves, which
wandb already draws from the scalar logs.

All figures are built with matplotlib only, and Arabic text is reshaped for
display when ``arabic-reshaper``/``python-bidi`` are installed. Those are
optional: without them the plots still render, with Arabic labels shown
unshaped, so a missing font package never breaks a training run.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: Colab workers have no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..schema import CATEGORIES, ID2TAG, OUTSIDE  # noqa: E402

try:  # optional: correct Arabic shaping/RTL in figures
    import arabic_reshaper
    from bidi.algorithm import get_display

    def ar(text: str) -> str:
        return get_display(arabic_reshaper.reshape(text))

except Exception:  # pragma: no cover - cosmetic only

    def ar(text: str) -> str:
        return text


_EDIT_COLOR = {
    "CS": "#2563eb",
    "ACRONYM": "#d97706",
    "ENTITY": "#7c3aed",
    "EMAIL": "#dc2626",
    "NUMBER": "#059669",
}


def plot_edit_decisions(
    tokens: list[str],
    pred_tags: list[str],
    gold_tags: list[str] | None = None,
    *,
    title: str = "What the model changes",
):
    """Render a sentence with each token shaded by the edit the model chose.

    This is the plot that answers "which words will it touch?" at a glance:
    untouched tokens stay grey, edited ones take their category colour, and a
    red underline marks a disagreement with the gold tags.
    """
    n = len(tokens)
    fig, ax = plt.subplots(figsize=(min(18, max(6, n * 0.9)), 1.9))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=8)

    for i, tok in enumerate(tokens):
        tag = pred_tags[i] if i < len(pred_tags) else OUTSIDE
        cat = None if tag == OUTSIDE else tag.split("-", 1)[1]
        color = _EDIT_COLOR.get(cat, "#e5e7eb") if cat else "#f3f4f6"
        ax.add_patch(
            plt.Rectangle((i + 0.05, 0.32), 0.9, 0.42, facecolor=color,
                          edgecolor="none", alpha=0.85 if cat else 1.0)
        )
        ax.text(
            i + 0.5, 0.53, ar(tok), ha="center", va="center", fontsize=9,
            color="white" if cat else "#111827",
        )
        if gold_tags is not None and i < len(gold_tags) and gold_tags[i] != tag:
            ax.plot([i + 0.1, i + 0.9], [0.26, 0.26], color="#ef4444", lw=2.5)
            ax.text(i + 0.5, 0.12, ar(gold_tags[i]), ha="center", fontsize=7, color="#ef4444")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=_EDIT_COLOR[c], label=c) for c in CATEGORIES
    ]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="#f3f4f6", label="unchanged"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=6, fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def plot_attention(
    tokens: list[str],
    attn: np.ndarray,
    *,
    title: str = "Attention (last layer, head-averaged)",
):
    """Token-by-token attention heatmap, for inspecting what the encoder uses."""
    n = min(len(tokens), attn.shape[-1])
    a = np.asarray(attn)[:n, :n]
    fig, ax = plt.subplots(figsize=(max(5, n * 0.45), max(4, n * 0.4)))
    im = ax.imshow(a, cmap="viridis", aspect="auto")
    labels = [ar(t) for t in tokens[:n]]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_confusion(cm: np.ndarray, labels: list[str], *, title: str = "Tag confusion"):
    """Row-normalised confusion over tags: shows *which* tag is being confused."""
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(title, fontsize=11)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if norm[i, j] > 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if norm[i, j] > 0.5 else "#111827")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_category_scores(metrics: dict[str, float], *, title: str = "Per-category F1"):
    """Bar chart of per-category F1, to expose a category that is lagging."""
    cats = [c for c in CATEGORIES if f"f1_{c}" in metrics]
    if not cats:
        return None
    vals = [metrics[f"f1_{c}"] for c in cats]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(cats, vals, color=[_EDIT_COLOR[c] for c in cats])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_safety(history: list[dict], *, title: str = "Safety: pass-through vs false edits"):
    """Track the two metrics that decide whether the model is safe to deploy."""
    if not history:
        return None
    steps = [h.get("step", i) for i, h in enumerate(history)]
    pa = [h.get("passthrough_accuracy", 0) for h in history]
    fer = [h.get("false_edit_rate", 0) for h in history]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(steps, pa, "-o", ms=3, color="#059669", label="pass-through accuracy")
    ax.plot(steps, fer, "-o", ms=3, color="#dc2626", label="false-edit rate")
    ax.axhline(1.0, ls=":", lw=1, color="#9ca3af")
    ax.set_xlabel("step")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def confusion_from_pairs(
    pred_ids: list[list[int]], gold_ids: list[list[int]], num_tags: int
) -> np.ndarray:
    """Build a tag confusion matrix, skipping ignored positions."""
    from ..schema import IGNORE_INDEX

    cm = np.zeros((num_tags, num_tags), dtype=np.int64)
    for p_row, g_row in zip(pred_ids, gold_ids):
        for p, g in zip(p_row, g_row):
            if g == IGNORE_INDEX:
                continue
            cm[int(g), int(p)] += 1
    return cm


def tag_labels(num_tags: int) -> list[str]:
    return [ID2TAG.get(i, str(i)) for i in range(num_tags)]
