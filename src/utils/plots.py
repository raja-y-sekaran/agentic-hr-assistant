"""Figures: chunking strategies compared, and the security layer's checks."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.paths import FIGURES

INK, MUTED, GRID, BLUE, AQUA = "#1c1b19", "#52514e", "#e4e3df", "#2a5bd7", "#1baf7a"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)


def chunking(df):
    d = df.sort_values("mrr")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharey=True,
                                 gridspec_kw=dict(width_ratios=[3, 2]))
    y = np.arange(len(d))
    a1.barh(y + 0.2, d.hit_at_1, height=0.38, color=BLUE, label="right section ranked 1st")
    a1.barh(y - 0.2, d.hit_at_k, height=0.38, color=AQUA, label="right section in top 3")
    a1.set_yticks(y, d.strategy)
    a1.set_xlim(0, 1.05)
    a1.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2)
    a1.set_title("Retrieval on 36 handbook questions", loc="left", fontsize=11)
    a2.barh(y, d.median_chars_returned, color="#9a9893", height=0.55)
    a2.set_title("Median characters per chunk", loc="left", fontsize=11)
    for ax in (a1, a2):
        _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "chunking_strategies.png")
    plt.close(fig)


def security(s: dict):
    labels = ["Attack messages blocked", "Malformed tool calls rejected", "Legitimate questions let through"]
    done = [s["attack_messages_blocked"], s["malformed_tool_calls_rejected"],
            s["legitimate_questions"] - s["legitimate_questions_blocked"]]
    tot = [s["attack_messages"], s["malformed_tool_calls"], s["legitimate_questions"]]
    fig, ax = plt.subplots(figsize=(7, 2.4), dpi=150)
    y = np.arange(3)[::-1]
    ax.barh(y, tot, color=GRID, height=0.55)
    ax.barh(y, done, color=[BLUE, BLUE, AQUA], height=0.55)
    for yi, d_, t in zip(y, done, tot):
        ax.text(t + 0.5, yi, f"{d_} / {t}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(tot) * 1.15)
    _style(ax)
    ax.set_title("Security layer on the red-team and question sets", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "security_layer.png")
    plt.close(fig)
