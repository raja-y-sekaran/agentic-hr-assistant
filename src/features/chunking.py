"""Eight ways to cut the handbook into chunks for retrieval (the 'Top 8 chunking strategies').

Every chunk keeps its character span, so retrieval can be scored by whether a retrieved chunk overlaps the
handbook section that answers the question.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.load_data import sections


def _chunk(text, s, e, **kw):
    return dict(text=text[s:e].strip(), start=s, end=e, **kw)


def _sentences(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+|\n{2,}", text):
        if text[start:m.start()].strip():
            spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def fixed_size(text, size=400):
    return [_chunk(text, i, min(i + size, len(text))) for i in range(0, len(text), size)]


def sliding_window(text, size=400, overlap=100):
    step = size - overlap
    return [_chunk(text, i, min(i + size, len(text))) for i in range(0, max(len(text) - overlap, 1), step)]


def recursive(text, size=400, seps=("\n\n", "\n", ". ", " ")):
    """Split on the largest separator that keeps pieces under `size`, then merge small neighbours."""
    def split(s, e, level):
        if e - s <= size or level == len(seps):
            return [(s, e)]
        parts, cur = [], s
        for m in re.finditer(re.escape(seps[level]), text[s:e]):
            parts.append((cur, s + m.end()))
            cur = s + m.end()
        parts.append((cur, e))
        out = []
        for a, b in parts:
            out += split(a, b, level + 1) if b - a > size else [(a, b)]
        return out
    pieces, merged = split(0, len(text), 0), []
    for a, b in pieces:
        if merged and b - merged[-1][0] <= size:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [_chunk(text, a, b) for a, b in merged if text[a:b].strip()]


def sentence_groups(text, n=3):
    sp = _sentences(text)
    return [_chunk(text, sp[i][0], sp[min(i + n, len(sp)) - 1][1]) for i in range(0, len(sp), n)]


def paragraph(text):
    return [_chunk(text, m.start(), m.end()) for m in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text)]


def markdown_heading(text):
    """One chunk per numbered section, heading included (structure-aware)."""
    return [_chunk(text, s["start"], s["end"], section=s["id"]) for s in sections(text)]


def semantic(text, threshold=0.08):
    """Start a new chunk where adjacent sentences stop being similar (TF-IDF cosine below `threshold`)."""
    sp = _sentences(text)
    sents = [text[a:b] for a, b in sp]
    X = TfidfVectorizer(stop_words="english").fit_transform(sents)
    sims = [cosine_similarity(X[i], X[i + 1])[0, 0] for i in range(len(sents) - 1)]
    out, start = [], 0
    for i, s in enumerate(sims):
        if s < threshold or sents[i + 1].lstrip().startswith("#"):
            out.append(_chunk(text, sp[start][0], sp[i][1]))
            start = i + 1
    out.append(_chunk(text, sp[start][0], sp[-1][1]))
    return out


def parent_child(text, child=2):
    """Small child chunks (2 sentences) are searched; the whole parent section is what gets returned."""
    secs = sections(text)
    out = []
    for s in secs:
        body_sp = [(a, b) for a, b in _sentences(text[s["start"]:s["end"]])]
        for i in range(0, len(body_sp), child):
            a, b = body_sp[i][0], body_sp[min(i + child, len(body_sp)) - 1][1]
            out.append(dict(text=text[s["start"] + a:s["start"] + b].strip(), start=s["start"], end=s["end"],
                            child_span=(s["start"] + a, s["start"] + b), section=s["id"]))
    return out


STRATEGIES = {
    "fixed_size": fixed_size, "sliding_window": sliding_window, "recursive": recursive,
    "sentence_groups": sentence_groups, "paragraph": paragraph, "markdown_heading": markdown_heading,
    "semantic": semantic, "parent_child": parent_child,
}
