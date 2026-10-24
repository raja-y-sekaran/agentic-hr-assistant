"""Offline retriever (TF-IDF + cosine) used to compare chunking strategies without an API key.
The agent apps use OpenAI embeddings through LangChain / Agno instead; the chunking comparison is independent of it."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.load_data import load_handbook, sections
from src.features.chunking import STRATEGIES


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.X = self.vec.fit_transform([c["text"] for c in chunks])

    def search(self, query: str, k: int = 3) -> list[dict]:
        s = cosine_similarity(self.vec.transform([query]), self.X)[0]
        out, seen = [], set()
        for i in np.argsort(-s):
            c = self.chunks[i]
            key = (c["start"], c["end"])      # parent-child: several children map to one parent
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(c, score=float(s[i])))
            if len(out) == k:
                break
        return out


def section_span(sec_id: str) -> tuple[int, int]:
    s = next(x for x in sections() if x["id"] == sec_id)
    return s["start"], s["end"]


def hit(chunk: dict, gold: str) -> bool:
    a, b = section_span(gold)
    s, e = chunk["start"], chunk["end"]
    return min(b, e) - max(a, s) > 20


def evaluate_strategies(questions: pd.DataFrame, k=3) -> pd.DataFrame:
    text, rows = load_handbook(), []
    for name, fn in STRATEGIES.items():
        chunks = fn(text)
        r = Retriever(chunks)
        hits1 = hitsk = rr = 0
        for q in questions.itertuples():
            res = r.search(q.question, k)
            ranks = [i for i, c in enumerate(res) if hit(c, q.section)]
            hits1 += bool(ranks and ranks[0] == 0)
            hitsk += bool(ranks)
            rr += 1 / (ranks[0] + 1) if ranks else 0
        n = len(questions)
        sizes = [c["end"] - c["start"] for c in chunks]
        rows.append(dict(strategy=name, chunks=len(chunks), median_chars_returned=int(np.median(sizes)),
                         hit_at_1=round(hits1 / n, 3), hit_at_k=round(hitsk / n, 3), mrr=round(rr / n, 3)))
    return pd.DataFrame(rows).sort_values(["hit_at_k", "mrr"], ascending=False).reset_index(drop=True)
