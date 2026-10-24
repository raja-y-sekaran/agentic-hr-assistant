"""Offline stages: load the handbook -> chunk 8 ways -> retrieval comparison -> HR tools on the question set ->
security layer on the red-team set -> metrics and figures. No API key needed."""
from __future__ import annotations

import json

from src.data.load_data import load_handbook, sections
from src.features.chunking import STRATEGIES
from src.models.evaluate import run_offline
from src.utils.paths import DATA_PROCESSED, REPORTS


def run():
    text = load_handbook()
    chunks = {name: fn(text) for name, fn in STRATEGIES.items()}
    (DATA_PROCESSED / "chunks.json").write_text(json.dumps(
        {k: [dict(start=c["start"], end=c["end"], text=c["text"][:160]) for c in v] for k, v in chunks.items()},
        indent=1))
    q, chunk_cmp, tools, sec = run_offline()
    chunk_cmp.to_csv(REPORTS / "chunking_comparison.csv", index=False)
    tools.to_csv(REPORTS / "tool_results.csv", index=False)
    metrics = dict(handbook=dict(sections=len(sections()), words=len(text.split())),
                   questions=dict(total=len(q), by_route=q.route.value_counts().to_dict()),
                   chunking=chunk_cmp.to_dict("records"),
                   tools=dict(questions=len(tools), correct=int(tools.correct.sum()),
                              cites_gold_section=int(tools.cites_gold_section.sum())),
                   security=sec)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics, chunk_cmp
