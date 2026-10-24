"""Offline evaluation (no API key): chunking strategies, HR tool correctness, and the security layer.
Live evaluation of a framework's agent (needs OPENAI_API_KEY) is in src/models/live_eval.py."""
from __future__ import annotations

import json

import pandas as pd

from src.core.hr_tools import TOOLS
from src.models.retriever import evaluate_strategies
from src.security.guard import check_input, run_tool
from src.utils.paths import ROOT

EVAL = ROOT / "eval"


def load_questions() -> pd.DataFrame:
    return pd.read_json(EVAL / "questions.jsonl", lines=True, dtype={"section": str})


def tool_correctness(q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in q[q.route == "tool"].itertuples():
        fn, model, _ = TOOLS[r.tool]
        out = run_tool(r.tool, r.args)
        rows.append(dict(question=r.question, tool=r.tool, result=json.dumps(out),
                         correct=all(m in json.dumps(out) for m in r.must_contain) and "error" not in out,
                         cites_gold_section=out.get("handbook_section") == r.section))
    return pd.DataFrame(rows)


def security(q: pd.DataFrame) -> dict:
    red = pd.read_json(EVAL / "redteam.jsonl", lines=True)
    attacks = red[red.kind == "input"]
    calls = red[red.kind == "tool_call"]
    blocked = [not check_input(t).ok for t in attacks.text]
    legit = q[q.route.isin(["rag", "tool", "out_of_scope"])].question
    false_blocks = [t for t in legit if not check_input(t).ok]
    rejected = ["error" in run_tool(r.tool, r.args) for r in calls.itertuples()]
    should_block = q[q.route == "blocked"].question
    return dict(attack_messages=len(attacks), attack_messages_blocked=int(sum(blocked)),
                malformed_tool_calls=len(calls), malformed_tool_calls_rejected=int(sum(rejected)),
                legitimate_questions=len(legit), legitimate_questions_blocked=len(false_blocks),
                false_blocks=false_blocks,
                question_set_blocked_items=int(sum(not check_input(t).ok for t in should_block)),
                question_set_blocked_total=len(should_block))


def run_offline():
    q = load_questions()
    chunk = evaluate_strategies(q[q.section.notna() & q.route.isin(["rag", "tool"])])
    tools = tool_correctness(q)
    sec = security(q)
    return q, chunk, tools, sec
