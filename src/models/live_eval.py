"""Live evaluation of one framework's agent over eval/questions.jsonl (needs OPENAI_API_KEY; costs API tokens).

    python main.py --live langgraph        # or langchain, crewai, agno

Scores, per question: the route taken, the facts the answer must contain, and whether it cites the gold section.
Writes reports/live_eval_<framework>.json. Not run in CI: the offline checks are what CI guarantees."""
from __future__ import annotations

import json
import time

from src.models.evaluate import load_questions
from src.utils.paths import REPORTS

ROUTE = {"rag": "handbook", "tool": "tool", "out_of_scope": "out_of_scope", "blocked": "blocked"}


def _ask_fn(framework: str):
    if framework == "langchain":
        from src.agents import langchain_agent as m
        a = m.build()
        return lambda q, i: m.ask(a, q, thread_id=f"eval-{i}")
    if framework == "langgraph":
        from src.agents import langgraph_agent as m
        g = m.build()
        return lambda q, i: m.ask(g, q, thread_id=f"eval-{i}")
    if framework == "crewai":
        from src.agents import crewai_crew as m
        c = m.build()
        return lambda q, i: m.ask(c, q)
    if framework == "agno":
        from src.agents import agno_team as m
        t = m.build()
        return lambda q, i: m.ask(t, q, session_id=f"eval-{i}")
    raise ValueError(framework)


def run(framework: str) -> dict:
    ask, q, rows = _ask_fn(framework), load_questions(), []
    for i, r in enumerate(q.itertuples()):
        t0 = time.time()
        try:
            a = ask(r.question, i)
            err = None
        except Exception as e:  # keep going; record the failure
            a, err = {}, repr(e)[:200]
        text = a.get("answer", "")
        rows.append(dict(question=r.question, expected_route=ROUTE[r.route], route=a.get("route"),
                         route_ok=a.get("route") == ROUTE[r.route],
                         facts_ok=all(m.lower() in text.lower() for m in r.must_contain),
                         cites_gold=(r.section in a.get("sources", [])) if r.section else None,
                         seconds=round(time.time() - t0, 1), error=err, answer=text))
    n = len(rows)
    summary = dict(framework=framework, questions=n,
                   route_accuracy=round(sum(x["route_ok"] for x in rows) / n, 3),
                   facts_correct=round(sum(x["facts_ok"] for x in rows) / n, 3),
                   cites_gold_section=round(sum(bool(x["cites_gold"]) for x in rows if x["cites_gold"] is not None)
                                            / max(1, sum(x["cites_gold"] is not None for x in rows)), 3),
                   errors=sum(x["error"] is not None for x in rows), rows=rows)
    (REPORTS / f"live_eval_{framework}.json").write_text(json.dumps(summary, indent=1))
    return summary
