"""Run the project (course layout).

    python main.py                       # offline: chunking comparison, HR tools, security layer, figures
    python main.py --live langgraph      # live agent evaluation (langchain | langgraph | crewai | agno); needs OPENAI_API_KEY
    python main.py --ask "How many sick days do I get?" --framework langgraph
"""
import argparse
import json

from src import pipeline
from src.utils import plots


def run_offline(quiet=False):
    m, chunk_cmp = pipeline.run()
    plots.chunking(chunk_cmp)
    plots.security(m["security"])
    if not quiet:
        print(chunk_cmp.to_string(index=False))
        print(json.dumps({"tools": m["tools"], "security": {k: v for k, v in m["security"].items()
                                                              if k != "false_blocks"}}, indent=1))
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", choices=["langchain", "langgraph", "crewai", "agno"])
    ap.add_argument("--ask")
    ap.add_argument("--framework", default="langgraph", choices=["langchain", "langgraph", "crewai", "agno"])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.live:
        from src.models import live_eval
        s = live_eval.run(a.live)
        print(json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=1))
    elif a.ask:
        from src.models.live_eval import _ask_fn
        print(json.dumps(_ask_fn(a.framework)(a.ask, 0), indent=1))
    else:
        run_offline(a.quiet)
