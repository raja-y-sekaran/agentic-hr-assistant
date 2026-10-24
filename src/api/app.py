"""FastAPI service.

    uvicorn src.api.app:app --reload      # then open http://127.0.0.1:8000/docs

/tools/{name} and /search work offline. /ask runs an agent and needs OPENAI_API_KEY."""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agents.toolkit import search_handbook
from src.core.hr_tools import TOOLS
from src.core.settings import settings
from src.security.guard import check_input, run_tool

app = FastAPI(title="Agentic HR assistant", version="1.0",
              description="Fictional company. HR tools, handbook search and agents behind one security layer.")
_AGENTS = {}


class Question(BaseModel):
    question: str = Field(examples=["I joined on 2026-03-15. When does my probation end?"])
    framework: Literal["langchain", "langgraph", "crewai", "agno"] = "langgraph"
    thread_id: str = "api"


@app.get("/health")
def health():
    return {"status": "ok", "llm_configured": bool(settings().openai_api_key)}


@app.get("/tools")
def tools():
    return {n: dict(description=d, input_schema=m.model_json_schema()) for n, (_, m, d) in TOOLS.items()}


@app.post("/tools/{name}")
def call_tool(name: str, args: dict):
    out = run_tool(name, args)
    if "error" in out:
        raise HTTPException(422, out)
    return out


@app.get("/search")
def search(q: str):
    v = check_input(q)
    if not v.ok:
        raise HTTPException(400, v.reasons)
    return {"results": search_handbook(q)}


@app.post("/ask")
def ask(body: Question):
    if not settings().openai_api_key:
        raise HTTPException(503, "OPENAI_API_KEY is not set; /tools and /search work without it")
    from src.models.live_eval import _ask_fn
    if body.framework not in _AGENTS:
        _AGENTS[body.framework] = _ask_fn(body.framework)
    return _AGENTS[body.framework](body.question, body.thread_id)
