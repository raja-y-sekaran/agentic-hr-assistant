"""Agno version: a team in route mode. The team leader reads the question and hands it to one member; each member
holds only its own tools, the same split as the CrewAI crew. Session history is kept in an in-memory db, the
team's output is the HRAnswer schema, and the security layer checks it as in every other version."""
from __future__ import annotations

from src.agents import toolkit as tk
from src.agents.crewai_crew import SPECIALISTS
from src.agents.toolkit import SYSTEM_PROMPT
from src.security.guard import SAFE_REPLY, HRAnswer, check_answer, check_input


def build(model=None):
    from agno.agent import Agent
    from agno.db.in_memory import InMemoryDb
    from agno.models.openai import OpenAIChat
    from agno.team import Team, TeamMode

    from src.core.settings import settings
    model = model or OpenAIChat(id=settings().llm_model, temperature=0)
    members = [Agent(name=role, role=goal, model=model, tools=fns, instructions=SYSTEM_PROMPT)
               for role, (goal, fns) in SPECIALISTS.items()]
    return Team(name="HR Manager", members=members, model=model, mode=TeamMode.route,
                instructions=SYSTEM_PROMPT + "\nSend each question to the one specialist whose tools fit it. "
                "If it is not an HR question, answer with route out_of_scope yourself.",
                output_schema=HRAnswer, db=InMemoryDb(), add_history_to_context=True)


def ask(team, question: str, session_id: str = "demo") -> dict:
    v = check_input(question)
    if not v.ok:
        return dict(HRAnswer(answer=SAFE_REPLY["input"], route="blocked").model_dump(), guard=v.reasons)
    out = team.run(question, session_id=session_id)
    content = out.content if hasattr(out, "content") else out
    v = check_answer(content.model_dump() if isinstance(content, HRAnswer) else content or {})
    if not v.ok:
        return dict(HRAnswer(answer=SAFE_REPLY["answer"], route="blocked").model_dump(), guard=v.reasons)
    return dict(v.value.model_dump(), guard=v.reasons)
