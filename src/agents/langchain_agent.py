"""LangChain version: one agent built with create_agent. RAG is a tool (search_handbook), not a fixed first step,
so the model decides whether to retrieve or to calculate. Conversation memory comes from a checkpointer keyed by
thread_id; the reply is forced into the HRAnswer JSON schema and then checked by the security layer.
LangSmith tracing switches on with LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env (no code change)."""
from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from src.agents.toolkit import ALL_TOOLS, SYSTEM_PROMPT
from src.security.guard import SAFE_REPLY, HRAnswer, check_answer, check_input


def lc_tools():
    return [StructuredTool.from_function(f) for f in ALL_TOOLS]


def build(model=None):
    """model: a LangChain chat model; default is OpenAI (needs OPENAI_API_KEY)."""
    if model is None:
        from langchain_openai import ChatOpenAI
        from src.core.settings import settings
        model = ChatOpenAI(model=settings().llm_model, temperature=0)
    return create_agent(model, lc_tools(), system_prompt=SYSTEM_PROMPT, response_format=HRAnswer,
                        checkpointer=InMemorySaver(), name="hr_assistant_langchain")


def ask(agent, question: str, thread_id: str = "demo") -> dict:
    v = check_input(question)
    if not v.ok:
        return dict(HRAnswer(answer=SAFE_REPLY["input"], route="blocked").model_dump(), guard=v.reasons)
    out = agent.invoke({"messages": [{"role": "user", "content": question}]},
                       config={"configurable": {"thread_id": thread_id}})
    a = out.get("structured_response")
    v = check_answer(a.model_dump() if a is not None else {})
    if not v.ok:
        return dict(HRAnswer(answer=SAFE_REPLY["answer"], route="blocked").model_dump(), guard=v.reasons)
    return dict(v.value.model_dump(), guard=v.reasons)
