"""LangGraph version: an explicit graph instead of one agent loop.

    guard_input -> route -> handbook:     retrieve -> answer
                         -> tool:         agent <-> tools (max 4 rounds) -> answer
                         -> out_of_scope: answer
    answer -> guard_answer -> END

The router is the LLM with a structured Route output; the tools node runs every call through the security layer;
the answer node writes the HRAnswer JSON. State carries the messages, so a checkpointer gives conversation memory.
LangGraph Studio can load this file through langgraph.json; LangSmith traces each node when tracing is on."""
from __future__ import annotations

import json
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from src.agents.toolkit import CALC_TOOLS, SYSTEM_PROMPT, search_handbook
from src.security.guard import SAFE_REPLY, HRAnswer, check_answer, check_input, run_tool

MAX_TOOL_ROUNDS = 4


class Route(BaseModel):
    route: Literal["handbook", "tool", "out_of_scope"]


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    question: str
    route: str
    context: str
    rounds: int
    answer: dict
    guard: list


def build(model=None, checkpointer=True):
    if model is None:
        from langchain_openai import ChatOpenAI
        from src.core.settings import settings
        model = ChatOpenAI(model=settings().llm_model, temperature=0)
    tools = [StructuredTool.from_function(f) for f in CALC_TOOLS]
    tool_llm = model.bind_tools(tools)
    router = model.with_structured_output(Route)
    writer = model.with_structured_output(HRAnswer)

    def guard_input(s: State):
        q = s["messages"][-1].content
        v = check_input(q)
        if not v.ok:
            return {"question": q, "route": "blocked", "guard": v.reasons,
                    "answer": HRAnswer(answer=SAFE_REPLY["input"], route="blocked").model_dump()}
        return {"question": q, "rounds": 0, "guard": [], "context": ""}

    def route(s: State):
        r = router.invoke([SystemMessage("Classify the HR question. handbook = policy text answers it; tool = it "
                                         "needs a calculation or a check with the user's numbers or dates; "
                                         "out_of_scope = not about this company's HR."),
                           HumanMessage(s["question"])])
        return {"route": r.route}

    def retrieve(s: State):
        return {"context": search_handbook(s["question"])}

    def agent(s: State):
        msg = tool_llm.invoke([SystemMessage(SYSTEM_PROMPT)] + s["messages"])
        return {"messages": [msg], "rounds": s.get("rounds", 0) + 1}

    def tools_node(s: State):
        out = []
        for c in s["messages"][-1].tool_calls:
            out.append(ToolMessage(json.dumps(run_tool(c["name"], c["args"])), tool_call_id=c["id"], name=c["name"]))
        return {"messages": out}

    def answer(s: State):
        if s.get("route") == "out_of_scope":
            return {"answer": HRAnswer(answer="That isn't an HR question I can help with. I answer questions about "
                                              "this company's HR policies.", route="out_of_scope").model_dump()}
        notes = s.get("context", "")
        tool_results = [m.content for m in s["messages"] if isinstance(m, ToolMessage)]
        a = writer.invoke([SystemMessage(SYSTEM_PROMPT + "\nWrite the final answer as HRAnswer. route is 'handbook' "
                                         "or 'tool'; sources are the handbook section ids you used."),
                           HumanMessage(f"Question: {s['question']}\n\nHandbook extracts:\n{notes}\n\n"
                                        f"Tool results:\n" + "\n".join(tool_results))])
        return {"answer": a.model_dump()}

    def guard_answer(s: State):
        v = check_answer(s["answer"])
        if not v.ok:
            return {"answer": HRAnswer(answer=SAFE_REPLY["answer"], route="blocked").model_dump(),
                    "guard": s.get("guard", []) + v.reasons}
        return {"answer": v.value.model_dump(), "guard": s.get("guard", []) + v.reasons}

    g = StateGraph(State)
    for name, fn in [("guard_input", guard_input), ("route", route), ("retrieve", retrieve), ("agent", agent),
                     ("tools", tools_node), ("answer", answer), ("guard_answer", guard_answer)]:
        g.add_node(name, fn)
    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", lambda s: "end" if s.get("route") == "blocked" else "route",
                            {"end": END, "route": "route"})
    g.add_conditional_edges("route", lambda s: s["route"],
                            {"handbook": "retrieve", "tool": "agent", "out_of_scope": "answer"})
    g.add_edge("retrieve", "answer")
    g.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls and
                            s["rounds"] <= MAX_TOOL_ROUNDS else "answer", {"tools": "tools", "answer": "answer"})
    g.add_edge("tools", "agent")
    g.add_edge("answer", "guard_answer")
    g.add_edge("guard_answer", END)
    return g.compile(checkpointer=InMemorySaver() if checkpointer else None)


def ask(graph, question: str, thread_id: str = "demo") -> dict:
    out = graph.invoke({"messages": [HumanMessage(question)]}, config={"configurable": {"thread_id": thread_id}})
    return dict(out["answer"], guard=out.get("guard", []))


graph = None  # LangGraph Studio entry point is make_graph()


def make_graph():
    return build(checkpointer=False)
