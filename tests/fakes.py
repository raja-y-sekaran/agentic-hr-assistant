"""A scripted chat model: returns pre-written messages and structured outputs in order, so the agent graphs can be
tested end to end with no API key and no network."""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import PrivateAttr


class ScriptedChatModel(BaseChatModel):
    messages: list = []
    structured: dict = {}
    _calls: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kw: Any) -> ChatResult:
        self._calls.append(messages)
        msg = self.messages.pop(0) if self.messages else AIMessage("")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kw):
        return self

    def with_structured_output(self, schema, **kw):
        def run(_):
            return self.structured[schema.__name__].pop(0)
        return RunnableLambda(run)


def tool_call(name, args, i="1"):
    return AIMessage("", tool_calls=[dict(name=name, args=args, id=f"call_{i}", type="tool_call")])
