"""The security layer between the LLM and the Python application.

An LLM's output is untrusted input to the program. Three checkpoints:
  1. check_input      the user's message: length, prompt-injection phrases, requests for other people's data
  2. check_tool_call  a tool call the LLM proposes: tool on the allowlist, arguments parsed into the tool's
                      pydantic model (types, ranges, enums, no unknown fields) before anything runs
  3. check_answer     the final answer: must parse as the HRAnswer JSON schema, cite real handbook sections,
                      and carry no personal data (emails, phone numbers, ID-like numbers)
Every decision is returned as a Verdict, so the caller can log it and show a safe message.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.hr_tools import TOOLS
from src.data.load_data import sections

MAX_INPUT_CHARS = 2000
INJECTION = [r"ignore (all |your |the )?(previous |prior )?(instructions|rules)", r"system prompt",
             r"you are now", r"developer mode", r"jailbreak", r"act as (an? )?(admin|root|developer)",
             r"reveal (your|the) (prompt|instructions)", r"disregard .{0,20}(policy|rules)"]
OTHER_PEOPLES_DATA = [r"(everyone|all employees|other employees|colleagues?)'?s? (salar|pay|address|phone|medical)",
                      r"\b(home address|phone number|salary|medical record)s? of\b",
                      r"(?i:\bwhat is |\bshow me |\bgive me )[A-Z][a-z]+'s (?i:home address|salary|pay|phone|medical)"]
# a phone number has 10+ digits; an ISO date (2026-12-30) has 8 and must survive
PII = {"email": r"[\w.+-]+@[\w-]+\.[\w.]+", "phone": r"(?<![\d-])\+?\d(?:[\s-]?\d){9,13}(?![\d-])",
       "id_number": r"\b\d{9,12}\b"}


@dataclass
class Verdict:
    ok: bool
    stage: str
    reasons: list = field(default_factory=list)
    value: object = None


def check_input(text: str) -> Verdict:
    reasons = []
    if not text or not text.strip():
        reasons.append("empty message")
    if len(text) > MAX_INPUT_CHARS:
        reasons.append(f"message longer than {MAX_INPUT_CHARS} characters")
    if any(re.search(p, text, re.I) for p in INJECTION):
        reasons.append("looks like an attempt to override the assistant's instructions")
    if any(re.search(p, text, 0 if "[A-Z]" in p else re.I) for p in OTHER_PEOPLES_DATA):
        reasons.append("asks for another employee's personal data")
    return Verdict(not reasons, "input", reasons, text)


def check_tool_call(name: str, args: dict | str) -> Verdict:
    if name not in TOOLS:
        return Verdict(False, "tool_call", [f"tool '{name}' is not on the allowlist"])
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return Verdict(False, "tool_call", ["arguments are not valid JSON"])
    model = TOOLS[name][1]
    strict = type(model.__name__ + "Strict", (model,), {"model_config": ConfigDict(extra="forbid")})
    try:
        parsed = strict(**args)
    except ValidationError as e:
        return Verdict(False, "tool_call", [f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()])
    return Verdict(True, "tool_call", [], model(**parsed.model_dump()))


def run_tool(name: str, args: dict | str) -> dict:
    """Validate, then run. Never raises on bad LLM output: returns an error dict the LLM can read."""
    v = check_tool_call(name, args)
    if not v.ok:
        return {"tool": name, "error": "rejected by the security layer", "reasons": v.reasons}
    return TOOLS[name][0](v.value)


class HRAnswer(BaseModel):
    """The only shape an answer may take on its way to the user (JSON-structured output)."""
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1, max_length=1500)
    route: Literal["handbook", "tool", "out_of_scope", "blocked"]
    sources: list[str] = Field(default_factory=list, description="Handbook section ids, e.g. ['4.2']")
    tool_used: str | None = None


def redact(text: str) -> tuple[str, list[str]]:
    found = []
    for kind, p in PII.items():
        if re.search(p, text):
            found.append(kind)
            text = re.sub(p, f"[{kind} removed]", text)
    return text, found


def check_answer(raw: str | dict) -> Verdict:
    reasons = []
    try:
        a = HRAnswer(**(json.loads(raw) if isinstance(raw, str) else raw))
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        return Verdict(False, "answer", [f"not a valid HRAnswer: {str(e).splitlines()[0]}"])
    known = {s["id"] for s in sections()}
    bad = [s for s in a.sources if s not in known]
    if bad:
        reasons.append(f"cites sections that do not exist: {bad}")
    if a.route in ("handbook", "tool") and not a.sources:
        reasons.append("a policy answer must cite at least one handbook section")
    if a.tool_used and a.tool_used not in TOOLS:
        reasons.append(f"claims an unknown tool: {a.tool_used}")
    text, pii = redact(a.answer)
    if pii:
        a = a.model_copy(update={"answer": text})
    return Verdict(not reasons, "answer", reasons + [f"redacted {k}" for k in pii], a)


SAFE_REPLY = {
    "input": "I can only help with your own HR questions about the handbook, leave, pay, travel and resignation.",
    "answer": "I couldn't produce a reliable answer. Please contact the HR helpdesk.",
}
