"""CrewAI version: a hierarchical crew. An HR manager agent reads the question and delegates to specialists,
each holding only the tools for its job (least privilege):
    Policy specialist         search_handbook
    Payroll specialist        payroll_cutoff, remote_work_allowance, travel_daily_allowance, earned_leave_accrual
    Compliance specialist     validate_leave_request, validate_reimbursement, probation_end_date
    Documentation specialist  notice_period_end, search_handbook (resignation and exit steps)
The task's output is the HRAnswer schema, then the security layer checks it as in the other versions."""
from __future__ import annotations

from src.agents import toolkit as tk
from src.agents.toolkit import SYSTEM_PROMPT
from src.security.guard import SAFE_REPLY, HRAnswer, check_answer, check_input

SPECIALISTS = {
    "HR Policy Specialist": ("Answer policy questions only from the handbook, citing section numbers.",
                             [tk.search_handbook]),
    "Payroll Specialist": ("Work out pay timing and allowances with the payroll tools.",
                           [tk.payroll_cutoff, tk.remote_work_allowance, tk.travel_daily_allowance,
                            tk.earned_leave_accrual]),
    "HR Compliance Specialist": ("Check leave requests, expense claims and probation dates against the rules.",
                                 [tk.validate_leave_request, tk.validate_reimbursement, tk.probation_end_date]),
    "HR Documentation Specialist": ("Explain resignation, notice and exit steps.",
                                    [tk.notice_period_end, tk.search_handbook]),
}


def build(llm=None):
    from crewai import LLM, Agent, Crew, Process, Task
    from crewai.tools import tool

    from src.core.settings import settings
    llm = llm or LLM(model=f"openai/{settings().llm_model}", temperature=0)
    agents = [Agent(role=role, goal=goal, backstory=SYSTEM_PROMPT, llm=llm, allow_delegation=False,
                    tools=[tool(f.__name__)(f) for f in fns], verbose=False)
              for role, (goal, fns) in SPECIALISTS.items()]
    manager = Agent(role="HR Manager", goal="Route each employee question to the right specialist and return one "
                    "accurate, cited answer.", backstory=SYSTEM_PROMPT, llm=llm, allow_delegation=True)
    task = Task(description="Employee question: {question}\nAnswer it for this employee. Use the specialists; "
                            "never calculate yourself. If it is not an HR question, set route to out_of_scope.",
                expected_output="An HRAnswer: answer, route, sources (handbook section ids), tool_used.",
                output_pydantic=HRAnswer)
    return Crew(agents=agents, tasks=[task], process=Process.hierarchical, manager_agent=manager, verbose=False)


def ask(crew, question: str) -> dict:
    v = check_input(question)
    if not v.ok:
        return dict(HRAnswer(answer=SAFE_REPLY["input"], route="blocked").model_dump(), guard=v.reasons)
    out = crew.kickoff(inputs={"question": question})
    v = check_answer(out.pydantic.model_dump() if getattr(out, "pydantic", None) else {})
    if not v.ok:
        return dict(HRAnswer(answer=SAFE_REPLY["answer"], route="blocked").model_dump(), guard=v.reasons)
    return dict(v.value.model_dump(), guard=v.reasons)
