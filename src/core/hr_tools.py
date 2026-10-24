"""The HR calculations the assistant can run, as plain Python with typed, validated inputs.

Every number comes from data/raw/policy.json (the same numbers the handbook states), every result is a JSON-ready
dict, and every result cites the handbook section it applies. The agent frameworks only wrap these functions, so
the rules are tested once and behave the same in LangChain, LangGraph, CrewAI and Agno.
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from src.data.load_data import load_policy

P = load_policy


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    y, m = d.year + m // 12, m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


# ---------- input models (the security layer validates LLM tool calls against these) ----------
class LeaveRequest(BaseModel):
    leave_type: Literal["annual", "sick", "casual"]
    days: int = Field(ge=1, le=60)
    notice_days: int = Field(ge=0, le=365, description="Days between today and the first day of leave")
    on_probation: bool = False
    has_medical_certificate: bool = False


class Accrual(BaseModel):
    completed_months: int = Field(ge=0, le=600)


class Probation(BaseModel):
    joining_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    extension_months: int = Field(0, ge=0, le=12)


class PayrollCheck(BaseModel):
    change_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class Reimbursement(BaseModel):
    days_since_return: int = Field(ge=0, le=3650)
    amount: float = Field(gt=0, le=1_000_000)
    travel_pre_approved: bool


class RemoteAllowance(BaseModel):
    work_model: Literal["office", "hybrid", "remote"]


class TravelAllowance(BaseModel):
    location: Literal["domestic", "international"]
    grade: int = Field(ge=1, le=4)
    days: int = Field(1, ge=1, le=365)


class Notice(BaseModel):
    grade: int = Field(ge=1, le=4)
    letter_received: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


# ---------- the tools ----------
def validate_leave_request(a: LeaveRequest) -> dict:
    lv, problems = P()["leave"], []
    if a.leave_type == "annual":
        sec = "4.2"
        if a.on_probation and not lv["probation_annual_leave_allowed"]:
            problems.append("annual leave cannot be taken during probation")
        if a.notice_days < lv["annual_min_notice_days"]:
            problems.append(f"needs at least {lv['annual_min_notice_days']} days' notice (given {a.notice_days})")
        if a.days > lv["annual_max_consecutive_days"]:
            problems.append(f"more than {lv['annual_max_consecutive_days']} consecutive days needs HR approval")
    elif a.leave_type == "sick":
        sec = "5"
        if a.days > lv["sick_days_per_year"]:
            problems.append(f"exceeds the {lv['sick_days_per_year']}-day yearly sick leave")
        if a.days > lv["sick_certificate_after_days"] and not a.has_medical_certificate:
            problems.append(f"a medical certificate is needed for more than {lv['sick_certificate_after_days']} days")
    else:
        sec = "6"
        if a.days > lv["casual_max_consecutive_days"]:
            problems.append(f"casual leave is limited to {lv['casual_max_consecutive_days']} consecutive days")
    return dict(tool="validate_leave_request", valid=not problems, problems=problems, handbook_section=sec)


def earned_leave_accrual(a: Accrual) -> dict:
    lv = P()["leave"]
    days = round(a.completed_months * lv["accrual_days_per_month"], 2)
    return dict(tool="earned_leave_accrual", completed_months=a.completed_months, accrued_days=days,
                capped_at_year=min(days, lv["annual_days_per_year"]) if a.completed_months <= 12 else None,
                handbook_section="4.1")


def probation_end_date(a: Probation) -> dict:
    pr = P()["probation"]
    if a.extension_months > pr["extension_max_months"]:
        return dict(tool="probation_end_date", error=f"extension is limited to {pr['extension_max_months']} months",
                    handbook_section="3")
    end = _add_months(_d(a.joining_date), pr["months"] + a.extension_months) - timedelta(days=1)
    return dict(tool="probation_end_date", joining_date=a.joining_date, probation_ends=end.isoformat(),
                months=pr["months"] + a.extension_months, handbook_section="3")


def payroll_cutoff(a: PayrollCheck) -> dict:
    d, cut = _d(a.change_date), P()["payroll"]["cutoff_day"]
    month = d if d.day <= cut else _add_months(d.replace(day=1), 1)
    return dict(tool="payroll_cutoff", change_date=a.change_date, cutoff_day=cut,
                paid_in=f"{calendar.month_name[month.month]} {month.year} salary",
                in_this_month=d.day <= cut, handbook_section="7")


def validate_reimbursement(a: Reimbursement) -> dict:
    r, problems, approvals = P()["reimbursement"], [], ["line manager"]
    if r["requires_approved_travel"] and not a.travel_pre_approved:
        problems.append("travel was not approved before it took place")
    if a.days_since_return > r["submit_within_days"]:
        problems.append(f"submitted after {r['submit_within_days']} days")
    if a.amount > r["max_claim_without_director"]:
        approvals.append("director")
    return dict(tool="validate_reimbursement", valid=not problems, problems=problems, approvals_needed=approvals,
                handbook_section="8.2")


def remote_work_allowance(a: RemoteAllowance) -> dict:
    return dict(tool="remote_work_allowance", work_model=a.work_model,
                monthly_allowance=P()["remote_allowance"][a.work_model], currency=P()["currency"],
                handbook_section="9")


def travel_daily_allowance(a: TravelAllowance) -> dict:
    rate = P()["travel_daily_allowance"][a.location][str(a.grade)]
    return dict(tool="travel_daily_allowance", location=a.location, grade=a.grade, daily_rate=rate, days=a.days,
                total=rate * a.days, currency=P()["currency"], handbook_section="8.1")


def notice_period_end(a: Notice) -> dict:
    n = P()["notice_period_days"][str(a.grade)]
    last = _d(a.letter_received) + timedelta(days=n)
    return dict(tool="notice_period_end", grade=a.grade, notice_days=n, last_working_day=last.isoformat(),
                handbook_section="10")


TOOLS = {
    "validate_leave_request": (validate_leave_request, LeaveRequest,
                               "Check a leave request (annual, sick or casual) against the leave rules."),
    "earned_leave_accrual": (earned_leave_accrual, Accrual, "Annual leave accrued after N completed months."),
    "probation_end_date": (probation_end_date, Probation, "Probation end date from the joining date."),
    "payroll_cutoff": (payroll_cutoff, PayrollCheck, "Which month's salary a pay change dated D will be paid in."),
    "validate_reimbursement": (validate_reimbursement, Reimbursement, "Check an expense claim and the approvals needed."),
    "remote_work_allowance": (remote_work_allowance, RemoteAllowance, "Monthly allowance for a work model."),
    "travel_daily_allowance": (travel_daily_allowance, TravelAllowance, "Daily and total travel allowance by grade."),
    "notice_period_end": (notice_period_end, Notice, "Notice period and last working day after a resignation."),
}
