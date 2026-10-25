"""Agent roles and task kinds, shared by the crew, the flow and the fake model."""

from __future__ import annotations

__all__ = [
    "BUILTIN_MANAGER",
    "COPYWRITER",
    "EDITOR",
    "GENERALIST",
    "MANAGER",
    "PLANNER",
    "PRICING",
    "QA",
    "RESEARCHER",
    "SPECIALISTS",
    "TASK_ROUTES",
    "task_kind",
]

MANAGER = "Launch Manager"
BUILTIN_MANAGER = "Crew Manager"
"""The role CrewAI gives the manager it builds from ``manager_llm``."""
PLANNER = "Task Execution Planner"
"""The role of CrewAI's own planning agent (``planning=True``)."""
RESEARCHER = "Market Researcher"
PRICING = "Pricing Analyst"
COPYWRITER = "Copywriter"
QA = "QA Reviewer"
GENERALIST = "Launch Generalist"
EDITOR = "Brief Editor"
SPECIALISTS = (RESEARCHER, PRICING, COPYWRITER, QA)

TASK_ROUTES: dict[str, str] = {
    "research": RESEARCHER,
    "pricing": PRICING,
    "copy": COPYWRITER,
    "review": QA,
}
"""Which specialist owns which kind of task: the delegation the manager should make."""

_TASK_PREFIXES: tuple[tuple[str, str], ...] = (
    ("research the market", "research"),
    ("price ", "pricing"),
    ("write the launch copy", "copy"),
    ("review the draft", "review"),
)


def task_kind(description: str) -> str | None:
    """The kind of a crew task, from the start of its description."""
    head = description.strip().lower()
    for prefix, kind in _TASK_PREFIXES:
        if head.startswith(prefix):
            return kind
    return None
