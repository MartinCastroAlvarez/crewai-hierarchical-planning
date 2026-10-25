"""Per-run counters and the delegation cap, built on CrewAI's global execution hooks.

CrewAI's ``before_llm_call`` and ``before_tool_call`` hooks are process-wide. This
module registers one of each, once, and routes them to the :class:`RunStats` of the
run that is active in the current context (a :class:`contextvars.ContextVar`, which
CrewAI propagates into Flow method threads). Outside :func:`recording` the hooks do
nothing, so importing this package never changes other crews in the process.

The delegation cap lives here: a ``before_tool_call`` hook that returns ``False``
blocks the call, and CrewAI hands the manager the observation "Tool execution blocked
by hook". Together with ``max_iter`` on the manager, this bounds the hierarchy even if
the manager model keeps delegating.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from crewai.hooks import LLMCallHookContext
from crewai.hooks import ToolCallHookContext
from crewai.hooks import register_before_llm_call_hook
from crewai.hooks import register_before_tool_call_hook

__all__ = [
    "DELEGATION_TOOLS",
    "Delegation",
    "RunStats",
    "current_stats",
    "install_hooks",
    "recording",
]

DELEGATION_TOOLS = frozenset({"delegate_work_to_coworker", "ask_question_to_coworker"})


@dataclass(frozen=True, slots=True)
class Delegation:
    """One delegation (or question) from the manager to a coworker."""

    tool: str
    coworker: str
    task: str
    allowed: bool


@dataclass
class RunStats:
    """What one crew or flow run did. Filled in by the hooks and the guardrail."""

    max_delegations: int = 8
    llm_calls: Counter[str] = field(default_factory=Counter)
    """Model calls by agent role (``Task Execution Planner`` is CrewAI's planner)."""
    tool_calls: Counter[str] = field(default_factory=Counter)
    delegations: list[Delegation] = field(default_factory=list)
    guardrail_failures: int = 0
    guardrail_passes: int = 0
    guardrail_errors: list[str] = field(default_factory=list)
    tasks_completed: int = 0
    started: float = field(default_factory=time.perf_counter)
    seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def total_llm_calls(self) -> int:
        return sum(self.llm_calls.values())

    @property
    def allowed_delegations(self) -> list[Delegation]:
        return [d for d in self.delegations if d.allowed]

    @property
    def blocked_delegations(self) -> int:
        return sum(1 for d in self.delegations if not d.allowed)

    def delegated_to(self) -> list[str]:
        """Coworkers that actually received work, in order."""
        return [d.coworker for d in self.allowed_delegations]

    def stop(self) -> None:
        self.seconds = time.perf_counter() - self.started

    def summary(self) -> dict[str, Any]:
        return {
            "tasks_completed": self.tasks_completed,
            "llm_calls": self.total_llm_calls,
            "llm_calls_by_role": dict(sorted(self.llm_calls.items())),
            "tool_calls": dict(sorted(self.tool_calls.items())),
            "delegations": len(self.allowed_delegations),
            "delegated_to": self.delegated_to(),
            "blocked_delegations": self.blocked_delegations,
            "guardrail_failures": self.guardrail_failures,
            "seconds": round(self.seconds, 3),
        }


_ACTIVE: ContextVar[RunStats | None] = ContextVar("hierarchical_crew_stats", default=None)
_INSTALLED = False
_INSTALL_LOCK = threading.Lock()


def current_stats() -> RunStats | None:
    """The :class:`RunStats` of the run in this context, if any."""
    return _ACTIVE.get()


def _role(agent: Any) -> str:
    role = getattr(agent, "role", None)
    return str(role) if role else "(direct call)"


def _on_llm_call(context: LLMCallHookContext) -> None:
    stats = _ACTIVE.get()
    if stats is not None:
        with stats._lock:
            stats.llm_calls[_role(context.agent)] += 1


def _on_tool_call(context: ToolCallHookContext) -> bool | None:
    stats = _ACTIVE.get()
    if stats is None:
        return None
    name = context.tool_name
    with stats._lock:
        if name not in DELEGATION_TOOLS:
            stats.tool_calls[name] += 1
            return None
        allowed = len(stats.allowed_delegations) < stats.max_delegations
        args = context.tool_input if isinstance(context.tool_input, dict) else {}
        stats.delegations.append(
            Delegation(
                tool=name,
                coworker=str(args.get("coworker", ""))[:80],
                task=str(args.get("task") or args.get("question") or "")[:200],
                allowed=allowed,
            )
        )
        if allowed:
            stats.tool_calls[name] += 1
            return None
        return False


def install_hooks() -> None:
    """Register the two global hooks (idempotent)."""
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        register_before_llm_call_hook(_on_llm_call)
        register_before_tool_call_hook(_on_tool_call)
        _INSTALLED = True


@contextmanager
def recording(stats: RunStats | None = None, *, max_delegations: int = 8) -> Iterator[RunStats]:
    """Make ``stats`` the active run for this context; reuse an outer one if nested."""
    install_hooks()
    outer = _ACTIVE.get()
    if stats is None and outer is not None:
        yield outer
        return
    active = stats if stats is not None else RunStats(max_delegations=max_delegations)
    token = _ACTIVE.set(active)
    try:
        yield active
    finally:
        _ACTIVE.reset(token)
        active.stop()
