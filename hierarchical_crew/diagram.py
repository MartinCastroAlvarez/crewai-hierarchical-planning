"""Mermaid diagrams generated from the real objects, not drawn by hand.

* :func:`flow_mermaid` reads ``LaunchFlow.flow_definition()``, CrewAI's static
  definition of the flow (starts, listeners, routers and their ``emit`` labels).
* :func:`crew_mermaid` reads a built ``Crew``: its process, manager, agents, their
  tools, the tasks and each task's ``context``.
"""

from __future__ import annotations

from typing import Any

from crewai import Crew
from crewai import Process
from crewai.flow.flow import Flow

from hierarchical_crew.roles import BUILTIN_MANAGER
from hierarchical_crew.roles import PLANNER

__all__ = ["crew_mermaid", "flow_mermaid"]


def _conditions(condition: Any) -> list[str]:
    if condition is None or condition is True or condition is False:
        return []
    if isinstance(condition, str):
        return [condition]
    if isinstance(condition, dict):
        names: list[str] = []
        for key in ("or", "and"):
            for item in condition.get(key, []) or []:
                names.extend(_conditions(item))
        return names
    return []


def flow_mermaid(flow_cls: type[Flow[Any]]) -> str:
    """A Mermaid flowchart of a Flow class: methods, router labels and listeners."""
    definition = flow_cls.flow_definition()
    methods = definition.methods
    lines = ["flowchart TD"]
    emits: dict[str, str] = {}
    for name, method in methods.items():
        shape = f"{name}{{{{{name}}}}}" if method.router else f"{name}[{name}]"
        if method.human_feedback is not None:
            shape = f"{name}[/{name}: human feedback/]"
        lines.append(f"    {shape}")
        for label in method.emit or []:
            emits[str(label)] = name
    for name, method in methods.items():
        if method.start is True:
            lines.append(f"    START((start)) --> {name}")
        for trigger in _conditions(method.listen):
            if trigger in methods:
                lines.append(f"    {trigger} --> {name}")
            elif trigger in emits:
                lines.append(f"    {emits[trigger]} -- {trigger} --> {name}")
    return "\n".join(lines)


def _node(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def crew_mermaid(crew: Crew) -> str:
    """A Mermaid chart of a crew: who delegates to whom, tools, and task context."""
    lines = ["flowchart LR"]
    hierarchical = crew.process == Process.hierarchical
    if hierarchical:
        manager = crew.manager_agent.role if crew.manager_agent else BUILTIN_MANAGER
        lines.append(f"    M[[{manager}]]")
        if crew.planning:
            lines.append(f"    P([{PLANNER}]) -. plan per task .-> M")
    for agent in crew.agents:
        node = _node(agent.role)
        lines.append(f"    {node}({agent.role})")
        if hierarchical:
            lines.append(f"    M -- delegate --> {node}")
        for tool in agent.tools or []:
            lines.append(f"    {node} --- {_node(tool.name)}[/{tool.name}/]")
    for index, task in enumerate(crew.tasks, start=1):
        label = task.description.split("\n", 1)[0].rstrip(".")
        lines.append(f'    T{index}["{index}. {label}"]')
        owner = _node(task.agent.role) if task.agent else "M"
        lines.append(f"    T{index} -.-> {owner}")
        upstreams = task.context if isinstance(task.context, list) else []
        for upstream in upstreams:
            lines.append(f"    T{crew.tasks.index(upstream) + 1} --> T{index}")
    return "\n".join(lines)
