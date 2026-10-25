"""``python -m hierarchical_crew "<request>"``: run the flow (or only the crew) once."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

from hierarchical_crew.config import MANAGERS
from hierarchical_crew.config import MODES
from hierarchical_crew.config import PROVIDERS
from hierarchical_crew.config import Settings
from hierarchical_crew.crew import MAX_REQUEST_CHARS
from hierarchical_crew.crew import build_crew
from hierarchical_crew.crew import run_crew
from hierarchical_crew.diagram import crew_mermaid
from hierarchical_crew.diagram import flow_mermaid
from hierarchical_crew.flow import FeedbackSource
from hierarchical_crew.flow import LaunchFlow
from hierarchical_crew.flow import ScriptedFeedback
from hierarchical_crew.flow import run_flow

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m hierarchical_crew",
        description="Product launch crew: CrewAI hierarchical process with planning, "
        "wrapped in a Flow (router, human feedback, guardrail). Local by default.",
    )
    p.add_argument("request", nargs="?", help="what to launch, e.g. 'Plan the launch of Aurora'")
    p.add_argument("--provider", choices=PROVIDERS, help="ollama (default), anthropic or fake")
    p.add_argument("--model", help="model id for the provider")
    p.add_argument("--mode", choices=MODES, help="crew process (default: planning)")
    p.add_argument("--manager", choices=MANAGERS, help="custom manager_agent or manager_llm")
    p.add_argument("--max-delegations", type=int, help="delegation cap per crew run")
    p.add_argument("--latency", type=float, help="seconds per call for the fake provider")
    p.add_argument("--crew-only", action="store_true", help="skip the flow, run the crew")
    p.add_argument(
        "--feedback",
        action="append",
        help="scripted human feedback (repeatable); default: ask on the console when "
        "interactive, otherwise approve",
    )
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    p.add_argument("--mermaid", choices=("flow", "crew"), help="print a diagram and exit")
    p.add_argument("--verbose", action="store_true", help="CrewAI's own console output")
    return p


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    changes: dict[str, Any] = {}
    if args.provider:
        changes["provider"] = args.provider
        if not args.model:
            changes["model"] = Settings.from_env({"LAUNCH_CREW_PROVIDER": args.provider}).model
    for name in ("model", "mode", "manager", "max_delegations"):
        value = getattr(args, name)
        if value is not None:
            changes[name] = value
    if args.latency is not None:
        changes["fake_latency"] = args.latency
    if args.verbose:
        changes["verbose"] = True
    return settings.with_changes(**changes) if changes else settings


def main(argv: list[str] | None = None, out: Callable[[str], Any] = print) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        settings = _settings(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.mermaid == "flow":
        out(flow_mermaid(LaunchFlow))
        return 0
    if args.mermaid == "crew":
        out(crew_mermaid(build_crew(settings)))
        return 0
    if not args.request:
        parser.error("a request is required")
    if len(args.request) > MAX_REQUEST_CHARS:
        parser.error(f"the request is longer than {MAX_REQUEST_CHARS} characters")
    try:
        if args.crew_only:
            result = run_crew(args.request, settings)
            payload: dict[str, Any] = {
                "mode": settings.mode,
                "status": "done",
                "brief": result.brief.model_dump(),
                "stats": result.stats.summary(),
            }
        else:
            feedback: FeedbackSource | None
            if args.feedback:
                feedback = ScriptedFeedback(args.feedback)
            elif sys.stdin.isatty():
                feedback = None  # CrewAI's ConsoleProvider asks you
            else:
                feedback = ScriptedFeedback(["approve"])
            flow = run_flow(args.request, settings, feedback=feedback)
            state = flow.state
            payload = {
                "mode": settings.mode,
                "route": state.route,
                "complexity": state.complexity,
                "decision": state.decision,
                "status": state.status,
                "errors": state.errors,
                "brief": state.final.model_dump() if state.final else None,
                "stats": state.stats,
            }
    except ValueError as exc:
        out(f"error: {exc}")
        return 2
    if args.json:
        out(json.dumps(payload, indent=2))
    else:
        _print_human(payload, out)
    return 0 if payload["status"] in ("done", "approved", "revised") else 1


def _print_human(payload: dict[str, Any], out: Callable[[str], Any]) -> None:
    stats = payload["stats"]
    head = f"[{payload['mode']}]"
    if "route" in payload:
        head += f" route={payload['route']} (complexity {payload['complexity']})"
        head += f" decision={payload['decision'] or '-'}"
    out(f"{head} status={payload['status']}")
    brief = payload["brief"]
    if brief:
        out(f"  {brief['product']} for {brief['target_segment']} at ${brief['price_usd']:.2f}")
        out(f"  tagline: {brief['tagline']}")
        for message in brief["key_messages"]:
            out(f"  - {message}")
        for risk in brief["risks"]:
            out(f"  risk: {risk}")
        for note in brief["qa_notes"]:
            out(f"  qa: {note}")
    for error in payload.get("errors") or []:
        out(f"  error: {error}")
    out(
        f"  tasks={stats.get('tasks_completed')} llm_calls={stats.get('llm_calls')} "
        f"delegations={stats.get('delegations')} -> {stats.get('delegated_to')} "
        f"guardrail_failures={stats.get('guardrail_failures')} seconds={stats.get('seconds')}"
    )
