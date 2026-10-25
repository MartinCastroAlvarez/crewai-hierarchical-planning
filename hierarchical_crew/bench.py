"""Benchmark: sequential vs hierarchical vs hierarchical + planning, and the cheap path.

Real CrewAI runs - real processes, delegation tools, planner, guardrail, tools and
dataset - with only the model replaced by :class:`~hierarchical_crew.fake.LaunchFakeLLM`
sleeping ``--latency`` seconds per call. The numbers therefore measure orchestration
(how many model round trips each process needs), not any model's quality or speed.

    python -m hierarchical_crew.bench --latency 0.25 --repeats 3 --json docs/benchmark/results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from collections.abc import Callable
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any

from hierarchical_crew.config import Settings
from hierarchical_crew.crew import run_crew
from hierarchical_crew.dataset import MarketData
from hierarchical_crew.flow import ScriptedFeedback
from hierarchical_crew.flow import run_flow
from hierarchical_crew.guardrails import brief_errors
from hierarchical_crew.roles import PLANNER

__all__ = ["CASES", "main", "run_benchmark", "table"]

CRAFT = "Plan the launch of Aurora: segments, competitor pricing, price, tagline and risks."
SIMPLE = "Write a tagline for Aurora."
CASES: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("crew: sequential", CRAFT, {"mode": "sequential"}),
    ("crew: hierarchical", CRAFT, {"mode": "hierarchical"}),
    ("crew: hierarchical + planning", CRAFT, {"mode": "planning"}),
    ("crew: hierarchical + planning (manager_llm)", CRAFT, {"mode": "planning", "manager": "llm"}),
    ("flow: complex request (routed to crew)", CRAFT, {"mode": "planning", "flow": True}),
    ("flow: simple request (routed to one agent)", SIMPLE, {"mode": "planning", "flow": True}),
)


def _one(request: str, settings: Settings, market: MarketData, use_flow: bool) -> dict[str, Any]:
    if use_flow:
        flow = run_flow(request, settings, market=market, feedback=ScriptedFeedback(["approve"]))
        summary = flow.stats.summary()
        valid = flow.state.status == "approved"
        summary["route"] = flow.state.route
    else:
        result = run_crew(request, settings, market=market)
        summary = result.stats.summary()
        valid = not brief_errors(result.brief, market)
        summary["route"] = "-"
    summary["valid"] = valid
    summary["planner_calls"] = summary["llm_calls_by_role"].get(PLANNER, 0)
    return summary


def run_benchmark(latency: float, repeats: int) -> list[dict[str, Any]]:
    market = MarketData.load()
    rows: list[dict[str, Any]] = []
    for name, request, options in CASES:
        opts = dict(options)
        use_flow = bool(opts.pop("flow", False))
        settings = Settings(provider="fake", fake_latency=latency, **opts)
        runs = [_one(request, settings, market, use_flow) for _ in range(repeats)]
        first = runs[0]
        rows.append(
            {
                "case": name,
                "route": first["route"],
                "tasks_completed": first["tasks_completed"],
                "delegations": first["delegations"],
                "llm_calls": first["llm_calls"],
                "planner_calls": first["planner_calls"],
                "tool_calls": sum(v for k, v in first["tool_calls"].items() if "coworker" not in k),
                "guardrail_failures": first["guardrail_failures"],
                "seconds": round(statistics.median(r["seconds"] for r in runs), 3),
                "valid": all(r["valid"] for r in runs),
            }
        )
    return rows


def table(rows: list[dict[str, Any]]) -> str:
    head = (
        "| Case | Tasks | Delegations | LLM calls | of which planner | Tool calls | "
        "Wall clock (s) | Valid brief |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    body = [
        f"| {r['case']} | {r['tasks_completed']} | {r['delegations']} | {r['llm_calls']} | "
        f"{r['planner_calls']} | {r['tool_calls']} | {r['seconds']:.3f} | "
        f"{'yes' if r['valid'] else 'no'} |"
        for r in rows
    ]
    return "\n".join([head, *body])


def main(argv: list[str] | None = None, out: Callable[[str], Any] = print) -> int:
    parser = argparse.ArgumentParser(prog="python -m hierarchical_crew.bench")
    parser.add_argument("--latency", type=float, default=0.25, help="seconds per model call")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None, help="also write results here")
    args = parser.parse_args(argv)
    if args.latency < 0 or not 1 <= args.repeats <= 20:
        parser.error("--latency must be >= 0 and --repeats between 1 and 20")
    rows = run_benchmark(args.latency, args.repeats)
    out(table(rows))
    if args.json is not None:
        meta = {
            "date": date.today().isoformat(),
            "python": platform.python_version(),
            "crewai": metadata.version("crewai"),
            "latency_per_call_s": args.latency,
            "repeats": args.repeats,
            "statistic": "median wall clock",
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
