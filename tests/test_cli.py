from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hierarchical_crew.bench import main as bench_main
from hierarchical_crew.bench import run_benchmark
from hierarchical_crew.bench import table
from hierarchical_crew.cli import main
from tests.conftest import COMPLEX
from tests.conftest import SIMPLE


def run(*argv: str) -> tuple[int, str]:
    lines: list[str] = []
    code = main(list(argv), out=lines.append)
    return code, "\n".join(lines)


def test_flow_run_prints_the_brief(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    code, text = run("--provider", "fake", COMPLEX)
    assert code == 0
    assert "route=crew (complexity 5) decision=approved status=approved" in text
    assert "at $131.99" in text and "delegations=4" in text


def test_scripted_feedback_and_json() -> None:
    code, text = run("--provider", "fake", "--json", "--feedback", "reject: no", SIMPLE)
    payload = json.loads(text)
    assert code == 1
    assert (payload["route"], payload["status"], payload["brief"]) == ("simple", "rejected", None)


def test_crew_only_sequential() -> None:
    code, text = run("--provider", "fake", "--crew-only", "--mode", "sequential", "--json", COMPLEX)
    payload = json.loads(text)
    assert code == 0 and payload["stats"]["delegations"] == 0
    assert payload["brief"]["price_usd"] == 131.99


def test_flags_reach_the_settings() -> None:
    code, text = run(
        "--provider",
        "fake",
        "--model",
        "m",
        "--manager",
        "llm",
        "--max-delegations",
        "4",
        "--latency",
        "0",
        "--crew-only",
        "--json",
        COMPLEX,
    )
    assert code == 0 and json.loads(text)["stats"]["llm_calls_by_role"]["Crew Manager"] == 8


def test_mermaid() -> None:
    assert (
        "choose_path -- crew --> run_launch_crew"
        in run("--provider", "fake", "--mermaid", "flow")[1]
    )
    assert "M -- delegate --> Pricing_Analyst" in run("--provider", "fake", "--mermaid", "crew")[1]


def test_errors() -> None:
    assert run("--provider", "fake", "Launch the toaster")[0] == 2
    with pytest.raises(SystemExit):
        main(["--provider", "fake"])
    with pytest.raises(SystemExit):
        main(["--provider", "fake", "x" * 2001])


def test_bad_environment_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://10.0.0.5:11434")
    with pytest.raises(SystemExit):
        main(["--provider", "ollama", SIMPLE])


def test_verbose_flag_runs(capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = run("--provider", "fake", "--verbose", "--crew-only", "--mode", "sequential", COMPLEX)
    assert code == 0


def test_benchmark_shape() -> None:
    rows = {r["case"]: r for r in run_benchmark(latency=0.0, repeats=1)}
    seq = rows["crew: sequential"]
    hier = rows["crew: hierarchical"]
    plan = rows["crew: hierarchical + planning"]
    simple = rows["flow: simple request (routed to one agent)"]
    assert (seq["delegations"], hier["delegations"], plan["delegations"]) == (0, 4, 4)
    assert plan["llm_calls"] == hier["llm_calls"] + 1 == seq["llm_calls"] + 9
    assert plan["planner_calls"] == 1 and hier["planner_calls"] == 0
    assert simple["llm_calls"] < seq["llm_calls"] and simple["route"] == "simple"
    assert all(r["valid"] and r["tasks_completed"] >= 1 for r in rows.values())
    assert table(list(rows.values())).count("\n") == len(rows) + 1


def test_benchmark_cli_writes_json(tmp_path: Path) -> None:
    lines: list[str] = []
    target = tmp_path / "r.json"
    assert (
        bench_main(["--latency", "0", "--repeats", "1", "--json", str(target)], out=lines.append)
        == 0
    )
    data = json.loads(target.read_text())
    assert data["meta"]["repeats"] == 1 and len(data["rows"]) == 6
    with pytest.raises(SystemExit):
        bench_main(["--repeats", "0"])
