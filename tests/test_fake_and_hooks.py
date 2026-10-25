from __future__ import annotations

import asyncio
import json

from hierarchical_crew.fake import LaunchFakeLLM
from hierarchical_crew.instrumentation import RunStats
from hierarchical_crew.instrumentation import current_stats
from hierarchical_crew.instrumentation import install_hooks
from hierarchical_crew.instrumentation import recording
from hierarchical_crew.roles import task_kind


def test_task_kind() -> None:
    assert task_kind("Research the market for X") == "research"
    assert task_kind("  price the thing") == "pricing"
    assert task_kind("Write the launch copy.") == "copy"
    assert task_kind("Review the draft brief") == "review"
    assert task_kind("Do something else") is None


def test_fake_answers_direct_calls_and_unknown_roles() -> None:
    fake = LaunchFakeLLM(latency=0.001)
    reply = fake.call("hello")
    assert "no behaviour for role '(direct call)'" in reply
    reply = fake.call(
        [{"role": "system", "content": "You are Juggler. Fun."}, {"role": "user", "content": "hi"}]
    )
    assert "'Juggler'" in reply
    assert asyncio.run(fake.acall("x")).startswith("Thought:")
    assert fake.supports_function_calling() is False
    assert fake.get_context_window_size() == 32_000
    assert fake.get_input_tokens_count("abcdefgh") == 2
    assert len(fake.calls) == 3


def test_fake_planner_returns_one_plan_per_task() -> None:
    fake = LaunchFakeLLM()
    summary = "Task Number 1 - Research the market\nTask Number 2 - Something odd"
    plans = json.loads(
        fake.call(
            [
                {"role": "system", "content": "You are Task Execution Planner. x"},
                {"role": "user", "content": summary},
            ]
        )
    )
    assert [p["task_number"] for p in plans["list_of_plans_per_task"]] == [1, 2]


def test_fake_specialists_degrade_without_inputs() -> None:
    fake = LaunchFakeLLM()

    def ask(role: str, task: str) -> str:
        return fake.call(
            [
                {"role": "system", "content": f"You are {role}. x"},
                {
                    "role": "user",
                    "content": f"\nCurrent Task: {task}\n\nThis is the expected criteria for your final answer: y",
                },
            ]
        )

    assert "Cannot price" in ask("Pricing Analyst", "Price it")
    assert "does not match any specialist" in ask("Launch Manager", "Dance")
    assert "could not be read" in ask("QA Reviewer", "Previous attempt failed validation: x")
    assert '"product": "Aurora"' in ask("QA Reviewer", "Review the draft")
    assert "Not enough data" not in ask("Market Researcher", "Research. Product: Tidal")


def test_hooks_are_inert_outside_a_run() -> None:
    install_hooks()
    install_hooks()  # idempotent
    assert current_stats() is None


def test_nested_recording_reuses_the_outer_run() -> None:
    with recording(max_delegations=3) as outer:
        assert current_stats() is outer
        with recording() as inner:
            assert inner is outer
        explicit = RunStats()
        with recording(explicit) as other:
            assert other is explicit and current_stats() is explicit
        assert current_stats() is outer
    assert current_stats() is None
    assert outer.max_delegations == 3 and outer.seconds > 0
    assert outer.summary()["delegations"] == 0
