"""The crew's control flow with the fake model: who gets delegated what, when the plan
is injected, how the guardrail retries, and how the hierarchy is bounded."""

from __future__ import annotations

import json

import pytest
from crewai import Crew
from crewai import Process

from hierarchical_crew.crew import build_agents
from hierarchical_crew.crew import build_crew
from hierarchical_crew.crew import build_manager
from hierarchical_crew.crew import build_tasks
from hierarchical_crew.crew import clean_request
from hierarchical_crew.crew import finished
from hierarchical_crew.crew import resolve_product
from hierarchical_crew.crew import run_crew
from hierarchical_crew.dataset import MarketData
from hierarchical_crew.fake import PLAN_MARKER
from hierarchical_crew.fake import LaunchFakeLLM
from hierarchical_crew.guardrails import brief_errors
from hierarchical_crew.instrumentation import recording
from hierarchical_crew.llm import Models
from hierarchical_crew.roles import BUILTIN_MANAGER
from hierarchical_crew.roles import COPYWRITER
from hierarchical_crew.roles import MANAGER
from hierarchical_crew.roles import PLANNER
from hierarchical_crew.roles import PRICING
from hierarchical_crew.roles import QA
from hierarchical_crew.roles import RESEARCHER
from hierarchical_crew.roles import SPECIALISTS
from tests.conftest import COMPLEX
from tests.conftest import brief_dict
from tests.conftest import settings

ORDER = [RESEARCHER, PRICING, COPYWRITER, QA]
TASK_HEADS = {
    RESEARCHER: "Research the market",
    PRICING: "Price the product",
    COPYWRITER: "Write the launch copy",
    QA: "Review the draft",
}


def delegate(coworker: str, task: str = "Research the market.\nProduct: Aurora") -> str:
    args = {"task": task, "context": "none", "coworker": coworker}
    return f"Thought: delegate\nAction: delegate_work_to_coworker\nAction Input: {json.dumps(args)}"


def test_hierarchical_delegates_each_task_to_its_owner(
    market: MarketData, models: Models, fake: LaunchFakeLLM
) -> None:
    result = run_crew(COMPLEX, settings(mode="hierarchical"), models=models, market=market)
    assert result.stats.delegated_to() == ORDER
    assert result.stats.blocked_delegations == 0
    assert result.stats.tasks_completed == 4
    assert brief_errors(result.brief, market) == []
    # each specialist only ever worked on its own task
    for role in SPECIALISTS:
        prompts = [c.prompt for c in fake.calls_by(role)]
        assert prompts, role
        assert all(TASK_HEADS[role] in p for p in prompts), role
        others = [h for r, h in TASK_HEADS.items() if r != role]
        assert not any(h in p.split("Current Task:")[1][:80] for p in prompts for h in others)
    # the manager did no work itself: its only tool is delegation
    assert set(result.stats.tool_calls) == {
        "delegate_work_to_coworker",
        "market_data",
        "calculator",
    }
    assert result.stats.llm_calls[MANAGER] == 8  # delegate + accept, per task
    assert PLANNER not in result.stats.llm_calls


def test_delegated_task_carries_the_upstream_context(models: Models, fake: LaunchFakeLLM) -> None:
    run_crew(COMPLEX, settings(mode="hierarchical"), models=models)
    pricing_prompt = fake.calls_by(PRICING)[0].prompt
    assert "Competitor prices: low $79.00, median $139.00" in pricing_prompt
    qa_prompt = fake.calls_by(QA)[0].prompt
    assert "Recommended price: $131.99" in qa_prompt and "Tagline:" in qa_prompt


def test_sequential_has_no_manager(models: Models) -> None:
    result = run_crew(COMPLEX, settings(mode="sequential"), models=models)
    assert result.stats.delegations == []
    assert MANAGER not in result.stats.llm_calls
    assert result.stats.total_llm_calls == 9
    assert result.brief.price_usd == 131.99


def test_planning_runs_the_planner_first_and_injects_the_plan(
    models: Models, fake: LaunchFakeLLM
) -> None:
    result = run_crew(COMPLEX, settings(mode="planning"), models=models)
    assert fake.calls[0].role == PLANNER
    assert fake.calls[0].structured == "PlannerTaskPydanticOutput"
    assert result.stats.llm_calls[PLANNER] == 1
    assert result.stats.delegated_to() == ORDER
    for number, role in enumerate(ORDER, start=1):
        marker = f"{PLAN_MARKER} for task {number}:"
        # the plan is appended to the task the manager sees ...
        manager_prompts = [c.prompt for c in fake.calls_by(MANAGER) if TASK_HEADS[role] in c.prompt]
        assert manager_prompts and all(marker in p for p in manager_prompts)
        # ... and travels with the delegation to the specialist
        assert marker in fake.calls_by(role)[0].prompt
    assert all(t.description.count(PLAN_MARKER) == 1 for t in result.output.tasks_output)


def test_no_plan_without_planning(models: Models, fake: LaunchFakeLLM) -> None:
    run_crew(COMPLEX, settings(mode="hierarchical"), models=models)
    assert not any(PLAN_MARKER in c.prompt for c in fake.calls)


def test_builtin_manager_from_manager_llm(models: Models) -> None:
    result = run_crew(COMPLEX, settings(mode="planning", manager="llm"), models=models)
    assert BUILTIN_MANAGER in result.stats.llm_calls
    assert MANAGER not in result.stats.llm_calls
    assert result.stats.delegated_to() == ORDER


BAD = "Thought: done\nFinal Answer: " + json.dumps(brief_dict(tagline="The best monitor ever."))


def test_guardrail_retries_then_rejects(models: Models, fake: LaunchFakeLLM) -> None:
    fake.script = {QA: [BAD, BAD, BAD]}
    with (
        recording() as stats,
        pytest.raises(Exception, match="guardrail validation after 2 retries"),
    ):
        run_crew(
            COMPLEX, settings(mode="sequential", guardrail_retries=2), models=models, stats=stats
        )
    assert len(fake.calls_by(QA)) == 3
    assert stats.guardrail_failures == 3 and stats.guardrail_passes == 0
    assert all("unsupported claim: 'best'" in e for e in stats.guardrail_errors)


def test_guardrail_feedback_lets_the_critic_repair(models: Models, fake: LaunchFakeLLM) -> None:
    fake.script = {QA: [BAD]}
    result = run_crew(COMPLEX, settings(mode="sequential"), models=models)
    qa_calls = fake.calls_by(QA)
    assert len(qa_calls) == 2
    assert "Previous attempt failed validation" in qa_calls[1].prompt
    assert "unsupported claim: 'best'" in qa_calls[1].prompt
    assert result.stats.guardrail_failures == 1 and result.stats.guardrail_passes == 1
    assert "best" not in result.brief.tagline.lower()
    assert any("Fixed after the guardrail" in n for n in result.brief.qa_notes)


def test_guardrail_retry_in_hierarchical_goes_back_through_the_manager(
    models: Models, fake: LaunchFakeLLM
) -> None:
    fake.script = {QA: [BAD]}
    result = run_crew(COMPLEX, settings(mode="hierarchical"), models=models)
    assert result.stats.delegated_to() == [*ORDER, QA]
    assert result.stats.guardrail_failures == 1
    assert brief_errors(result.brief, MarketData.load()) == []


def research_only_crew(models: Models, market: MarketData, **changes: object) -> Crew:
    s = settings(mode="hierarchical", **changes)
    return Crew(
        agents=list(build_agents(s, models, market).values()),
        tasks=build_tasks(s, market, None)[:1],
        process=Process.hierarchical,
        manager_agent=build_manager(s, models),
        memory=False,
        cache=False,
        tracing=False,
    )


def test_delegation_cap_blocks_extra_delegations(
    market: MarketData, models: Models, fake: LaunchFakeLLM
) -> None:
    tries = [
        delegate(RESEARCHER, f"Research the market, try {i}.\nProduct: Aurora") for i in range(5)
    ]
    fake.script = {MANAGER: [*tries, "Thought: ok\nFinal Answer: done"]}
    crew = research_only_crew(models, market, max_iter=10)
    with recording(max_delegations=2) as stats:
        output = finished(crew.kickoff(inputs={"product": "Aurora", "request": "r"}))
    assert output.raw == "done"
    assert len(stats.allowed_delegations) == 2 and stats.blocked_delegations == 3
    assert len([c for c in fake.calls_by(RESEARCHER) if "Final Answer" in c.reply]) == 2
    assert "Tool execution blocked by hook" in fake.calls_by(MANAGER)[-1].prompt


def test_max_iter_ends_a_manager_that_never_stops(
    market: MarketData, models: Models, fake: LaunchFakeLLM
) -> None:
    fake.script = {MANAGER: [delegate(RESEARCHER)] * 50}
    crew = research_only_crew(models, market, max_iter=3)
    with recording(max_delegations=50) as stats:
        finished(crew.kickoff(inputs={"product": "Aurora", "request": "r"}))
    assert len(fake.calls_by(MANAGER)) <= 5
    assert len(stats.allowed_delegations) <= 3


def test_delegating_to_an_unknown_coworker_reaches_nobody(
    market: MarketData, models: Models, fake: LaunchFakeLLM
) -> None:
    fake.script = {MANAGER: [delegate("Root Shell"), "Thought: ok\nFinal Answer: done"]}
    crew = research_only_crew(models, market)
    with recording() as stats:
        finished(crew.kickoff(inputs={"product": "Aurora", "request": "r"}))
    assert not any(fake.calls_by(role) for role in SPECIALISTS)
    assert stats.delegated_to() == ["Root Shell"]
    assert "Root Shell" in fake.calls_by(MANAGER)[-1].prompt


def test_only_the_manager_may_delegate(market: MarketData, models: Models) -> None:
    agents = build_agents(settings(), models, market)
    assert not any(a.allow_delegation for a in agents.values())
    assert build_manager(settings(), models).allow_delegation
    crew = build_crew(settings(mode="planning"), models, market)
    assert crew.process == Process.hierarchical and crew.planning
    assert crew.memory is False and all(t.agent is None for t in crew.tasks)
    sequential = build_crew(settings(mode="sequential"), models, market)
    assert [t.agent.role if t.agent else None for t in sequential.tasks] == ORDER


def test_request_validation(market: MarketData) -> None:
    assert clean_request("  Launch {Aurora}\x00 now ") == "Launch  Aurora   now"
    with pytest.raises(ValueError, match="empty"):
        clean_request(" \n ")
    with pytest.raises(ValueError, match="longer"):
        clean_request("x" * 2001)
    assert resolve_product("launch lumen soon", market) == "Lumen"
    with pytest.raises(ValueError, match="Aurora, Tidal, Lumen"):
        resolve_product("launch the toaster", market)
    with pytest.raises(TypeError):
        finished("not an output")
