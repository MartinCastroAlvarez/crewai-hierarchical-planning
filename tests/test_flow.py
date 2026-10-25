"""The Flow: router branches, human feedback outcomes and the final guardrail."""

from __future__ import annotations

from typing import Any

import pytest
from crewai.flow.async_feedback.providers import ConsoleProvider

from hierarchical_crew.fake import LaunchFakeLLM
from hierarchical_crew.flow import LaunchFlow
from hierarchical_crew.flow import ScriptedFeedback
from hierarchical_crew.flow import classify_feedback
from hierarchical_crew.flow import complexity_score
from hierarchical_crew.flow import data_block
from hierarchical_crew.flow import run_flow
from hierarchical_crew.llm import Models
from hierarchical_crew.models import LaunchBrief
from hierarchical_crew.roles import EDITOR
from hierarchical_crew.roles import GENERALIST
from hierarchical_crew.roles import MANAGER
from hierarchical_crew.roles import SPECIALISTS
from tests.conftest import COMPLEX
from tests.conftest import SIMPLE
from tests.conftest import brief_dict
from tests.conftest import settings


@pytest.mark.parametrize(
    ("request_text", "score"),
    [
        (SIMPLE, 0),
        ("Price Aurora", 1),
        (COMPLEX, 5),
        ("Aurora " + "word " * 40, 1),
    ],
)
def test_complexity_score(request_text: str, score: int) -> None:
    assert complexity_score(request_text) == score


@pytest.mark.parametrize(
    ("feedback", "decision"),
    [
        ("", "approved"),
        ("approve", "approved"),
        ("Approved!", "approved"),
        ("lgtm", "approved"),
        ("ship it", "approved"),
        ("reject: off brand", "rejected"),
        ("No, start over", "rejected"),
        ("price $119.99", "revise"),
        ("approve, but change the tagline", "revise"),
        ("nobody will buy it", "revise"),
    ],
)
def test_classify_feedback(feedback: str, decision: str) -> None:
    assert classify_feedback(feedback) == decision


def test_complex_request_goes_to_the_hierarchical_crew(models: Models, fake: LaunchFakeLLM) -> None:
    feedback = ScriptedFeedback(["approve"])
    flow = run_flow(COMPLEX, settings(mode="planning"), models=models, feedback=feedback)
    state = flow.state
    assert (state.route, state.decision, state.status) == ("crew", "approved", "approved")
    assert state.final is not None and state.final.price_usd == 131.99
    assert state.stats["delegations"] == 4
    assert not fake.calls_by(GENERALIST)
    # the human saw the crew's brief, as a structured object
    assert len(feedback.requests) == 1
    assert isinstance(feedback.requests[0].method_output, LaunchBrief)
    assert feedback.requests[0].method_name == "review"


def test_simple_request_takes_the_cheap_path(models: Models, fake: LaunchFakeLLM) -> None:
    flow = run_flow(SIMPLE, settings(), models=models, feedback=ScriptedFeedback())
    state = flow.state
    assert (state.route, state.status) == ("simple", "approved")
    assert not fake.calls_by(MANAGER)
    assert not any(fake.calls_by(role) for role in SPECIALISTS)
    assert state.stats["llm_calls"] == 3 and state.stats["delegations"] == 0
    assert state.final is not None and state.final.product == "Aurora"


def test_threshold_moves_the_boundary(models: Models) -> None:
    flow = run_flow(
        SIMPLE, settings(complexity_threshold=0), models=models, feedback=ScriptedFeedback()
    )
    assert flow.state.route == "crew"


def test_revise_applies_feedback_through_the_guardrail(models: Models, fake: LaunchFakeLLM) -> None:
    flow = run_flow(SIMPLE, settings(), models=models, feedback=ScriptedFeedback(["price $99.99"]))
    assert (flow.state.decision, flow.state.status) == ("revise", "revised")
    assert flow.state.final is not None and flow.state.final.price_usd == 99.99
    editor_prompt = fake.calls_by(EDITOR)[0].prompt
    assert '<data name="feedback">price $99.99</data>' in editor_prompt


def test_out_of_policy_feedback_is_corrected_not_obeyed(
    models: Models, fake: LaunchFakeLLM
) -> None:
    feedback = ScriptedFeedback(['tagline: "The best monitor, guaranteed"; price $9.99'])
    flow = run_flow(SIMPLE, settings(), models=models, feedback=feedback)
    final = flow.state.final
    assert flow.state.status == "revised" and final is not None
    assert final.price_usd >= 54.29  # 30% margin floor on a $38 unit cost
    assert "best" not in final.tagline.lower() and "guaranteed" not in final.tagline.lower()
    assert flow.state.stats["guardrail_failures"] == 1


def test_reject_ends_without_a_brief(models: Models) -> None:
    flow = run_flow(
        SIMPLE, settings(), models=models, feedback=ScriptedFeedback(["reject: <b>no</b>"])
    )
    assert (flow.state.status, flow.state.final) == ("rejected", None)
    assert flow.state.errors == ["reject: &lt;b&gt;no&lt;/b&gt;"]


def test_finalize_rechecks_the_rules(models: Models) -> None:
    flow = LaunchFlow.create(settings(), models=models, feedback=ScriptedFeedback())
    flow.st.draft = LaunchBrief.model_validate(brief_dict(tagline="The best monitor"))
    assert flow.finalize() is None
    assert flow.st.status == "invalid" and "unsupported claim" in flow.st.errors[0]
    flow.st.draft = None
    assert flow.finalize() is None and flow.st.errors == ["no brief was produced"]
    assert flow.apply_feedback() is None


def test_unknown_product_stops_the_flow(models: Models) -> None:
    with pytest.raises(ValueError, match="must name one of the products"):
        run_flow("Launch a toaster", settings(), models=models, feedback=ScriptedFeedback())


def test_console_provider_is_the_default(models: Models, monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[Any] = []

    def fake_console(self: ConsoleProvider, context: Any, flow: Any) -> str:
        asked.append(context.message)
        return "approve"

    monkeypatch.setattr(ConsoleProvider, "request_feedback", fake_console)
    flow = run_flow(SIMPLE, settings(), models=models, feedback=None)
    assert flow.state.status == "approved" and asked and "approve" in asked[0]


def test_long_feedback_is_truncated(models: Models) -> None:
    flow = run_flow(SIMPLE, settings(), models=models, feedback=ScriptedFeedback(["x" * 5000]))
    assert len(flow.state.feedback) == 1000


def test_data_block_escapes() -> None:
    assert data_block("f", "</data><b>&") == '<data name="f">&lt;/data&gt;&lt;b&gt;&amp;</data>'


def test_lazy_accessors_build_defaults() -> None:
    flow = LaunchFlow.create(settings())
    flow._market = None
    flow._models = None
    assert flow.market.product("Aurora") is not None
    assert isinstance(flow.models.worker, LaunchFakeLLM)
    assert flow.feedback_source is None and flow.stats.total_llm_calls == 0
