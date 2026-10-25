from __future__ import annotations

import json

import pytest
from crewai.tasks.task_output import TaskOutput

from hierarchical_crew.dataset import MarketData
from hierarchical_crew.guardrails import brief_errors
from hierarchical_crew.guardrails import make_brief_guardrail
from hierarchical_crew.guardrails import scrub_claims
from hierarchical_crew.instrumentation import recording
from hierarchical_crew.models import LaunchBrief
from tests.conftest import brief_dict


def errors_for(market: MarketData, **changes: object) -> list[str]:
    return brief_errors(LaunchBrief.model_validate(brief_dict(**changes)), market)


def test_a_good_brief_passes(market: MarketData) -> None:
    assert errors_for(market) == []


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"product": "Toaster"}, "unknown product"),
        ({"price_usd": 40.0}, "minimum for a 30% gross margin"),
        ({"price_usd": 0.0}, "outside the competitive range"),
        ({"price_usd": 400.0}, "outside the competitive range"),
        ({"tagline": "The best monitor."}, "unsupported claim: 'best'"),
        ({"tagline": "The #1 monitor."}, "unsupported claim: '#1'"),
        ({"positioning": "Guaranteed cleaner air."}, "positioning makes an unsupported claim"),
        ({"key_messages": ["100% accurate"]}, "key_messages[0] makes an unsupported claim"),
        ({"key_messages": ["Order at https://evil.example"]}, "contains a link"),
        ({"key_messages": ["Mail ceo@example.com"]}, "e-mail address"),
        ({"tagline": "<script>x</script>"}, "markup"),
        (
            {"tagline": "one two three four five six seven eight nine ten eleven twelve 13"},
            "12 words",
        ),
        ({"tagline": ""}, "tagline is empty"),
        ({"key_messages": [" "]}, "key_messages is empty"),
        ({"risks": []}, "risks is empty"),
    ],
)
def test_rules(market: MarketData, changes: dict[str, object], fragment: str) -> None:
    assert any(fragment in e for e in errors_for(market, **changes)), errors_for(market, **changes)


def test_claim_matching_is_word_bound(market: MarketData) -> None:
    assert errors_for(market, positioning="Our bestselling line, now with a curated case.") == []
    assert scrub_claims("Meet Aurora, the best monitor!") == "Meet Aurora, the monitor!"
    assert scrub_claims("Guaranteed 100% risk-free") == ""


def output(raw: str) -> TaskOutput:
    return TaskOutput(description="d", raw=raw, agent="QA Reviewer")


def test_guardrail_accepts_and_counts(market: MarketData) -> None:
    guardrail = make_brief_guardrail(market)
    with recording() as stats:
        ok, result = guardrail(output(json.dumps(brief_dict())))
        bad, error = guardrail(output(json.dumps(brief_dict(tagline="The best."))))
        broken, parse_error = guardrail(output("not json"))
    assert ok and LaunchBrief.model_validate_json(result).product == "Aurora"
    assert not bad and "unsupported claim" in error
    assert not broken and "not valid LaunchBrief JSON" in parse_error
    assert (stats.guardrail_passes, stats.guardrail_failures) == (1, 2)
    assert len(stats.guardrail_errors) == 2


def test_guardrail_uses_the_pydantic_output_when_present(market: MarketData) -> None:
    out = output("ignored")
    out.pydantic = LaunchBrief.model_validate(brief_dict())
    assert make_brief_guardrail(market)(out)[0] is True
    assert make_brief_guardrail(market)(output(json.dumps(brief_dict())))[0] is True
