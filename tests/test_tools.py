from __future__ import annotations

import json

import pytest

from hierarchical_crew.dataset import MarketData
from hierarchical_crew.tools import CalculatorTool
from hierarchical_crew.tools import MarketDataTool
from hierarchical_crew.tools import ToolError
from hierarchical_crew.tools import calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", "7"),
        ("floor(min(139.0, 149.0) * 0.95) - 0.01", "131.99"),
        ("round((131.99 - 38.0) / 131.99 * 100, 1)", "71.2"),
        ("$129 * 0.9", "116.1"),
        ("-2 ** 3", "-8"),
        ("sqrt(16) + abs(-1) + ceil(0.2) + max(1, 2, 3)", "9"),
        ("7 // 2 + 7 % 2", "4"),
    ],
)
def test_calculator_computes(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "open('/etc/passwd')",
        "(1).__class__",
        "x + 1",
        "'a' * 3",
        "[1, 2][0]",
        "lambda: 1",
        "9 ** 9 ** 9",
        "10 ** 100",
        "1 / 0",
        "sqrt(-1)",
        "round(1.5, ndigits=1)",
        "True + 1",
        "",
        "1 +",
        "1" + " + 1" * 80,
        "+".join(["1"] * 40),
    ],
)
def test_calculator_refuses(expression: str) -> None:
    with pytest.raises(ToolError):
        calculate(expression)


def test_calculator_tool_reports_errors_as_text() -> None:
    tool = CalculatorTool()
    assert tool.run(expression="2 * 21") == "42"
    assert tool.run(expression="__import__('os')").startswith("error:")


def test_market_tool_returns_json_rows(market: MarketData) -> None:
    tool = MarketDataTool(market=market)
    payload = json.loads(tool.run(dataset="competitors", query="air-quality monitor"))
    assert payload["dataset"] == "competitors"
    assert len(payload["rows"]) == 5
    assert {r["category"] for r in payload["rows"]} == {"air-quality monitor"}


def test_market_tool_handles_misses_and_bad_names(market: MarketData) -> None:
    tool = MarketDataTool(market=market)
    assert tool._run(dataset="products", query="toaster").startswith("no rows")
    assert tool._run(dataset="../../etc/passwd").startswith("error: unknown dataset")


def test_market_tool_schema_only_names_datasets() -> None:
    schema = MarketDataTool.model_fields["args_schema"].default.model_json_schema()
    assert schema["properties"]["dataset"]["enum"] == ["products", "segments", "competitors"]
    assert schema["additionalProperties"] is False
