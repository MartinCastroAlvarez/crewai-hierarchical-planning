"""The crew's two tools: sandboxed market-data lookup and a safe calculator.

There is deliberately no code-execution, file-path, shell or web tool. Both tools take
small, pydantic-validated arguments and return a short string the model can read.

* :class:`MarketDataTool` searches one of three in-memory tables (see
  :mod:`hierarchical_crew.dataset`). The model names a dataset, never a path.
* :class:`CalculatorTool` walks the AST of an arithmetic expression against an
  allowlist. No ``eval``: numbers, ``+ - * / // % **``, unary signs and a few math
  functions, with caps on length, node count, exponent and magnitude.
"""

from __future__ import annotations

import ast
import json
import math
import operator
from collections.abc import Callable
from typing import Any
from typing import Literal

from crewai.tools import BaseTool
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from hierarchical_crew.dataset import DatasetError
from hierarchical_crew.dataset import MarketData

__all__ = [
    "CalculatorTool",
    "MarketDataTool",
    "ToolError",
    "calculate",
]

MAX_EXPRESSION = 200


class ToolError(ValueError):
    """A tool refused its input. The message is safe to show to the model."""


# --------------------------------------------------------------------------- calculator
_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
}
_MAX_NODES = 64
_MAX_EXPONENT = 32
_MAX_MAGNITUDE = 1e15


def _number(value: Any) -> float | int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ToolError("only numbers are allowed")
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError("the result is not a finite number")
    if abs(value) > _MAX_MAGNITUDE:
        raise ToolError("the result is too large")
    return value


def _evaluate(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        return _number(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _number(_UNARY[type(node.op)](_evaluate(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > _MAX_EXPONENT or abs(left) > 1e6):
            raise ToolError("exponent or base too large")
        try:
            return _number(_BINARY[type(node.op)](left, right))
        except ZeroDivisionError:
            raise ToolError("division by zero") from None
        except OverflowError:
            raise ToolError("the result is too large") from None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
        and 1 <= len(node.args) <= 4
    ):
        args = [_evaluate(a) for a in node.args]
        try:
            return _number(_FUNCTIONS[node.func.id](*args))
        except (ValueError, TypeError):
            raise ToolError(f"invalid arguments for {node.func.id}()") from None
    raise ToolError(f"unsupported expression element: {type(node).__name__}")


def _format(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}".rstrip("0").rstrip(".")


def calculate(expression: str) -> str:
    """Evaluate a small arithmetic expression safely and return the result as text."""
    text = expression.strip().replace("$", "")
    if not text:
        raise ToolError("empty expression")
    if len(text) > MAX_EXPRESSION:
        raise ToolError(f"expression longer than {MAX_EXPRESSION} characters")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ToolError("not an arithmetic expression") from None
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise ToolError("expression too complex")
    return _format(_evaluate(tree))


class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(
        ...,
        max_length=MAX_EXPRESSION,
        description="Arithmetic only, e.g. 'round(139 * 0.95, 2)'. "
        "Allowed: numbers, + - * / // % **, abs round min max sqrt floor ceil.",
    )


class CalculatorTool(BaseTool):
    """Safe arithmetic for the pricing analyst."""

    name: str = "calculator"
    description: str = (
        "Evaluate an arithmetic expression (prices, margins, percentages). "
        "Returns the number, or 'error: ...'."
    )
    args_schema: type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        try:
            return calculate(expression)
        except ToolError as exc:
            return f"error: {exc}"


# --------------------------------------------------------------------------- market data
class MarketDataInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["products", "segments", "competitors"] = Field(
        ..., description="Which table to search: products, segments or competitors."
    )
    query: str = Field(
        default="",
        max_length=120,
        description="Words that must all appear in a row, e.g. 'air-quality monitor'. "
        "Empty returns the first rows.",
    )


class MarketDataTool(BaseTool):
    """Keyword search over the bundled market tables. Read-only, no paths."""

    name: str = "market_data"
    description: str = (
        "Search the local market dataset. Tables: products (sku, name, category, "
        "unit_cost_usd), segments (category, segment, region, households_k, adoption_pct, "
        "wtp_usd = willingness to pay) and competitors (category, competitor, product, "
        "price_usd, rating). Returns matching rows as JSON."
    )
    args_schema: type[BaseModel] = MarketDataInput
    market: MarketData = Field(exclude=True)

    def _run(self, dataset: str, query: str = "") -> str:
        try:
            rows = self.market.search(dataset, query[:120])
        except DatasetError as exc:
            return f"error: {exc}"
        if not rows:
            return f"no rows in {dataset} match {query!r}"
        return json.dumps({"dataset": dataset, "rows": rows}, ensure_ascii=False)
