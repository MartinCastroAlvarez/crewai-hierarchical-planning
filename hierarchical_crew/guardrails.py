"""Business rules for a launch brief, used as a CrewAI task guardrail and by the flow.

A guardrail in CrewAI is a function ``(TaskOutput) -> (ok, result_or_error)``. When it
returns ``False``, CrewAI re-runs the task with the error in the prompt, up to
``guardrail_max_retries`` times, and then raises. The same rules run once more in
the flow's final step, after a human may have changed the brief.

Rules (all checked against the dataset, not the model's word):

* the brief parses into :class:`~hierarchical_crew.models.LaunchBrief`;
* the product exists in the dataset;
* the price covers the unit cost with at least a 30% gross margin and sits between
  half the cheapest and 1.25x the most expensive competitor;
* no unsupported claims (superlatives, guarantees, health cures);
* no links, e-mail addresses or markup in customer-facing copy (prompt-injected
  data must not turn the brief into a phishing vehicle);
* the tagline has at most 12 words.
"""

# No ``from __future__ import annotations`` here: CrewAI inspects the guardrail's return
# annotation at runtime and needs the real ``tuple[bool, Any]``, not a string.
import re
from collections.abc import Callable
from typing import Any

from crewai.tasks.task_output import TaskOutput
from pydantic import ValidationError

from hierarchical_crew.dataset import MarketData
from hierarchical_crew.instrumentation import current_stats
from hierarchical_crew.models import LaunchBrief
from hierarchical_crew.models import parse_brief

__all__ = [
    "BANNED_CLAIMS",
    "MIN_MARGIN",
    "brief_errors",
    "make_brief_guardrail",
    "scrub_claims",
]

MIN_MARGIN = 0.30
MAX_TAGLINE_WORDS = 12
BANNED_CLAIMS: tuple[str, ...] = (
    "best",
    "#1",
    "number one",
    "guaranteed",
    "guarantee",
    "risk-free",
    "cure",
    "cures",
    "100%",
    "never fails",
    "world's first",
)
_CLAIM = re.compile(
    "|".join(rf"(?<![\w#]){re.escape(c)}(?![\w%])" for c in BANNED_CLAIMS), re.IGNORECASE
)
_LINK = re.compile(r"(https?://|www\.|\b[\w.+-]+@[\w-]+\.\w+|<[a-z/!])", re.IGNORECASE)


def _copy_fields(brief: LaunchBrief) -> list[tuple[str, str]]:
    fields = [("tagline", brief.tagline), ("positioning", brief.positioning)]
    fields += [(f"key_messages[{i}]", m) for i, m in enumerate(brief.key_messages)]
    return fields


def scrub_claims(text: str) -> str:
    """Remove banned claims from ``text`` (the QA reviewer's fix), tidying spaces."""
    cleaned = _CLAIM.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned).strip()
    return re.sub(r"[\s,;:-]+$", "", cleaned)


def brief_errors(brief: LaunchBrief, market: MarketData) -> list[str]:
    """Every rule the brief breaks, as short messages a model can act on."""
    errors: list[str] = [
        f"{name} is empty"
        for name in ("target_segment", "positioning", "tagline", "price_rationale")
        if not getattr(brief, name).strip()
    ]
    if not [m for m in brief.key_messages if m.strip()]:
        errors.append("key_messages is empty")
    if not brief.risks:
        errors.append("risks is empty: name at least one launch risk")
    product = market.product(brief.product)
    if product is None:
        names = ", ".join(market.product_names())
        return [*errors, f"unknown product {brief.product!r}; the dataset has: {names}"]
    cost = float(product["unit_cost_usd"])
    floor_price = round(cost / (1 - MIN_MARGIN), 2)
    if brief.price_usd < floor_price:
        errors.append(
            f"price ${brief.price_usd:.2f} is below ${floor_price:.2f}, the minimum for a "
            f"{MIN_MARGIN:.0%} gross margin on a ${cost:.2f} unit cost"
        )
    band = market.price_band(product["category"])
    if band is not None:
        low, high = round(band.low * 0.5, 2), round(band.high * 1.25, 2)
        if not low <= brief.price_usd <= high:
            errors.append(
                f"price ${brief.price_usd:.2f} is outside the competitive range "
                f"${low:.2f}-${high:.2f}"
            )
    for name, text in _copy_fields(brief):
        claim = _CLAIM.search(text)
        if claim:
            errors.append(f"{name} makes an unsupported claim: {claim.group(0)!r}")
        if _LINK.search(text):
            errors.append(f"{name} contains a link, an e-mail address or markup")
    if len(brief.tagline.split()) > MAX_TAGLINE_WORDS:
        errors.append(f"tagline has more than {MAX_TAGLINE_WORDS} words")
    return errors


GuardrailFn = Callable[[TaskOutput], tuple[bool, Any]]


def make_brief_guardrail(market: MarketData) -> GuardrailFn:
    """A CrewAI task guardrail that accepts only a brief passing :func:`brief_errors`."""

    def launch_brief_guardrail(output: TaskOutput) -> tuple[bool, Any]:
        stats = current_stats()
        try:
            brief = parse_brief(output.pydantic if output.pydantic is not None else output.raw)
        except (ValidationError, ValueError) as exc:
            error = f"The brief is not valid LaunchBrief JSON: {str(exc)[:400]}"
        else:
            problems = brief_errors(brief, market)
            if not problems:
                if stats is not None:
                    stats.guardrail_passes += 1
                return True, brief.model_dump_json()
            error = "Fix these problems and return the full brief as JSON: " + "; ".join(problems)
        if stats is not None:
            stats.guardrail_failures += 1
            stats.guardrail_errors.append(error)
        return False, error

    return launch_brief_guardrail
