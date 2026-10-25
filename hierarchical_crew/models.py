"""The structured output of the crew: a launch brief (``Task.output_pydantic``)."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

__all__ = ["LaunchBrief", "extract_json_object", "parse_brief"]

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LaunchBrief(BaseModel):
    """What the launch crew hands over: who to sell to, at what price, with which words.

    The schema checks shape only (types, sizes). Whether a brief is *acceptable* -
    non-empty copy, a real product, a price with margin, no unsupported claims - is the
    guardrail's job, so a weak draft reaches the guardrail and gets a retry instead of
    failing CrewAI's pydantic conversion outright.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    product: str = Field(max_length=60)
    target_segment: str = Field(default="", max_length=120)
    positioning: str = Field(default="", max_length=400)
    tagline: str = Field(default="", max_length=90)
    price_usd: float = Field(default=0.0, ge=0, le=100_000)
    price_rationale: str = Field(default="", max_length=400)
    key_messages: list[str] = Field(default_factory=list, max_length=5)
    risks: list[str] = Field(default_factory=list, max_length=6)
    qa_notes: list[str] = Field(default_factory=list, max_length=8)


def _first_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in a model reply (tolerates fences, prose and <think>)."""
    cleaned = _THINK.sub("", text)
    for block in _FENCE.findall(cleaned):
        found = _first_object(block)
        if found is not None:
            return found
    return _first_object(cleaned)


def parse_brief(value: Any) -> LaunchBrief:
    """Coerce a model, dict or reply text into a validated :class:`LaunchBrief`."""
    if isinstance(value, LaunchBrief):
        return value
    if isinstance(value, BaseModel):
        return LaunchBrief.model_validate(value.model_dump())
    if isinstance(value, dict):
        return LaunchBrief.model_validate(value)
    data = extract_json_object(str(value))
    if data is None:
        raise ValueError("the output contains no JSON object")
    return LaunchBrief.model_validate(data)
