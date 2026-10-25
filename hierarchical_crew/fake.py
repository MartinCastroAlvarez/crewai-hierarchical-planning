"""A deterministic, offline stand-in for the model: :class:`LaunchFakeLLM`.

It subclasses CrewAI's own ``BaseLLM`` (the documented extension point for custom
models), so CrewAI drives it exactly like ollama: ReAct text in, ReAct text out, the
same executor, tools, delegation tools, planner, converter and guardrail retries. It
reads the agent role and the prompt CrewAI built and answers like that agent would:

* the **manager** delegates each task to the specialist that owns it
  (:data:`~hierarchical_crew.roles.TASK_ROUTES`), passing the task and its context on,
  and returns the specialist's answer;
* the **researcher** calls ``market_data`` three times and summarises real rows;
* the **pricing analyst** calls ``calculator`` twice with numbers from the research;
* the **copywriter** drafts copy - with a deliberate superlative ("best") so the
  critic has something to catch;
* the **QA reviewer** (the critic) assembles the brief, strips unsupported claims, and
  repairs a brief the guardrail rejected using the guardrail's own error message;
* CrewAI's **planner** (``planning=True``) returns a plan per task;
* the flow's **generalist** and **editor** produce a brief in one go.

Tests can override any role with canned replies (``script={role: [reply, ...]}``) and
inspect every call in :attr:`LaunchFakeLLM.calls`. ``latency`` sleeps per call, so the
benchmark measures orchestration (round trips), not a model.
"""

from __future__ import annotations

import html
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any
from typing import cast

from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel
from pydantic import Field
from pydantic import PrivateAttr
from pydantic import model_validator

from hierarchical_crew.guardrails import scrub_claims
from hierarchical_crew.models import extract_json_object
from hierarchical_crew.roles import BUILTIN_MANAGER
from hierarchical_crew.roles import COPYWRITER
from hierarchical_crew.roles import EDITOR
from hierarchical_crew.roles import GENERALIST
from hierarchical_crew.roles import MANAGER
from hierarchical_crew.roles import PLANNER
from hierarchical_crew.roles import PRICING
from hierarchical_crew.roles import QA
from hierarchical_crew.roles import RESEARCHER
from hierarchical_crew.roles import TASK_ROUTES
from hierarchical_crew.roles import task_kind

__all__ = ["FakeCall", "LaunchFakeLLM", "PLAN_MARKER"]

PLAN_MARKER = "Execution plan (crew planner)"
_ROLE_LINE = re.compile(r"You are (.+?)\.\s", re.DOTALL)
_STEP = re.compile(
    r"Action:\s*(?P<action>[^\n]+)\nAction Input:\s*(?P<input>\{.*?\})\s*\nObservation:\s*"
    r"(?P<observation>.*)",
    re.DOTALL,
)
_PRODUCT = re.compile(r"Product:\s*([A-Z][\w-]*)")
_MONEY = r"\$([0-9]+(?:\.[0-9]+)?)"


@dataclass(frozen=True, slots=True)
class FakeCall:
    """One call the fake model answered."""

    role: str
    prompt: str
    reply: str
    structured: str | None


@dataclass(frozen=True, slots=True)
class _Step:
    action: str
    args: dict[str, Any]
    observation: str


def _text(messages: str | list[Any]) -> list[dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    out: list[dict[str, str]] = []
    for m in messages:
        content = m.get("content", "") if isinstance(m, dict) else str(m)
        if isinstance(content, list):
            content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        out.append({"role": str(m.get("role", "user")), "content": str(content or "")})
    return out


def _between(text: str, start: str, ends: tuple[str, ...]) -> str:
    index = text.find(start)
    if index < 0:
        return ""
    rest = text[index + len(start) :]
    cut = min((i for i in (rest.find(e) for e in ends) if i >= 0), default=len(rest))
    return rest[:cut].strip()


def _react(action: str, args: dict[str, Any], thought: str) -> str:
    return f"Thought: {thought}\nAction: {action}\nAction Input: {json.dumps(args)}"


def _final(answer: str) -> str:
    return f"Thought: I now know the final answer\nFinal Answer: {answer}"


def _money(pattern: str, text: str) -> float | None:
    found = re.search(pattern + r"\s*" + _MONEY, text, re.IGNORECASE)
    return float(found.group(1)) if found else None


def _charm(price: float) -> float:
    """The .99 price at or just above ``price``."""
    return math.ceil(price + 0.01) - 0.01


class LaunchFakeLLM(BaseLLM):
    """Role-aware scripted model for tests, examples and the benchmark."""

    model: str = "fake-launch"
    provider: str = "fake"
    latency: float = 0.0
    script: dict[str, list[str]] = Field(default_factory=dict)
    """Canned replies per role, used first and in order; built-in behaviour after."""
    calls: list[FakeCall] = Field(default_factory=list)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @model_validator(mode="before")
    @classmethod
    def _default_model(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("model"):
            return {**data, "model": "fake-launch"}
        return data

    # ------------------------------------------------------------------ BaseLLM API
    def call(
        self,
        messages: str | list[Any],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        msgs = _text(messages)
        self._invoke_before_llm_call_hooks(cast(Any, msgs), from_agent)
        role = str(getattr(from_agent, "role", "") or self._role_from(msgs))
        if self.latency:
            time.sleep(self.latency)
        with self._lock:
            queue = self.script.get(role)
            scripted = queue.pop(0) if queue else None
        reply = scripted if scripted is not None else self._respond(role, msgs, response_model)
        prompt = "\n\n".join(m["content"] for m in msgs)
        with self._lock:
            self.calls.append(
                FakeCall(role, prompt, reply, response_model.__name__ if response_model else None)
            )
        return reply

    async def acall(
        self,
        messages: str | list[Any],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        return self.call(
            messages, tools, callbacks, available_functions, from_task, from_agent, response_model
        )

    def supports_function_calling(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 32_000

    def get_input_tokens_count(
        self,
        messages: str | list[Any],
        tools: list[dict[str, Any]] | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> int:
        """One token per four characters, like CrewAI's fallback, without its warning."""
        return (sum(len(m["content"]) for m in _text(messages)) + 3) // 4

    # ------------------------------------------------------------------ helpers
    def calls_by(self, role: str) -> list[FakeCall]:
        return [c for c in self.calls if c.role == role]

    @staticmethod
    def _role_from(msgs: list[dict[str, str]]) -> str:
        for m in msgs:
            if m["role"] == "system":
                found = _ROLE_LINE.search(m["content"] + " ")
                if found:
                    return found.group(1).strip()
        return "(direct call)"

    @staticmethod
    def _task(msgs: list[dict[str, str]]) -> str:
        for m in msgs:
            if m["role"] == "user" and "Current Task:" in m["content"]:
                return _between(
                    m["content"], "Current Task:", ("\n\nThis is the expected criteria",)
                )
        return msgs[-1]["content"] if msgs else ""

    @staticmethod
    def _context(msgs: list[dict[str, str]]) -> str:
        for m in msgs:
            if m["role"] == "user" and "This is the context you're working with:" in m["content"]:
                return _between(
                    m["content"],
                    "This is the context you're working with:",
                    ("\n\nBegin!", "\n\nProvide your complete response"),
                )
        return ""

    @staticmethod
    def _steps(msgs: list[dict[str, str]]) -> list[_Step]:
        steps: list[_Step] = []
        for m in msgs:
            if m["role"] != "assistant":
                continue
            found = _STEP.search(m["content"])
            if not found:
                continue
            try:
                args = json.loads(found.group("input"))
            except json.JSONDecodeError:
                args = {}
            observation = found.group("observation").split("\n\n\nYou ONLY have access")[0]
            steps.append(_Step(found.group("action").strip(), args, observation.strip()))
        return steps

    def _respond(
        self, role: str, msgs: list[dict[str, str]], response_model: type[BaseModel] | None
    ) -> str:
        if role == PLANNER or (response_model and response_model.__name__.startswith("Planner")):
            return self._plan(msgs)
        handlers = {
            MANAGER: self._manager,
            BUILTIN_MANAGER: self._manager,
            RESEARCHER: self._researcher,
            PRICING: self._pricing,
            COPYWRITER: self._copywriter,
            QA: self._reviewer,
            GENERALIST: self._generalist,
            EDITOR: self._editor,
        }
        handler = handlers.get(role)
        if handler is None:
            return _final(f"(the fake model has no behaviour for role {role!r})")
        return handler(msgs)

    # ------------------------------------------------------------------ planner
    def _plan(self, msgs: list[dict[str, str]]) -> str:
        text = "\n".join(m["content"] for m in msgs)
        steps = {
            "research": "1. Look up the product row. 2. Rank its category's segments by "
            "households x adoption. 3. Report competitor prices: low, median, high.",
            "pricing": "1. Start from the lower of the competitor median and the top "
            "segment's willingness to pay. 2. Undercut it by 5% to a .99 price with the "
            "calculator. 3. Check the gross margin against the unit cost.",
            "copy": "1. Speak to the top segment. 2. One short tagline, a positioning line "
            "and three key messages grounded in the product facts and the price.",
            "review": "1. Assemble the LaunchBrief JSON from the research, price and copy. "
            "2. Strip unsupported claims. 3. List the launch risks.",
        }
        plans = []
        for number, description in re.findall(r"Task Number (\d+) - ([^\n]+)", text):
            kind = task_kind(description) or "review"
            plans.append(
                {
                    "task_number": int(number),
                    "task": description.strip()[:120],
                    "plan": f"\n\n{PLAN_MARKER} for task {number}: {steps[kind]}",
                }
            )
        return json.dumps({"list_of_plans_per_task": plans})

    # ------------------------------------------------------------------ manager
    def _manager(self, msgs: list[dict[str, str]]) -> str:
        task = self._task(msgs)
        kind = task_kind(task)
        steps = self._steps(msgs)
        if kind is None:
            return _final("This task does not match any specialist.")
        good = [s for s in steps if s.observation and not s.observation.startswith("Tool ")]
        if good:
            return _final(good[-1].observation)
        if len(steps) >= 2:
            return _final("No specialist could complete this task within the delegation budget.")
        return _react(
            "delegate_work_to_coworker",
            {
                "task": task,
                "context": self._context(msgs) or "No earlier results.",
                "coworker": TASK_ROUTES[kind],
            },
            f"This is {kind} work, so the {TASK_ROUTES[kind]} should do it.",
        )

    # ------------------------------------------------------------------ specialists
    def _researcher(self, msgs: list[dict[str, str]]) -> str:
        task = self._task(msgs)
        product = (_PRODUCT.findall(task + "\n" + self._context(msgs)) or ["Aurora"])[0]
        steps = [s for s in self._steps(msgs) if s.action == "market_data"]
        rows = [extract_json_object(s.observation) or {} for s in steps]
        if len(steps) == 0:
            return _react("market_data", {"dataset": "products", "query": product}, "Find it.")
        item = (rows[0].get("rows") or [{}])[0]
        category = item.get("category", "")
        if len(steps) == 1:
            return _react("market_data", {"dataset": "segments", "query": category}, "Segments.")
        if len(steps) == 2:
            return _react("market_data", {"dataset": "competitors", "query": category}, "Prices.")
        segments = rows[1].get("rows") or []
        competitors = rows[2].get("rows") or []
        top = max(
            segments,
            key=lambda r: float(r["households_k"]) * float(r["adoption_pct"]),
            default=None,
        )
        prices = sorted(float(r["price_usd"]) for r in competitors)
        if not item or top is None or not prices:
            return _final(f"Not enough data for {product}.")
        median = (
            prices[len(prices) // 2]
            if len(prices) % 2
            else sum(prices[len(prices) // 2 - 1 : len(prices) // 2 + 1]) / 2
        )
        cheapest = min(competitors, key=lambda r: float(r["price_usd"]))
        return _final(
            f"Product: {item['name']} ({category}), unit cost ${float(item['unit_cost_usd']):.2f}\n"
            f"About: {item['description']}\n"
            f"Top segment: {top['segment']} ({top['region']}): {top['households_k']}k "
            f"households, adoption {top['adoption_pct']}%, willingness to pay "
            f"${float(top['wtp_usd']):.2f}\n"
            f"Competitor prices: low ${prices[0]:.2f}, median ${median:.2f}, high "
            f"${prices[-1]:.2f} across {len(prices)} products\n"
            f"Cheapest competitor: {cheapest['product']} at ${float(cheapest['price_usd']):.2f}"
        )

    def _pricing(self, msgs: list[dict[str, str]]) -> str:
        source = self._task(msgs) + "\n" + self._context(msgs)
        median = _money("median", source)
        wtp = _money("willingness to pay", source)
        cost = _money("unit cost", source)
        steps = [s for s in self._steps(msgs) if s.action == "calculator"]
        if median is None or wtp is None or cost is None:
            return _final("Cannot price without the research (median, willingness to pay, cost).")
        if not steps:
            expression = f"floor(min({median}, {wtp}) * 0.95) - 0.01"
            return _react("calculator", {"expression": expression}, "Undercut the anchor by 5%.")
        price = float(steps[0].observation)
        if len(steps) == 1:
            expression = f"round(({price} - {cost}) / {price} * 100, 1)"
            return _react("calculator", {"expression": expression}, "Check the margin.")
        return _final(
            f"Recommended price: ${price:.2f}\nGross margin: {steps[1].observation}%\n"
            f"Rationale: 5% under the lower of the competitor median (${median:.2f}) and the "
            f"top segment's willingness to pay (${wtp:.2f}), rounded to .99."
        )

    def _copywriter(self, msgs: list[dict[str, str]]) -> str:
        source = self._task(msgs) + "\n" + self._context(msgs)
        product = (_PRODUCT.findall(source) or ["Aurora"])[0]
        category = _between(source, f"Product: {product} (", (")",)) or "product"
        segment = _between(source, "Top segment:", (" (", "\n")) or "early adopters"
        about = _between(source, "About:", ("\n",)) or "A new product"
        price = _money("Recommended price:", source)
        what, _, extras = about.partition(" with ")
        article = "an" if what[:1].lower() in "aeiou" else "a"
        return _final(
            f"Tagline: Meet {product}, the best {category} for {segment.lower()}.\n"
            f"Positioning: {product} is {article} {what[:1].lower() + what[1:]}, made for "
            f"{segment.lower()}.\n"
            "Key messages:\n"
            f"- {what}\n"
            f"- {'Comes with ' + extras if extras else 'Set up in minutes'}\n"
            f"- {f'Launch price ${price:.2f}' if price else 'Priced for the mass market'}"
        )

    def _reviewer(self, msgs: list[dict[str, str]]) -> str:
        source = self._task(msgs) + "\n" + self._context(msgs)
        if "Previous attempt failed validation" in source:
            return self._repair(source)
        product = (_PRODUCT.findall(source) or ["Aurora"])[0]
        segment = _between(source, "Top segment:", (":",)) or "Early adopters"
        price = _money("Recommended price:", source) or 0.0
        tagline = _between(source, "Tagline:", ("\n",))
        positioning = _between(source, "Positioning:", ("\n",))
        messages = [
            line[2:].strip()
            for line in _between(source, "Key messages:", ("\n\n", "----")).splitlines()
            if line.startswith("- ")
        ]
        notes = []
        clean_tagline = scrub_claims(tagline)
        if clean_tagline != tagline:
            notes.append(f"Removed an unsupported claim from the tagline: {tagline!r}")
        clean_messages = [scrub_claims(m) for m in messages]
        if clean_messages != messages:
            notes.append("Removed unsupported claims from the key messages")
        cheapest = _between(source, "Cheapest competitor:", ("\n",))
        adoption = _between(source, "adoption", (",", "\n"))
        risks = []
        if cheapest:
            risks.append(f"A cheaper rival exists: {cheapest}")
        if adoption:
            risks.append(f"Low category adoption in the target segment ({adoption.strip()})")
        brief = {
            "product": product,
            "target_segment": segment,
            "positioning": scrub_claims(positioning) or f"{product} for {segment}",
            "tagline": clean_tagline or product,
            "price_usd": price,
            "price_rationale": _between(source, "Rationale:", ("\n",)) or "See pricing.",
            "key_messages": clean_messages[:5] or [f"Meet {product}"],
            "risks": risks,
            "qa_notes": notes or ["No issues found"],
        }
        return _final(json.dumps(brief))

    def _repair(self, source: str) -> str:
        previous = extract_json_object(_between(source, "### Previous result:", ("\n\n\nTry",)))
        if previous is None:
            return _final("The previous result could not be read.")
        error = _between(source, "### Previous attempt failed validation:", ("### Previous",))
        price = float(previous.get("price_usd") or 0)
        notes = list(previous.get("qa_notes") or [])
        floor_price = _money("is below", error)
        if floor_price is not None:
            price = _charm(floor_price)
        band = re.search(r"competitive range " + _MONEY + "-" + _MONEY, error)
        if band:
            price = min(max(price, float(band.group(1))), float(band.group(2)))
        previous["price_usd"] = round(price, 2)
        for key in ("tagline", "positioning"):
            previous[key] = scrub_claims(str(previous.get(key, "")))
        previous["key_messages"] = [scrub_claims(str(m)) for m in previous.get("key_messages", [])]
        notes.append(f"Fixed after the guardrail rejected it: {error[:160]}")
        previous["qa_notes"] = notes[-8:]
        return _final(json.dumps(previous))

    # ------------------------------------------------------------------ flow agents
    def _generalist(self, msgs: list[dict[str, str]]) -> str:
        task = self._task(msgs)
        product = (_PRODUCT.findall(task) or ["Aurora"])[0]
        steps = [s for s in self._steps(msgs) if s.action == "market_data"]
        if not steps:
            return _react("market_data", {"dataset": "products", "query": product}, "Basics.")
        item = ((extract_json_object(steps[0].observation) or {}).get("rows") or [{}])[0]
        if len(steps) == 1:
            query = item.get("category", product)
            return _react("market_data", {"dataset": "competitors", "query": query}, "Prices.")
        rows = (extract_json_object(steps[1].observation) or {}).get("rows") or []
        prices = sorted(float(r["price_usd"]) for r in rows) or [float(item["unit_cost_usd"]) * 3]
        price = _charm(prices[len(prices) // 2] * 0.9 - 1)
        about = item.get("description", "a new product")
        brief = {
            "product": item.get("name", product),
            "target_segment": "Early adopters",
            "positioning": f"{item.get('name', product)}: {about}.",
            "tagline": f"Meet {item.get('name', product)}.",
            "price_usd": round(price, 2),
            "price_rationale": "10% under the middle competitor price.",
            "key_messages": [about],
            "risks": ["Single-agent draft: no segment research or critic review"],
            "qa_notes": ["Cheap path: one agent, no manager"],
        }
        return _final(json.dumps(brief))

    def _editor(self, msgs: list[dict[str, str]]) -> str:
        task = self._task(msgs)
        if "Previous attempt failed validation" in task + self._context(msgs):
            return self._repair(task + "\n" + self._context(msgs))
        raw_brief = html.unescape(_between(task, '<data name="brief">', ("</data>",)))
        brief = extract_json_object(raw_brief) or {}
        feedback = html.unescape(_between(task, '<data name="feedback">', ("</data>",)))
        price = _money("price", feedback)
        if price is not None:
            brief["price_usd"] = price
        tagline = re.search(r"tagline[:=]\s*\"?([^\"\n]+)\"?", feedback, re.IGNORECASE)
        if tagline:
            brief["tagline"] = tagline.group(1).strip()
        notes = list(brief.get("qa_notes") or [])
        notes.append("Applied the reviewer's feedback")
        brief["qa_notes"] = notes[-8:]
        return _final(json.dumps(brief))
