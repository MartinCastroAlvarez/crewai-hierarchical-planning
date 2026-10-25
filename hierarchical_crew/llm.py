"""Model factory: local ollama by default, Anthropic as an opt-in, a fake for offline runs."""

from __future__ import annotations

from dataclasses import dataclass

from crewai import LLM
from crewai.llms.base_llm import BaseLLM

from hierarchical_crew.config import Settings
from hierarchical_crew.fake import LaunchFakeLLM

__all__ = ["Models", "build_llm", "build_models"]


@dataclass(frozen=True, slots=True)
class Models:
    """The two model slots of the crew: workers, and manager + planner."""

    worker: BaseLLM
    manager: BaseLLM

    @property
    def planner(self) -> BaseLLM:
        return self.manager


def build_llm(settings: Settings, model: str | None = None) -> BaseLLM:
    """One CrewAI model for ``settings.provider``.

    * ``ollama``: ``LLM(model="ollama/<model>")``, CrewAI's native OpenAI-compatible
      client pointed at ``OLLAMA_HOST`` (loopback unless explicitly allowed).
    * ``anthropic``: ``LLM(model="anthropic/<model>")``; needs ``crewai[anthropic]``
      and your own ``ANTHROPIC_API_KEY``. Never used by this project's tests or CI.
    * ``fake``: :class:`~hierarchical_crew.fake.LaunchFakeLLM`.
    """
    name = model or settings.model
    if settings.provider == "fake":
        return LaunchFakeLLM(model=name, latency=settings.fake_latency)
    if settings.provider == "anthropic":
        return LLM(model=f"anthropic/{name}", temperature=settings.temperature)
    return LLM(
        model=f"ollama/{name}",
        base_url=settings.ollama_host,
        temperature=settings.temperature,
    )


def build_models(settings: Settings) -> Models:
    """Worker and manager models. The fake shares one instance so tests see every call."""
    worker = build_llm(settings)
    if settings.provider == "fake" and settings.manager_model is None:
        return Models(worker=worker, manager=worker)
    return Models(worker=worker, manager=build_llm(settings, settings.manager_model))
