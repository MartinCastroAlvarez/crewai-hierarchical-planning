"""Settings, read from ``LAUNCH_CREW_*`` environment variables or passed in directly."""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from typing import Literal
from typing import cast
from urllib.parse import urlsplit

__all__ = [
    "DEFAULT_MODELS",
    "MANAGERS",
    "MODES",
    "PROVIDERS",
    "Manager",
    "Mode",
    "Provider",
    "Settings",
    "is_loopback_url",
]

Provider = Literal["ollama", "anthropic", "fake"]
PROVIDERS: tuple[Provider, ...] = ("ollama", "anthropic", "fake")
Mode = Literal["sequential", "hierarchical", "planning"]
MODES: tuple[Mode, ...] = ("sequential", "hierarchical", "planning")
Manager = Literal["agent", "llm"]
MANAGERS: tuple[Manager, ...] = ("agent", "llm")

DEFAULT_MODELS: dict[Provider, str] = {
    "ollama": "qwen3:4b",
    "anthropic": "claude-sonnet-5-5",
    "fake": "fake-launch",
}


def is_loopback_url(url: str) -> bool:
    """True when ``url`` (or a bare ``host[:port]``) points at this machine."""
    if "://" not in url:
        url = f"http://{url}"
    host = (urlsplit(url).hostname or "").strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _int(env: Mapping[str, str], key: str, default: int, *, minimum: int) -> int:
    raw = env.get(key, "")
    value = default if raw == "" else int(raw)
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value}")
    return value


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key, "")
    value = default if raw == "" else float(raw)
    if value < 0:
        raise ValueError(f"{key} must be >= 0, got {value}")
    return value


def _choice(env: Mapping[str, str], key: str, default: str, choices: tuple[str, ...]) -> str:
    value = (env.get(key) or default).strip().lower()
    if value not in choices:
        raise ValueError(f"{key} must be one of {choices}, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the crew, the flow and the CLI can be tuned with."""

    provider: Provider = "ollama"
    model: str = DEFAULT_MODELS["ollama"]
    manager_model: str | None = None
    """Model for the manager and the planner; ``None`` = ``model``."""
    ollama_host: str = "http://127.0.0.1:11434"
    allow_remote_ollama: bool = False
    temperature: float = 0.0
    mode: Mode = "planning"
    """``sequential`` (no manager), ``hierarchical`` or ``planning`` (hierarchical + planner)."""
    manager: Manager = "agent"
    """``agent``: a custom ``manager_agent``; ``llm``: CrewAI's own manager from ``manager_llm``."""
    max_delegations: int = 8
    """Delegations (and questions) the manager may make in one crew run."""
    max_iter: int = 6
    """CrewAI ``max_iter`` per agent and task: tool calls before a forced final answer."""
    guardrail_retries: int = 2
    """Re-runs of the final task when its guardrail rejects the brief."""
    complexity_threshold: int = 3
    """The flow sends requests scoring at least this much to the crew."""
    data_dir: str | None = None
    """A folder with ``products.csv``/``segments.csv``/``competitors.csv``; default: bundled."""
    fake_latency: float = 0.0
    """Seconds per model call for the ``fake`` provider (benchmarks)."""
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.manager not in MANAGERS:
            raise ValueError(f"manager must be one of {MANAGERS}")
        for name in ("max_delegations", "max_iter"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.guardrail_retries < 0 or self.guardrail_retries > 5:
            raise ValueError("guardrail_retries must be between 0 and 5")
        if self.fake_latency < 0 or self.temperature < 0:
            raise ValueError("fake_latency and temperature must be >= 0")
        if (
            self.provider == "ollama"
            and not self.allow_remote_ollama
            and not is_loopback_url(self.ollama_host)
        ):
            raise ValueError(
                f"OLLAMA_HOST={self.ollama_host!r} is not loopback; set "
                "LAUNCH_CREW_ALLOW_REMOTE_OLLAMA=1 to send prompts to another host"
            )

    def with_changes(self, **changes: Any) -> Settings:
        """A copy with some fields replaced (validated again)."""
        return replace(self, **changes)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        provider = cast(Provider, _choice(e, "LAUNCH_CREW_PROVIDER", "ollama", PROVIDERS))
        mode = cast(Mode, _choice(e, "LAUNCH_CREW_MODE", "planning", MODES))
        manager = cast(Manager, _choice(e, "LAUNCH_CREW_MANAGER", "agent", MANAGERS))
        return cls(
            provider=provider,
            model=e.get("LAUNCH_CREW_MODEL") or DEFAULT_MODELS[provider],
            manager_model=e.get("LAUNCH_CREW_MANAGER_MODEL") or None,
            ollama_host=e.get("OLLAMA_HOST") or "http://127.0.0.1:11434",
            allow_remote_ollama=e.get("LAUNCH_CREW_ALLOW_REMOTE_OLLAMA", "") == "1",
            temperature=_float(e, "LAUNCH_CREW_TEMPERATURE", 0.0),
            mode=mode,
            manager=manager,
            max_delegations=_int(e, "LAUNCH_CREW_MAX_DELEGATIONS", 8, minimum=1),
            max_iter=_int(e, "LAUNCH_CREW_MAX_ITER", 6, minimum=1),
            guardrail_retries=_int(e, "LAUNCH_CREW_GUARDRAIL_RETRIES", 2, minimum=0),
            complexity_threshold=_int(e, "LAUNCH_CREW_COMPLEXITY_THRESHOLD", 3, minimum=0),
            data_dir=e.get("LAUNCH_CREW_DATA_DIR") or None,
            fake_latency=_float(e, "LAUNCH_CREW_FAKE_LATENCY", 0.0),
            verbose=e.get("LAUNCH_CREW_VERBOSE", "") == "1",
        )
