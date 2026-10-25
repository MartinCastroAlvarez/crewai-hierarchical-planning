from __future__ import annotations

import os

import pytest
from crewai.llms.base_llm import BaseLLM

from hierarchical_crew.config import Settings
from hierarchical_crew.config import is_loopback_url
from hierarchical_crew.fake import LaunchFakeLLM
from hierarchical_crew.llm import build_llm
from hierarchical_crew.llm import build_models
from hierarchical_crew.offline import OFFLINE_ENV
from hierarchical_crew.offline import disable_telemetry


def test_telemetry_is_off_after_import() -> None:
    for key, value in OFFLINE_ENV.items():
        assert os.environ[key] == value
    env = {"CREWAI_TRACING_ENABLED": "true"}
    disable_telemetry(env)
    assert env["CREWAI_TRACING_ENABLED"] == "false" and env["OTEL_SDK_DISABLED"] == "true"


def test_crewai_telemetry_reports_disabled() -> None:
    from crewai.telemetry import Telemetry

    assert Telemetry._is_telemetry_disabled()


def test_from_env_defaults_and_overrides() -> None:
    s = Settings.from_env({})
    assert (s.provider, s.model, s.mode, s.manager) == ("ollama", "qwen3:4b", "planning", "agent")
    s = Settings.from_env(
        {
            "LAUNCH_CREW_PROVIDER": "anthropic",
            "LAUNCH_CREW_MODE": "sequential",
            "LAUNCH_CREW_MANAGER": "llm",
            "LAUNCH_CREW_MAX_DELEGATIONS": "3",
            "LAUNCH_CREW_GUARDRAIL_RETRIES": "1",
            "LAUNCH_CREW_FAKE_LATENCY": "0.5",
            "LAUNCH_CREW_MANAGER_MODEL": "claude-haiku-4-5-20251001",
            "LAUNCH_CREW_VERBOSE": "1",
        }
    )
    assert (s.provider, s.model, s.mode, s.manager) == (
        "anthropic",
        "claude-sonnet-5-5",
        "sequential",
        "llm",
    )
    assert (s.max_delegations, s.guardrail_retries, s.fake_latency, s.verbose) == (3, 1, 0.5, True)
    assert s.manager_model == "claude-haiku-4-5-20251001"


@pytest.mark.parametrize(
    "env",
    [
        {"LAUNCH_CREW_PROVIDER": "openai"},
        {"LAUNCH_CREW_MODE": "swarm"},
        {"LAUNCH_CREW_MAX_DELEGATIONS": "0"},
        {"LAUNCH_CREW_FAKE_LATENCY": "-1"},
        {"LAUNCH_CREW_GUARDRAIL_RETRIES": "9"},
        {"OLLAMA_HOST": "http://192.168.1.10:11434"},
    ],
)
def test_bad_settings(env: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        Settings.from_env(env)


def test_direct_settings_validation() -> None:
    with pytest.raises(ValueError):
        Settings(provider="openai")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Settings(manager="boss")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Settings(provider="fake", max_iter=0)
    assert Settings(provider="fake").with_changes(mode="sequential").mode == "sequential"


def test_remote_ollama_needs_opt_in() -> None:
    env = {"OLLAMA_HOST": "gpu-box:11434", "LAUNCH_CREW_ALLOW_REMOTE_OLLAMA": "1"}
    assert Settings.from_env(env).ollama_host == "gpu-box:11434"


@pytest.mark.parametrize(
    ("url", "loopback"),
    [
        ("http://127.0.0.1:11434", True),
        ("localhost:11434", True),
        ("http://[::1]:11434", True),
        ("http://10.0.0.1", False),
        ("https://ollama.example.com", False),
    ],
)
def test_is_loopback_url(url: str, loopback: bool) -> None:
    assert is_loopback_url(url) is loopback


def test_build_ollama_llm_without_network() -> None:
    llm = build_llm(Settings(model="llama3.2"))
    assert isinstance(llm, BaseLLM)
    assert llm.model == "llama3.2"
    assert str(llm.base_url).startswith("http://127.0.0.1:11434")


def test_build_anthropic_llm_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    llm = build_llm(Settings(provider="anthropic", model="claude-sonnet-5-5"))
    assert isinstance(llm, BaseLLM) and "claude-sonnet-5-5" in llm.model


def test_fake_models_share_one_instance() -> None:
    models = build_models(Settings(provider="fake"))
    assert models.worker is models.manager is models.planner
    assert isinstance(models.worker, LaunchFakeLLM)
    split = build_models(Settings(provider="fake", manager_model="other"))
    assert split.worker is not split.manager and split.manager.model == "other"
