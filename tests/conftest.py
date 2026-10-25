"""Shared fixtures. Every test runs with sockets blocked: any connection attempt fails
the test, so "no network calls" is checked by the whole suite, not one test."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest

from hierarchical_crew.config import Settings
from hierarchical_crew.dataset import MarketData
from hierarchical_crew.fake import LaunchFakeLLM
from hierarchical_crew.llm import Models

COMPLEX = "Plan the launch of Aurora: segments, competitor pricing, price, tagline and risks."
SIMPLE = "Write a tagline for Aurora."


class NetworkBlocked(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    attempts: list[Any] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        attempts.append(args[1:] if len(args) > 1 else args)
        raise NetworkBlocked(f"network access attempted: {args[1:]!r}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    yield attempts
    assert attempts == [], f"tests must not touch the network: {attempts}"


@pytest.fixture(scope="session")
def market() -> MarketData:
    return MarketData.load()


@pytest.fixture
def fake() -> LaunchFakeLLM:
    return LaunchFakeLLM()


@pytest.fixture
def models(fake: LaunchFakeLLM) -> Models:
    return Models(worker=fake, manager=fake)


def settings(**changes: Any) -> Settings:
    return Settings(provider="fake", **changes)


def brief_dict(**changes: Any) -> dict[str, Any]:
    brief: dict[str, Any] = {
        "product": "Aurora",
        "target_segment": "Parents of newborns (US)",
        "positioning": "Aurora is an indoor air-quality monitor for parents of newborns.",
        "tagline": "Meet Aurora, the air-quality monitor for new parents.",
        "price_usd": 131.99,
        "price_rationale": "5% under the competitor median.",
        "key_messages": ["Tracks CO2, PM2.5, VOCs and humidity", "30-day battery"],
        "risks": ["A cheaper rival exists"],
        "qa_notes": [],
    }
    brief.update(changes)
    return brief
