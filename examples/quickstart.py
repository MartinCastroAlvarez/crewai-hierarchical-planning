"""Offline tour: the three crew processes, then the Flow's two routes and a revision.

    python -m examples.quickstart

Uses the fake model, so it needs no ollama, no key and no network. Swap
``provider="fake"`` for ``provider="ollama"`` (and ``ollama pull qwen3:4b``) to run the
same code on a local model.
"""

from __future__ import annotations

from hierarchical_crew.config import Settings
from hierarchical_crew.crew import run_crew
from hierarchical_crew.flow import ScriptedFeedback
from hierarchical_crew.flow import run_flow

REQUEST = "Plan the launch of Aurora: segments, competitor pricing, price, tagline and risks."


def main() -> None:
    for mode in ("sequential", "hierarchical", "planning"):
        result = run_crew(REQUEST, Settings(provider="fake", mode=mode))
        s = result.stats
        print(
            f"{mode:>12}: ${result.brief.price_usd:.2f} '{result.brief.tagline}' "
            f"llm_calls={s.total_llm_calls} delegated_to={s.delegated_to()}"
        )
        assert result.brief.price_usd == 131.99

    complex_flow = run_flow(REQUEST, Settings(provider="fake"), feedback=ScriptedFeedback())
    print(f"flow complex -> route={complex_flow.state.route} status={complex_flow.state.status}")
    assert complex_flow.state.route == "crew"

    revised = run_flow(
        "Write a tagline for Tidal.",
        Settings(provider="fake"),
        feedback=ScriptedFeedback(["price $44.99"]),
    )
    final = revised.state.final
    assert final is not None and final.price_usd == 44.99
    print(
        f"flow simple  -> route={revised.state.route} decision={revised.state.decision} "
        f"status={revised.state.status} price=${final.price_usd:.2f} "
        f"llm_calls={revised.state.stats['llm_calls']}"
    )


if __name__ == "__main__":
    main()
