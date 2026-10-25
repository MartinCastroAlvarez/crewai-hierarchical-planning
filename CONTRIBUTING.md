# Contributing to crewai-hierarchical-planning

## TL;DR

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install poetry
poetry install --all-extras
poetry run pytest
poetry run ruff check hierarchical_crew tests examples && poetry run ruff format --check hierarchical_crew tests examples
poetry run mypy hierarchical_crew tests
poetry run bandit -c pyproject.toml -q -r hierarchical_crew
poetry run pip-audit --ignore-vuln PYSEC-2026-311 --ignore-vuln PYSEC-2026-3813 --ignore-vuln PYSEC-2026-3814 --ignore-vuln PYSEC-2026-3815
poetry run python -m examples.quickstart
```

Open a PR against `main` from a `feat/...`, `fix/...`, `docs/...` or `chore/...` branch.

## What this project will not accept

- Tests that call a real model or touch the network.
- Hosted providers, telemetry, tracing, CrewAI memory or knowledge turned on by default.
- Tools that execute code, take file paths, or reach the network.
- Delegation from specialists, or a loop that `max_iter` and the delegation cap don't bound.
- Hand-edited benchmark numbers or hand-drawn Mermaid diagrams.

## What it does accept

- New specialists or processes (e.g. a consensual process when CrewAI ships it), with
  delegation tests.
- New brief rules, each with a failing and a passing case.
- New feedback providers that implement CrewAI's `request_feedback` protocol.

## Updating the benchmark table

```bash
poetry run python -m hierarchical_crew.bench --json docs/benchmark/results.json
```

Paste the printed table into the README together with the machine, versions and date.

## House rules

1. Tests alongside features; security-relevant changes get a hostile case.
2. README, ARCHITECTURE/SECURITY (when relevant) and CHANGELOG change with the code.
3. No secrets in commits; `.env.example` holds defaults only.
