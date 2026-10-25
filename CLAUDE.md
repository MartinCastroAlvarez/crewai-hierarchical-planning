# CLAUDE.md

Contract between this repository and any AI agent contributing to it.

## Required reading

1. This file. 2. [`README.md`](README.md). 3. [`ARCHITECTURE.md`](ARCHITECTURE.md).
4. [`SECURITY.md`](SECURITY.md).

## Mission

`crewai-hierarchical-planning` shows the hierarchical / supervisor architecture with
planning in current CrewAI (`Process.hierarchical`, `manager_agent` / `manager_llm`,
`planning=True`, delegation, task `context`, `output_pydantic`, guardrails, a Flow with
`@start/@router/@listen`, `@human_feedback` and Pydantic state) as small, correct and
tested code.

## The rules

1. **Local and silent by default.** ollama on loopback is the default; hosted providers
   stay optional adapters. Never weaken `offline.py`, never enable CrewAI memory,
   knowledge, tracing or `share_crew` by default.
2. **Model output, tool input, dataset cells and human feedback are untrusted.** The
   brief always passes `brief_errors` (task guardrail and `finalize`). Human text enters
   prompts only through `flow.data_block`.
3. **The code bounds the hierarchy, not the model.** Only the manager delegates;
   `max_iter`, the delegation-cap hook and `guardrail_max_retries` stay in place.
4. **No code-execution or path tools.** The calculator walks the AST; the dataset tool
   takes a name from an enum. Keep it that way.
5. **Offline tests.** `LaunchFakeLLM` (a CrewAI `BaseLLM`) only; sockets are blocked by
   an autouse fixture. Tests check control flow (delegations, plans, retries, routes),
   not prose.
6. **README numbers come from real runs.** Rerun `python -m hierarchical_crew.bench
   --json docs/benchmark/results.json` and paste its table. Never edit a number by hand.
   Mermaid diagrams come from `--mermaid flow|crew`.
7. Never add `# noqa`, `# nosec` or `# type: ignore` to the package, skip a test, or
   loosen lint/type/coverage config. Fix the code.

## Done means

`poetry run pytest` (coverage >= 85%), `ruff check`, `ruff format --check`, `mypy`
(strict), `bandit` and `pip-audit` all green; `python -m examples.quickstart` runs;
README and CHANGELOG are updated.

## Git

Small commits, imperative subjects, never force-push `main`, never commit secrets or `.env`.
