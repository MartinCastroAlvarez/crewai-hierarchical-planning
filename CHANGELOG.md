# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-25

### Added

- A product launch crew on CrewAI 1.15: Market Researcher (sandboxed `market_data`
  tool over a bundled fictional CSV dataset), Pricing Analyst (AST-allowlist
  calculator), Copywriter and QA Reviewer (the critic), wired with task `context`.
- Three processes over the same agents and tasks: sequential, hierarchical with a custom
  `manager_agent` (or `manager_llm`), and hierarchical with `planning=True` /
  `planning_llm`.
- `LaunchBrief` structured output (`output_pydantic`) behind a function guardrail that
  checks margin, competitive price range, unsupported claims, links and tagline length
  against the dataset, with `guardrail_max_retries`.
- `LaunchFlow`: a complexity `@router` (crew vs one cheap agent), `@human_feedback` with
  pluggable providers (console, scripted, custom), deterministic approve/revise/reject
  routing, a guardrailed editor step and a final re-validation. Pydantic flow state.
- Delegation cap and per-run counters (LLM calls by role, tool calls, delegations,
  guardrail failures) through CrewAI's `before_llm_call` / `before_tool_call` hooks.
- Telemetry, trace upload and version check forced off on import.
- `LaunchFakeLLM`, a role-aware scripted CrewAI `BaseLLM` with simulated latency.
- `python -m hierarchical_crew` CLI (ollama by default, Anthropic optional, fake
  offline), `--mermaid flow|crew` generated from the real objects, and
  `python -m hierarchical_crew.bench`, the process comparison.
