# Architecture

```
hierarchical_crew/
├── __init__.py         forces the telemetry opt-outs before CrewAI is imported
├── offline.py          OFFLINE_ENV: CREWAI_DISABLE_TELEMETRY, OTEL_SDK_DISABLED, ...
├── config.py           Settings (LAUNCH_CREW_* env), loopback guard for OLLAMA_HOST
├── roles.py            agent roles, task kinds, TASK_ROUTES (which specialist owns what)
├── dataset.py          MarketData: three allowlisted CSVs, sandboxed loader
├── tools.py            MarketDataTool, CalculatorTool (CrewAI BaseTool subclasses)
├── models.py           LaunchBrief (output_pydantic), tolerant JSON extraction
├── guardrails.py       brief_errors (business rules), make_brief_guardrail (task guardrail)
├── instrumentation.py  RunStats + global CrewAI hooks: counters, delegation cap
├── fake.py             LaunchFakeLLM (CrewAI BaseLLM): role-aware scripted model
├── llm.py              build_llm: ollama / anthropic / fake
├── crew.py             agents, manager, tasks, build_crew (3 processes), run_crew
├── flow.py             LaunchFlow (router, @human_feedback, finalize), run_flow
├── diagram.py          Mermaid from Flow.flow_definition() and from a built Crew
├── bench.py            the process comparison
├── cli.py, __main__.py python -m hierarchical_crew
└── data/               the fictional dataset (ships in the wheel)
```

## One run, end to end

1. `run_flow(request, settings)` creates `LaunchFlow` with its models, dataset and a
   feedback source, then calls `kickoff(inputs={"request": ...})`.
2. `classify` cleans the request (no control characters or template braces, at most
   2,000 characters), resolves the product against the dataset, and scores complexity.
3. `choose_path` emits `crew` or `simple`.
4. **crew:** `run_crew` builds a **fresh** `Crew`, because the planner mutates task
   descriptions. It enters `recording(stats)` and runs `kickoff(inputs={product,
   request})`. In planning mode CrewAI first runs its `Task Execution Planner` on
   `planning_llm`. Then each task goes to the manager, which calls
   `delegate_work_to_coworker(task, context, coworker)`. CrewAI runs the coworker's
   `execute_task`, and the manager returns the observation as the task's answer. On the
   last task, CrewAI converts the answer to `LaunchBrief` and runs the guardrail. A
   rejection re-runs the task, through the manager again, with the error and the
   previous result in the prompt.
   **simple:** a one-agent sequential crew with one guardrailed task.
5. `review` returns the draft. `@human_feedback` hands it to `_FlowFeedback`, which
   forwards to the flow's own source (console, scripted or custom).
6. `decide` classifies the reply with fixed regexes, then `apply_feedback` (editor task,
   guardrailed), `finalize` (`brief_errors` again) or `record_rejection` runs.

## Why the global hooks, and why they are safe

CrewAI's `before_llm_call` / `before_tool_call` hooks are process-wide. The package
registers one of each, once. They read a `ContextVar` set by `recording()`, which CrewAI
propagates into flow method threads, and they do nothing when it is unset. So two runs
in different contexts never share counters, and other crews in the same process are not
affected. The delegation cap is a `before_tool_call` hook that returns `False` after
`max_delegations` allowed delegations. That is CrewAI's documented way to block a tool
call.

## Why the fake is a `BaseLLM`

Patching `litellm` would skip CrewAI's own code paths, and CrewAI 1.x doesn't route
ollama through litellm anyway: it uses its native OpenAI-compatible client. Subclassing
`BaseLLM` is the supported extension point. The executor, the ReAct parser, the
delegation tools, the planner's structured output, the converter and the guardrail loop
all run for real. The fake answers in the same ReAct text format a model would
(`supports_function_calling() -> False`) and calls `_invoke_before_llm_call_hooks` like a
native provider, so the counters see its calls.

## Extending

- **A new specialist:** add a role in `roles.py`, an agent in `build_agents`, a task in
  `build_tasks` (its description must start with the prefix `task_kind` knows), a route
  in `TASK_ROUTES`, and a fake handler. Then add tests for its delegation.
- **A new rule:** add it to `brief_errors`, with a parametrized case in
  `tests/test_guardrails.py`. The task guardrail and `finalize` both pick it up.
- **A new feedback channel:** implement `request_feedback(context, flow) -> str`, or raise
  `HumanFeedbackPending` to pause, and pass it as `feedback=` to `run_flow`.
