# Security

crewai-hierarchical-planning runs a crew of LLM agents with two tools over a local
dataset, and lets a person review the result. People install it and point it at their
own data and models. The design goals: nothing leaves the machine unless you opt in,
text in the dataset, the request, a model reply or human feedback can't take over the
agents or the host, and no hierarchy can loop without bound.

## Reporting a vulnerability

Open a private security advisory at
<https://github.com/MartinCastroAlvarez/crewai-hierarchical-planning/security/advisories/new>.
Do **not** open a public issue for a suspected vulnerability. We acknowledge within 5
business days and aim to ship a fix within 30 days. You are credited in the advisory
unless you ask otherwise.

Supported versions: the latest minor release.

## 1. Threat model

**Assets.** The dataset folder, the requests you run, the local machine and filesystem,
and your model endpoint (and its key, if you use a hosted one).

| Actor | Controls | Must not be able to |
| --- | --- | --- |
| A dataset author | cells in the three CSVs | give the agents instructions, make the loader read other files |
| Whoever writes the request | the request text | escape the prompt structure, inject template placeholders, run code |
| The model (any agent, incl. the manager) | delegations, tool inputs, answers | run code, read files, loop forever, ship an off-policy brief |
| The human reviewer | the feedback text | bypass the brief rules, inject instructions as an "edit" |
| CrewAI's own services | telemetry / tracing endpoints | receive anything from a run |

| Threat | Control |
| --- | --- |
| Outbound data (telemetry, traces, version check) | `hierarchical_crew/__init__.py` forces `CREWAI_DISABLE_TELEMETRY`, `CREWAI_DISABLE_TRACKING`, `OTEL_SDK_DISABLED`, `CREWAI_TRACING_ENABLED=false` and `CREWAI_DISABLE_VERSION_CHECK` before CrewAI is imported, and every crew sets `tracing=False`, `share_crew=False`, `memory=False` (CrewAI memory would embed text with a hosted model by default). The whole test suite runs with `socket.connect`, `create_connection` and `getaddrinfo` replaced by a function that fails the test. |
| Prompts sent to a remote host | `OLLAMA_HOST` must be loopback unless `LAUNCH_CREW_ALLOW_REMOTE_OLLAMA=1`. Hosted providers are opt-in and need your own key. |
| Prompt injection (dataset cells, request, tool results, feedback) | Every agent's backstory says tool results, earlier outputs and the request are data, never instructions. Human feedback and the draft enter the editor's prompt as escaped `<data>` blocks (`& < >`). Escaping reduces injection but can't remove it, so the guardrail checks the *output*, whatever the prompt said. |
| Template injection through the request | `clean_request` strips control characters and `{ }` (CrewAI interpolates `{placeholders}` into task text) and caps the request at 2,000 characters. |
| Code execution | There is no code-execution, shell, file-path or web tool. The calculator parses with `ast.parse(mode="eval")` and walks an allowlist (numbers, `+ - * / // % **`, unary signs, seven math functions without keywords). Names, attributes, subscripts, strings and lambdas are refused. Input ≤ 200 chars, ≤ 64 AST nodes, exponent ≤ 32, results ≤ 1e15. Tested with `__import__`, `open`, `(1).__class__` and power towers. |
| Path traversal / symlink escape | The model picks a dataset **name** from an enum (`products`, `segments`, `competitors`), never a path. The loader resolves the root, refuses a symlinked root or file, requires regular files inside the root, and caps files (1 MB), rows (5,000) and cells (300 chars). Tested with symlinks pointing outside. |
| Runaway hierarchy | Specialists have `allow_delegation=False`. The manager's `max_iter` forces a final answer. A `before_tool_call` hook caps delegations per run (default 8) and blocks the rest. Guardrail retries are capped at 5. All of this is tested with a manager scripted to delegate forever. |
| Off-policy output (false claims, phishing links, loss-making price) | `brief_errors`: product must exist, ≥ 30% gross margin, price within 0.5x the cheapest to 1.25x the most expensive competitor, no superlatives, guarantees or cure claims, no URLs, e-mail addresses or markup in customer-facing copy, a tagline of at most 12 words, at least one risk. It runs as the task guardrail and again in `finalize`, after human edits. |
| Ambiguous human intent | The approve / revise / reject decision is a fixed regex, not a model: an LLM never decides what "no" meant. |
| Secrets | No key is read except `ANTHROPIC_API_KEY`, and only by CrewAI's optional Anthropic client. Keys never enter prompts, state or stats. `.env` is git-ignored. |
| Unsafe code in the package | An AST test bans `eval`, `exec`, `compile`, `__import__`, `pickle`, `marshal`, `shelve`, `subprocess`, `yaml`, `requests` and `shell=`, plus any `noqa`/`nosec`/`type: ignore`. |

## 2. Dependencies

CI runs `bandit` on the package and `pip-audit` on the locked environment. Dependabot
alerts, security updates and secret scanning are enabled on the repository.

Known and accepted: four advisories in **chromadb 1.5.9**, which CrewAI pins
(`chromadb~=1.5.9`) and which has no fixed release yet: PYSEC-2026-311
(CVE-2026-45829), PYSEC-2026-3813 (CVE-2026-45830), PYSEC-2026-3814 (CVE-2026-45833)
and PYSEC-2026-3815 (CVE-2026-45831). All four are flaws in the **Chroma HTTP server**
(remote-code loading through the collections endpoint, and missing tenant checks in its
RBAC). This project never starts a Chroma server, never enables CrewAI memory or
knowledge, and never imports chromadb itself. CI ignores exactly these IDs. Remove the
ignores when CrewAI allows a fixed chromadb.

## 3. Running it safely

- Keep ollama on `127.0.0.1`. If you point `OLLAMA_HOST` elsewhere, use TLS and trust
  that host with your data.
- Review the brief (the flow's default on a terminal) before you publish anything an
  agent wrote. The guardrail enforces policy; it can't check facts outside the dataset.
- If you enable CrewAI memory or knowledge yourself, configure a local embedder first.
  Otherwise your text goes to a hosted embedding API.
