"""Package-wide guards: no dynamic code execution, no silenced findings, data ships."""

from __future__ import annotations

import ast
from pathlib import Path

import hierarchical_crew

PACKAGE = Path(hierarchical_crew.__file__).parent
SOURCES = sorted(PACKAGE.rglob("*.py"))
BANNED_CALLS = {"eval", "exec", "compile", "__import__"}
BANNED_MODULES = {"pickle", "marshal", "shelve", "subprocess", "yaml", "requests", "os.system"}


def test_no_dynamic_execution_or_risky_modules() -> None:
    assert SOURCES
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in BANNED_CALLS, f"{path.name}: {node.func.id}()"
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    assert kw.arg != "shell", f"{path.name}: shell= argument"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in BANNED_MODULES, (
                        f"{path.name}: {alias.name}"
                    )
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in BANNED_MODULES, (
                    f"{path.name}: {node.module}"
                )


def test_no_silenced_findings() -> None:
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for marker in ("# noqa", "# nosec", "# type: ignore"):
            assert marker not in text, f"{path.name} contains {marker!r}"


def test_no_code_execution_tools() -> None:
    from hierarchical_crew.config import Settings
    from hierarchical_crew.crew import build_agents
    from hierarchical_crew.dataset import MarketData
    from hierarchical_crew.llm import build_models

    s = Settings(provider="fake")
    agents = build_agents(s, build_models(s), MarketData.load())
    tools = {t.name for a in agents.values() for t in a.tools or []}
    assert tools == {"market_data", "calculator"}
    assert not any("allow_code_execution" in a.model_fields_set for a in agents.values())


def test_bundled_data_ships_with_the_package() -> None:
    files = sorted(p.name for p in (PACKAGE / "data").iterdir())
    assert files == ["README.md", "competitors.csv", "products.csv", "segments.csv"]
