"""Tool registry scaffolding for Mother TUI agent capabilities."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import llm


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: list[llm.Tool | type[llm.Toolbox] | Callable[..., Any]] = []

    def register(self, tool: llm.Tool | type[llm.Toolbox] | Callable[..., Any]) -> None:
        self._tools.append(tool)

    def tools(self) -> list[llm.Tool | type[llm.Toolbox] | Callable[..., Any]]:
        return list(self._tools)

    def is_empty(self) -> bool:
        return len(self._tools) == 0


def get_default_tools(
    tools_enabled: bool = False,
    allowlist: frozenset[str] | None = None,
    cwd: Path | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    if tools_enabled:
        from mother.tools.bash_tool import DEFAULT_ALLOWLIST, make_bash_tool

        effective_allowlist = allowlist if allowlist is not None else DEFAULT_ALLOWLIST
        effective_cwd = cwd if cwd is not None else Path.cwd()
        registry.register(make_bash_tool(effective_allowlist, effective_cwd))
    return registry
