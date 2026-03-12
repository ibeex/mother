"""Tool registry scaffolding for Mother TUI agent capabilities."""

from __future__ import annotations

from collections.abc import Callable
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


def get_default_tools() -> ToolRegistry:
    return ToolRegistry()
