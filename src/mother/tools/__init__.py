"""Tool registry scaffolding for Mother TUI agent capabilities."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic_ai import Tool

from mother.agent_modes import DEFAULT_AGENT_PROFILE, AgentProfile
from mother.tools.bash_guard import BashGuardDecision


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: list[Tool[None]] = []

    def register(self, tool: Tool[None] | Callable[..., object]) -> None:
        if isinstance(tool, Tool):
            self._tools.append(tool)
            return
        self._tools.append(Tool(tool, takes_ctx=False))

    def tools(self) -> list[Tool[None]]:
        return list(self._tools)

    def is_empty(self) -> bool:
        return len(self._tools) == 0


def get_default_tools(
    tools_enabled: bool = False,
    cwd: Path | None = None,
    ca_bundle_path: str = "",
    agent_profile: AgentProfile = DEFAULT_AGENT_PROFILE,
    request_bash_approval: Callable[[BashGuardDecision], bool] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    if not tools_enabled:
        return registry

    from mother.tools.web_fetch_tool import make_web_fetch_tool
    from mother.tools.web_search_tool import make_web_search_tool

    if agent_profile == "standard":
        from mother.tools.bash_tool import make_bash_tool

        effective_cwd = cwd if cwd is not None else Path.cwd()
        registry.register(make_bash_tool(cwd=effective_cwd, request_approval=request_bash_approval))

    registry.register(make_web_search_tool(ca_bundle_path=ca_bundle_path))
    registry.register(make_web_fetch_tool(ca_bundle_path=ca_bundle_path))
    return registry
