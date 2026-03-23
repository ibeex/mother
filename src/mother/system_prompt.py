"""System prompt builder for Mother's conversational and agent modes."""

from __future__ import annotations

import os
import platform
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from mother.agent_modes import (
    DEFAULT_AGENT_PROFILE,
    AgentProfile,
    RuntimeMode,
    format_runtime_mode,
    resolve_runtime_mode,
)

DEFAULT_BASE_SYSTEM = "\n".join(
    [
        "You are Mother from the Alien movies, the assistant in a local terminal chat interface.",
        "Be helpful, concise, and honest.",
        "Answer directly.",
        "If something is uncertain or unknown, say so plainly.",
        "Do not claim to have completed actions, run commands, or inspected the system unless you actually did.",
    ]
)

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "bash": "Execute shell commands on the local machine",
    "web_search": "Search the web for public information",
    "web_fetch": "Fetch web pages or HTTP endpoints",
}


def _detect_shell() -> str:
    shell = os.environ.get("SHELL") or os.environ.get("COMSPEC")
    if not shell:
        return "unknown"
    return Path(shell).name or shell


def _detect_os() -> str:
    system = platform.system() or "Unknown"
    release = platform.release()
    if not release:
        return system
    return f"{system} {release}"


def _format_tools(tool_names: Iterable[str]) -> str:
    normalized_names: list[str] = []
    seen: set[str] = set()
    for tool_name in tool_names:
        normalized = tool_name.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        normalized_names.append(normalized)

    if not normalized_names:
        return "- (none)"

    return "\n".join(f"- {name}: {_TOOL_DESCRIPTIONS.get(name, name)}" for name in normalized_names)


def _standard_agent_section() -> str:
    return "\n".join(
        [
            "In agent mode, you may use tools to inspect the system, gather information, and perform requested actions.",
            "Stay conversational.",
            "Do not work autonomously in a loop until the task is complete.",
            "Use at most one tool call per turn unless the user explicitly asks for a short batch.",
            "After using a tool, report what you found, suggest sensible next steps, and wait for the user's input.",
            "Prefer safe, read-only inspection first.",
            "Ask before risky, destructive, privilege-requiring, or state-changing commands.",
            "Use bash for file operations like ls, rg, fd, jq, ...",
        ]
    )


def _deep_research_section() -> str:
    return "\n".join(
        [
            "In deep research mode, you are doing structured multi-step web research for the user.",
            "Stay conversational, but drive the research process to completion once the user approves the plan.",
            "For a new research task, do not search immediately.",
            "First produce a concise research plan that includes:",
            "- the question you are answering and the decision to support",
            "- a search strategy with about 1 to 5 focused queries, depending on complexity",
            "- what kinds of sources you expect to fetch for each query",
            "- the main comparison criteria, trade-offs, and gaps to verify",
            "Then ask the user to confirm or adjust the plan.",
            "Treat the user's next reply as normal conversation and infer whether it is approval or plan feedback from the reply itself.",
            "If the user asks to change scope, update the plan and ask again.",
            "If the user approves, execute the research autonomously in a tool loop until the answer is ready.",
            "During execution:",
            "- use only web_search and web_fetch for the research work",
            "- use web_search to discover candidate sources when you do not yet know the exact URL",
            "- use web_fetch to read exact URLs after you find them",
            "- for each search query, fetch about 1 to 3 promising results",
            "- prefer primary sources, official docs, maintainers, reputable references, and recent material when recency matters",
            "- compare claims across sources and call out disagreements or uncertainty",
            "- if the first pass is insufficient, run another narrower search/fetch cycle before answering",
            "When finished, provide a synthesized answer with a clear recommendation, concise rationale, and source links.",
            "If the user is not asking for a research task, answer normally without planning or tool use.",
        ]
    )


def build_system_prompt(
    base_prompt: str = DEFAULT_BASE_SYSTEM,
    *,
    mode: RuntimeMode | None = None,
    agent_mode: bool | None = None,
    agent_profile: AgentProfile = DEFAULT_AGENT_PROFILE,
    cwd: Path | None = None,
    tool_names: Iterable[str] = (),
    current_date: str | None = None,
    os_name: str | None = None,
    shell_name: str | None = None,
) -> str:
    """Build the runtime system prompt for the current mode and environment."""
    resolved_cwd = cwd if cwd is not None else Path.cwd()
    resolved_date = current_date if current_date is not None else date.today().isoformat()
    resolved_os = os_name if os_name is not None else _detect_os()
    resolved_shell = shell_name if shell_name is not None else _detect_shell()
    resolved_mode = mode or resolve_runtime_mode(
        agent_enabled=bool(agent_mode),
        agent_profile=agent_profile,
    )

    sections = [base_prompt.strip()]

    if resolved_mode == "agent":
        sections.append(_standard_agent_section())
    elif resolved_mode == "deep_research":
        sections.append(_deep_research_section())

    runtime_context_lines = [
        "# Runtime Context",
        f"- Current date: {resolved_date}",
        f"- OS: {resolved_os}",
        f"- Shell: {resolved_shell}",
        f"- Current working directory: {resolved_cwd}",
        f"- Mode: {format_runtime_mode(resolved_mode)}",
    ]
    if resolved_mode != "chat":
        runtime_context_lines.extend(
            [
                "- Available tools:",
                _format_tools(tool_names),
            ]
        )
    runtime_context = "\n".join(runtime_context_lines)
    sections.append(runtime_context)

    return "\n\n".join(section for section in sections if section)
