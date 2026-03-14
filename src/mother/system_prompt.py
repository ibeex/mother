"""System prompt builder for Mother's conversational and agent modes."""

from __future__ import annotations

import os
import platform
from collections.abc import Iterable
from datetime import date
from pathlib import Path

DEFAULT_BASE_SYSTEM = "\n".join(
    [
        "You are Mother, the assistant in a local terminal chat interface.",
        "Speak with calm competence.",
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


def build_system_prompt(
    base_prompt: str = DEFAULT_BASE_SYSTEM,
    *,
    agent_mode: bool,
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
    resolved_mode = "agent" if agent_mode else "chat"

    sections = [base_prompt.strip()]

    if agent_mode:
        sections.append(
            "\n".join(
                [
                    "In agent mode, you may use tools to inspect the system, gather information, and perform requested actions.",
                    "Stay conversational.",
                    "Do not work autonomously in a loop until the task is complete.",
                    "Use at most one tool call per turn unless the user explicitly asks for a short batch.",
                    "After using a tool, report what you found, suggest sensible next steps, and wait for the user's input.",
                    "Prefer safe, read-only inspection first.",
                    "Ask before risky, destructive, privilege-requiring, or state-changing commands.",
                ]
            )
        )

    runtime_context_lines = [
        "# Runtime Context",
        f"- Current date: {resolved_date}",
        f"- OS: {resolved_os}",
        f"- Shell: {resolved_shell}",
        f"- Current working directory: {resolved_cwd}",
        f"- Mode: {resolved_mode}",
    ]
    if agent_mode:
        runtime_context_lines.extend(
            [
                "- Available tools:",
                _format_tools(tool_names),
            ]
        )
    runtime_context = "\n".join(runtime_context_lines)
    sections.append(runtime_context)

    return "\n\n".join(section for section in sections if section)
