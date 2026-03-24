"""Structured records for direct user bash executions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class BashExecution:
    command: str
    output: str
    exit_code: int | None
    timestamp: datetime
    exclude_from_context: bool = False
    truncated: bool = False
    full_output_path: str | None = None


def _normalized_output(output: str) -> str:
    """Return shell output in a readable form, even when it is empty."""
    normalized = output.rstrip()
    if normalized:
        return normalized
    return "(no output)"


def format_for_context(execution: BashExecution) -> str:
    """Convert a BashExecution into an LLM-friendly text block."""
    lines = [
        "Shell command:",
        "```sh",
        execution.command,
        "```",
        "",
        "Output:",
        "```",
        _normalized_output(execution.output),
        "```",
    ]
    if execution.exit_code not in (0, None):
        lines.append(f"Command exited with code {execution.exit_code}")
    if execution.truncated and execution.full_output_path:
        lines.append(f"[Output truncated. Full output: {execution.full_output_path}]")
    return "\n".join(lines)


def format_for_display(execution: BashExecution) -> str:
    """Convert a BashExecution into a plain-text block for the shell output widget."""
    lines = ["Command:", execution.command, "", "Output:", _normalized_output(execution.output)]
    if execution.exit_code not in (0, None):
        lines.extend(["", f"Exit code: {execution.exit_code}"])
    if execution.truncated and execution.full_output_path:
        lines.extend(["", f"Full output: {execution.full_output_path}"])
    return "\n".join(lines)
