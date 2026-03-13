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


def format_for_context(execution: BashExecution) -> str:
    """Convert a BashExecution into an LLM-friendly text block."""
    lines: list[str] = [f"Ran `{execution.command}`"]
    lines.append("```")
    lines.append(execution.output.rstrip())
    lines.append("```")
    if execution.exit_code not in (0, None):
        lines.append(f"Command exited with code {execution.exit_code}")
    if execution.truncated and execution.full_output_path:
        lines.append(f"[Output truncated. Full output: {execution.full_output_path}]")
    return "\n".join(lines)
