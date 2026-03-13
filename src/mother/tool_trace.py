"""Formatting helpers for visible tool execution traces in the TUI."""

import json

MAX_OUTPUT_PREVIEW_LINES = 5


def _format_section(title: str, body: str) -> str:
    """Format a titled plain-text section."""
    return f"{title}:\n{body}"


def format_tool_arguments(arguments: dict[str, object]) -> str:
    """Render tool arguments as plain text."""
    lines: list[str] = []

    command = arguments.get("command")
    extras = {key: value for key, value in arguments.items() if key != "command"}

    if isinstance(command, str) and command.strip():
        lines.append(_format_section("Command", command))

    rendered_arguments = extras if lines else arguments
    if rendered_arguments:
        if lines:
            lines.append("")
        lines.append(
            _format_section(
                "Arguments",
                json.dumps(rendered_arguments, indent=2, sort_keys=True, default=repr),
            )
        )

    return "\n".join(lines)


def format_tool_output_preview(output: str, max_lines: int = MAX_OUTPUT_PREVIEW_LINES) -> str:
    """Render only the first ``max_lines`` of tool output."""
    trimmed_output = output.rstrip()
    if not trimmed_output:
        return "(no output)"

    lines = trimmed_output.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)

    remaining = len(lines) - max_lines
    preview_lines = lines[:max_lines]
    preview_lines.append(f"... ({remaining} more lines)")
    return "\n".join(preview_lines)


def format_tool_event(
    tool_name: str,
    arguments: dict[str, object],
    *,
    status: str,
    output: str | None = None,
) -> str:
    """Render a tool lifecycle event as plain text."""
    lines = [f"Tool: {tool_name}", f"Status: {status}"]

    rendered_arguments = format_tool_arguments(arguments)
    if rendered_arguments:
        lines.extend(["", rendered_arguments])

    if output is not None:
        preview_output = format_tool_output_preview(output)
        lines.extend(["", _format_section("Output", preview_output)])

    return "\n".join(lines)
