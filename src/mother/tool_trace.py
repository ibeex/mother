"""Formatting helpers for visible tool execution traces in the TUI."""

import json


def _format_section(title: str, body: str) -> str:
    """Format a titled plain-text section."""
    return f"{title}:\n{body}"


def _is_empty_argument(key: str, value: object) -> bool:
    """Return whether a tool argument is effectively empty for display."""
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return True
        if key == "headers_json" and normalized == "{}":
            return True
        return False
    if isinstance(value, dict | list | tuple | set):
        return not value
    return False


def _filter_tool_arguments(arguments: dict[str, object]) -> dict[str, object]:
    """Drop empty values from rendered tool arguments."""
    return {key: value for key, value in arguments.items() if not _is_empty_argument(key, value)}


def _format_argument_body(arguments: dict[str, object]) -> str:
    """Render argument mappings without the outermost braces."""
    rendered = json.dumps(arguments, indent=2, sort_keys=True, default=repr)
    lines = rendered.splitlines()
    if len(lines) >= 2 and lines[0] == "{" and lines[-1] == "}":
        return "\n".join(lines[1:-1])
    return rendered


def format_tool_arguments(arguments: dict[str, object]) -> str:
    """Render tool arguments as plain text."""
    lines: list[str] = []

    filtered_arguments = _filter_tool_arguments(arguments)
    command = filtered_arguments.get("command")
    extras = {key: value for key, value in filtered_arguments.items() if key != "command"}

    if isinstance(command, str) and command.strip():
        lines.append(_format_section("Command", command))

    rendered_arguments = extras if lines else filtered_arguments
    if rendered_arguments:
        if lines:
            lines.append("")
        lines.append(_format_section("Arguments", _format_argument_body(rendered_arguments)))

    return "\n".join(lines)


def format_tool_output(output: str) -> str:
    """Render full tool output, normalizing empty results for display."""
    trimmed_output = output.rstrip()
    if not trimmed_output:
        return "(no output)"
    return trimmed_output


def format_tool_limit_recovery(
    *,
    tool_call_limit: int | None,
    mode: str,
    profile: str,
) -> str:
    """Render a visible trace entry for text-only recovery after a tool-limit hit."""
    profile_label = profile.replace("_", " ")
    lines = [
        "Recovery: tool-call limit reached",
        f"Mode: {mode}",
        f"Profile: {profile_label}",
    ]
    if tool_call_limit is not None:
        lines.append(f"Tool-call limit: {tool_call_limit}")
    lines.extend(
        [
            "",
            "Status: finishing this turn with a text-only reply using completed tool results",
            "Next step: ask me to continue in another turn if you want the next inspection step",
        ]
    )
    return "\n".join(lines)


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
        rendered_output = format_tool_output(output)
        lines.extend(["", _format_section("Output", rendered_output)])

    return "\n".join(lines)
