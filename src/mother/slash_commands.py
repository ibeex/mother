"""Slash command metadata and matching helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlashCommand:
    """A built-in slash command shown in autocomplete."""

    command: str
    help: str
    hint: str | None = None


SLASH_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/save", "Save the current session to markdown"),
    SlashCommand("/quit", "Quit Mother"),
    SlashCommand("/exit", "Quit Mother"),
    SlashCommand("/models", "Browse and switch models"),
)


def current_slash_query(text: str) -> str | None:
    """Return the active slash-command token, if the prompt is completing one."""
    if not text:
        return None
    if "\n" in text:
        return None

    candidate = text.lstrip()
    if not candidate.startswith("/"):
        return None
    if any(character.isspace() for character in candidate[1:]):
        return None
    return candidate


def filter_slash_commands(
    commands: list[SlashCommand] | tuple[SlashCommand, ...],
    query: str,
) -> list[SlashCommand]:
    """Return slash commands whose command name matches the current query."""
    normalized = query.strip().lstrip("/").lower()
    if not normalized:
        return list(commands)

    prefix_matches = [
        command
        for command in commands
        if command.command.lower().lstrip("/").startswith(normalized)
    ]
    if prefix_matches:
        return prefix_matches

    return [command for command in commands if normalized in command.command.lower()]
