"""Parsing for direct user shell commands: !command and !!command."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NormalPrompt:
    text: str


@dataclass
class ShellCommand:
    command: str
    include_in_context: bool


@dataclass
class SaveSessionCommand:
    command: str = "/save"


@dataclass
class QuitAppCommand:
    command: str = "/quit"


def should_submit_on_enter(text: str) -> bool:
    """Return whether Enter should submit immediately for this built-in slash command."""
    if "\n" in text:
        return False
    parsed = parse_user_input(text)
    return isinstance(parsed, SaveSessionCommand | QuitAppCommand)


def parse_user_input(
    text: str,
) -> NormalPrompt | SaveSessionCommand | QuitAppCommand | ShellCommand:
    """Parse user input for built-in slash commands and ! / !! shell commands.

    - ``/save`` or ``/export`` → SaveSessionCommand()
    - ``/quit`` or ``/exit``   → QuitAppCommand()
    - ``!!command``            → ShellCommand(..., include_in_context=False)
    - ``!command``             → ShellCommand(..., include_in_context=True)
    - anything else            → NormalPrompt(text)

    A bare ``!`` (with no command) is treated as a NormalPrompt.
    """
    normalized = text.strip().lower()
    if normalized in {"/save", "/export"}:
        return SaveSessionCommand(command=normalized)
    if normalized in {"/quit", "/exit"}:
        return QuitAppCommand(command=normalized)

    if text.startswith("!!"):
        command = text[2:].strip()
        if command:
            return ShellCommand(command=command, include_in_context=False)
    elif text.startswith("!"):
        command = text[1:].strip()
        if command:
            return ShellCommand(command=command, include_in_context=True)
    return NormalPrompt(text=text)
