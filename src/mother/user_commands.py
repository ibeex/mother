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


def parse_user_input(text: str) -> NormalPrompt | ShellCommand:
    """Parse user input for ! or !! prefixes.

    - ``!!command`` → ShellCommand(..., include_in_context=False)
    - ``!command``  → ShellCommand(..., include_in_context=True)
    - anything else → NormalPrompt(text)

    A bare ``!`` (with no command) is treated as a NormalPrompt.
    """
    if text.startswith("!!"):
        command = text[2:].strip()
        if command:
            return ShellCommand(command=command, include_in_context=False)
    elif text.startswith("!"):
        command = text[1:].strip()
        if command:
            return ShellCommand(command=command, include_in_context=True)
    return NormalPrompt(text=text)
