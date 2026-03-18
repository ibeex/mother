"""Parsing for direct user shell commands: !command and !!command."""

from __future__ import annotations

from dataclasses import dataclass

_MODELS_COMMAND = "/models"


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


@dataclass
class ModelsCommand:
    command: str = _MODELS_COMMAND
    query: str | None = None


def current_model_query(text: str) -> str | None:
    """Return the active ``/models`` query for inline model completion."""
    if not text or "\n" in text:
        return None

    candidate = text.lstrip()
    normalized = candidate.lower()
    if normalized == _MODELS_COMMAND:
        return None
    if not normalized.startswith(f"{_MODELS_COMMAND} "):
        return None
    return candidate[len(_MODELS_COMMAND) :].strip()


def should_expand_models_query(text: str) -> bool:
    """Return whether Tab or a typed character should expand ``/models`` to ``/models ``."""
    if not text or "\n" in text:
        return False
    return text.lstrip().lower() == _MODELS_COMMAND


def should_submit_on_enter(text: str) -> bool:
    """Return whether Enter should submit immediately for this built-in slash command."""
    if "\n" in text:
        return False
    parsed = parse_user_input(text)
    return isinstance(parsed, SaveSessionCommand | QuitAppCommand | ModelsCommand)


def parse_user_input(
    text: str,
) -> NormalPrompt | SaveSessionCommand | QuitAppCommand | ModelsCommand | ShellCommand:
    """Parse user input for built-in slash commands and ! / !! shell commands.

    - ``/save`` or ``/export``   → SaveSessionCommand()
    - ``/quit`` or ``/exit``     → QuitAppCommand()
    - ``/models``                → ModelsCommand()
    - ``/models query``          → ModelsCommand(query=...)
    - ``!!command``              → ShellCommand(..., include_in_context=False)
    - ``!command``               → ShellCommand(..., include_in_context=True)
    - anything else              → NormalPrompt(text)

    A bare ``!`` (with no command) is treated as a NormalPrompt.
    """
    candidate = text.strip()
    normalized = candidate.lower()
    if normalized in {"/save", "/export"}:
        return SaveSessionCommand(command=normalized)
    if normalized in {"/quit", "/exit"}:
        return QuitAppCommand(command=normalized)
    if normalized == _MODELS_COMMAND:
        return ModelsCommand()
    if normalized.startswith(f"{_MODELS_COMMAND} "):
        return ModelsCommand(query=candidate[len(_MODELS_COMMAND) :].strip())

    if text.startswith("!!"):
        command = text[2:].strip()
        if command:
            return ShellCommand(command=command, include_in_context=False)
    elif text.startswith("!"):
        command = text[1:].strip()
        if command:
            return ShellCommand(command=command, include_in_context=True)
    return NormalPrompt(text=text)
