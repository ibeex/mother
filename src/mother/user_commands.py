"""Parsing for direct user shell commands: !command and !!command."""

from __future__ import annotations

from dataclasses import dataclass

from mother.slash_commands import current_slash_argument_query, should_expand_slash_argument

_AGENT_COMMAND = "/agent"
_MODELS_COMMAND = "/models"
_REASONING_COMMAND = "/reasoning"
_COUNCIL_COMMAND = "/council"


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
class AgentModeCommand:
    command: str = _AGENT_COMMAND
    mode: str | None = None


@dataclass
class ModelsCommand:
    command: str = _MODELS_COMMAND
    query: str | None = None


@dataclass
class ReasoningCommand:
    command: str = _REASONING_COMMAND
    effort: str | None = None


@dataclass
class CouncilCommand:
    command: str = _COUNCIL_COMMAND
    prompt: str | None = None


def current_model_query(text: str) -> str | None:
    """Return the active ``/models`` query for inline model completion."""
    query = current_slash_argument_query(text)
    if query is None or query.command != _MODELS_COMMAND:
        return None
    return query.query


def current_reasoning_query(text: str) -> str | None:
    """Return the active ``/reasoning`` query for inline completion."""
    query = current_slash_argument_query(text)
    if query is None or query.command != _REASONING_COMMAND:
        return None
    return query.query


def should_expand_models_query(text: str) -> bool:
    """Return whether Tab or a typed character should expand ``/models`` to ``/models ``."""
    return text.lstrip().lower() == _MODELS_COMMAND and should_expand_slash_argument(text)


def should_expand_reasoning_query(text: str) -> bool:
    """Return whether Tab or a typed character should expand ``/reasoning`` to ``/reasoning ``."""
    return text.lstrip().lower() == _REASONING_COMMAND and should_expand_slash_argument(text)


def should_submit_on_enter(text: str) -> bool:
    """Return whether Enter should submit immediately for this built-in slash command."""
    if "\n" in text:
        return False
    parsed = parse_user_input(text)
    if isinstance(parsed, CouncilCommand):
        return parsed.prompt is not None
    return isinstance(
        parsed,
        SaveSessionCommand | QuitAppCommand | AgentModeCommand | ModelsCommand | ReasoningCommand,
    )


def is_council_multiline_input(text: str) -> bool:
    """Return whether the prompt is composing a multiline ``/council`` question."""
    if "\n" not in text:
        return False
    candidate = text.lstrip()
    normalized = candidate.lower()
    if not normalized.startswith(_COUNCIL_COMMAND):
        return False
    if len(candidate) == len(_COUNCIL_COMMAND):
        return False
    return candidate[len(_COUNCIL_COMMAND)].isspace()


def parse_user_input(
    text: str,
) -> (
    NormalPrompt
    | SaveSessionCommand
    | QuitAppCommand
    | AgentModeCommand
    | ModelsCommand
    | ReasoningCommand
    | CouncilCommand
    | ShellCommand
):
    """Parse user input for built-in slash commands and ! / !! shell commands.

    - ``/save`` or ``/export``   → SaveSessionCommand()
    - ``/quit`` or ``/exit``     → QuitAppCommand()
    - ``/agent``                 → AgentModeCommand()
    - ``/agent standard``        → AgentModeCommand(mode="standard")
    - ``/agent conversational``  → AgentModeCommand(mode="conversational")
    - ``/agent deep research``   → AgentModeCommand(mode="deep research")
    - ``/models``                → ModelsCommand()
    - ``/models query``          → ModelsCommand(query=...)
    - ``/reasoning``             → ReasoningCommand()
    - ``/reasoning value``       → ReasoningCommand(effort=...)
    - ``/council``               → CouncilCommand()
    - ``/council question``      → CouncilCommand(prompt=...)
    - ``/council\nquestion``     → CouncilCommand(prompt=...)
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
    if normalized == _AGENT_COMMAND:
        return AgentModeCommand()
    if normalized.startswith(f"{_AGENT_COMMAND} "):
        return AgentModeCommand(mode=candidate[len(_AGENT_COMMAND) :].strip())
    if normalized == _MODELS_COMMAND:
        return ModelsCommand()
    if normalized.startswith(f"{_MODELS_COMMAND} "):
        return ModelsCommand(query=candidate[len(_MODELS_COMMAND) :].strip())
    if normalized == _REASONING_COMMAND:
        return ReasoningCommand()
    if normalized.startswith(f"{_REASONING_COMMAND} "):
        return ReasoningCommand(effort=candidate[len(_REASONING_COMMAND) :].strip())
    if normalized == _COUNCIL_COMMAND:
        return CouncilCommand()
    if normalized.startswith(_COUNCIL_COMMAND) and len(candidate) > len(_COUNCIL_COMMAND):
        separator = candidate[len(_COUNCIL_COMMAND)]
        if separator.isspace():
            prompt = candidate[len(_COUNCIL_COMMAND) :].strip()
            return CouncilCommand(prompt=prompt or None)

    if text.startswith("!!"):
        command = text[2:].strip()
        if command:
            return ShellCommand(command=command, include_in_context=False)
    elif text.startswith("!"):
        command = text[1:].strip()
        if command:
            return ShellCommand(command=command, include_in_context=True)
    return NormalPrompt(text=text)
