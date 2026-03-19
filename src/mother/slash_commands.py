"""Slash command metadata and matching helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from mother.model_picker import filter_available_models, get_available_models
from mother.reasoning import format_reasoning_effort, normalize_reasoning_effort


@dataclass(frozen=True)
class SlashCommand:
    """A built-in slash command shown in autocomplete."""

    command: str
    help: str
    hint: str | None = None


@dataclass(frozen=True)
class SlashArgumentQuery:
    """The active inline argument query for a slash command."""

    command: str
    query: str


@dataclass(frozen=True)
class SlashArgumentChoice:
    """A single inline argument completion choice."""

    value: str
    label: str
    search_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class SlashArgumentSpec:
    """Inline completion behavior for a slash command's argument."""

    command: str
    complete: Callable[[str], list[SlashArgumentChoice]]
    resolve: Callable[[str], str | None]


SLASH_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/save", "Save the current session to markdown"),
    SlashCommand("/quit", "Quit Mother"),
    SlashCommand("/exit", "Quit Mother"),
    SlashCommand("/agent", "Toggle agent mode"),
    SlashCommand("/models", "Browse and switch models"),
    SlashCommand("/reasoning", "Set reasoning effort", "auto|off|low|medium|high|xhigh"),
)

_REASONING_ARGUMENT_CHOICES: tuple[SlashArgumentChoice, ...] = (
    SlashArgumentChoice("auto", "auto"),
    SlashArgumentChoice("off", "off", ("none",)),
    SlashArgumentChoice("low", "low"),
    SlashArgumentChoice("medium", "medium"),
    SlashArgumentChoice("high", "high"),
    SlashArgumentChoice("xhigh", "xhigh"),
)


def _filter_argument_choices(
    choices: Iterable[SlashArgumentChoice],
    query: str,
) -> list[SlashArgumentChoice]:
    """Filter inline argument choices, preferring prefix matches."""
    items = list(choices)
    normalized = query.strip().lower()
    if not normalized:
        return items

    prefix_matches = [
        choice
        for choice in items
        if choice.value.lower().startswith(normalized)
        or any(term.lower().startswith(normalized) for term in choice.search_terms)
    ]
    if prefix_matches:
        return prefix_matches

    return [
        choice
        for choice in items
        if normalized in choice.value.lower()
        or normalized in choice.label.lower()
        or any(normalized in term.lower() for term in choice.search_terms)
    ]


def filter_model_argument_choices(query: str) -> list[SlashArgumentChoice]:
    """Return inline completion choices for ``/models``."""
    return [
        SlashArgumentChoice(model_id, label)
        for model_id, label in filter_available_models(query, get_available_models())
    ]


def resolve_model_argument(query: str) -> str | None:
    """Resolve a ``/models`` query to a concrete model id, if possible."""
    matches = filter_model_argument_choices(query)
    if not matches:
        return None

    normalized = query.strip().lower()
    for match in matches:
        if match.value.lower() == normalized:
            return match.value
    return matches[0].value


def filter_reasoning_argument_choices(query: str) -> list[SlashArgumentChoice]:
    """Return inline completion choices for ``/reasoning``."""
    return _filter_argument_choices(_REASONING_ARGUMENT_CHOICES, query)


def resolve_reasoning_argument(query: str) -> str | None:
    """Resolve a ``/reasoning`` query to a canonical user-facing value."""
    normalized = normalize_reasoning_effort(query)
    if normalized is not None:
        return format_reasoning_effort(normalized)

    matches = filter_reasoning_argument_choices(query)
    if not matches:
        return None
    return matches[0].value


SLASH_ARGUMENT_SPECS: tuple[SlashArgumentSpec, ...] = (
    SlashArgumentSpec(
        command="/models",
        complete=filter_model_argument_choices,
        resolve=resolve_model_argument,
    ),
    SlashArgumentSpec(
        command="/reasoning",
        complete=filter_reasoning_argument_choices,
        resolve=resolve_reasoning_argument,
    ),
)

_SLASH_ARGUMENT_SPECS_BY_COMMAND: dict[str, SlashArgumentSpec] = {
    spec.command.lower(): spec for spec in SLASH_ARGUMENT_SPECS
}


def get_slash_argument_spec(command: str) -> SlashArgumentSpec | None:
    """Return inline completion metadata for a slash command, if configured."""
    return _SLASH_ARGUMENT_SPECS_BY_COMMAND.get(command.lower())


def current_slash_argument_query(text: str) -> SlashArgumentQuery | None:
    """Return the active slash-command argument query, if any."""
    if not text or "\n" in text:
        return None

    candidate = text.lstrip()
    normalized = candidate.lower()
    for spec in sorted(SLASH_ARGUMENT_SPECS, key=lambda item: len(item.command), reverse=True):
        command = spec.command.lower()
        if not normalized.startswith(f"{command} "):
            continue
        return SlashArgumentQuery(spec.command, candidate[len(spec.command) :].strip())
    return None


def should_expand_slash_argument(text: str) -> bool:
    """Return whether Tab or a typed character should expand a slash argument command."""
    if not text or "\n" in text:
        return False
    return text.lstrip().lower() in _SLASH_ARGUMENT_SPECS_BY_COMMAND


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
