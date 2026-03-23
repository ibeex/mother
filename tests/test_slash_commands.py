"""Tests for slash-command autocomplete helpers."""

from unittest.mock import patch

from mother.slash_commands import (
    SLASH_COMMANDS,
    current_slash_argument_query,
    current_slash_query,
    filter_agent_argument_choices,
    filter_model_argument_choices,
    filter_reasoning_argument_choices,
    filter_slash_commands,
    get_slash_argument_spec,
    should_expand_slash_argument,
)


def test_current_slash_query_accepts_single_token_slash_command() -> None:
    assert current_slash_query("/sa") == "/sa"


def test_current_slash_query_rejects_arguments() -> None:
    assert current_slash_query("/save now") is None


def test_current_slash_query_rejects_trailing_space() -> None:
    assert current_slash_query("/save ") is None


def test_current_slash_query_rejects_multiline_input() -> None:
    assert current_slash_query("/save\nmore") is None


def test_current_slash_argument_query_detects_models_arguments() -> None:
    query = current_slash_argument_query("/models opus")

    assert query is not None
    assert query.command == "/models"
    assert query.query == "opus"


def test_current_slash_argument_query_detects_agent_arguments() -> None:
    query = current_slash_argument_query(" /agent deep ")

    assert query is not None
    assert query.command == "/agent"
    assert query.query == "deep"


def test_current_slash_argument_query_detects_reasoning_arguments() -> None:
    query = current_slash_argument_query(" /reasoning hi ")

    assert query is not None
    assert query.command == "/reasoning"
    assert query.query == "hi"


def test_should_expand_slash_argument_for_supported_commands() -> None:
    assert should_expand_slash_argument("/agent") is True
    assert should_expand_slash_argument("/models") is True
    assert should_expand_slash_argument(" /reasoning") is True
    assert should_expand_slash_argument("/save") is False


def test_get_slash_argument_spec_returns_registered_commands() -> None:
    assert get_slash_argument_spec("/agent") is not None
    assert get_slash_argument_spec("/models") is not None
    assert get_slash_argument_spec("/reasoning") is not None
    assert get_slash_argument_spec("/save") is None


def test_filter_slash_commands_matches_save_prefix() -> None:
    matches = filter_slash_commands(SLASH_COMMANDS, "/sa")

    assert [command.command for command in matches] == ["/save"]


def test_filter_agent_argument_choices_support_conversational_aliases() -> None:
    matches = filter_agent_argument_choices("conv")

    assert [choice.value for choice in matches] == ["standard"]


def test_filter_model_argument_choices_supports_fuzzy_subsequence_matches() -> None:
    available_models = [
        ("local_1", "local_1 — local 1"),
        ("local_2", "local_2 — local 2"),
        ("local_3", "local_3 — local 3"),
    ]

    with patch("mother.slash_commands.get_available_models", return_value=available_models):
        matches = filter_model_argument_choices("lo3")

    assert [choice.value for choice in matches] == ["local_3"]


def test_filter_reasoning_argument_choices_supports_fuzzy_subsequence_matches() -> None:
    matches = filter_reasoning_argument_choices("mdm")

    assert [choice.value for choice in matches] == ["medium"]


def test_default_slash_commands_expose_save() -> None:
    assert any(command.command == "/save" for command in SLASH_COMMANDS)


def test_default_slash_commands_expose_quit_aliases() -> None:
    commands = {command.command for command in SLASH_COMMANDS}

    assert "/quit" in commands
    assert "/exit" in commands


def test_default_slash_commands_expose_agent() -> None:
    assert any(command.command == "/agent" for command in SLASH_COMMANDS)


def test_default_slash_commands_expose_models() -> None:
    assert any(command.command == "/models" for command in SLASH_COMMANDS)


def test_default_slash_commands_expose_reasoning() -> None:
    assert any(command.command == "/reasoning" for command in SLASH_COMMANDS)
