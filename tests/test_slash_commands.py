"""Tests for slash-command autocomplete helpers."""

from mother.slash_commands import SLASH_COMMANDS, current_slash_query, filter_slash_commands


def test_current_slash_query_accepts_single_token_slash_command() -> None:
    assert current_slash_query("/sa") == "/sa"


def test_current_slash_query_rejects_arguments() -> None:
    assert current_slash_query("/save now") is None


def test_current_slash_query_rejects_trailing_space() -> None:
    assert current_slash_query("/save ") is None


def test_current_slash_query_rejects_multiline_input() -> None:
    assert current_slash_query("/save\nmore") is None


def test_filter_slash_commands_matches_save_prefix() -> None:
    matches = filter_slash_commands(SLASH_COMMANDS, "/sa")

    assert [command.command for command in matches] == ["/save"]


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
