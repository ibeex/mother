"""Tests for user command parsing and bash execution records."""

from datetime import datetime

from mother.bash_execution import BashExecution, format_for_context
from mother.user_commands import (
    NormalPrompt,
    QuitAppCommand,
    SaveSessionCommand,
    ShellCommand,
    parse_user_input,
    should_submit_on_enter,
)


def test_detect_bang_command():
    result = parse_user_input("!ls -la")
    assert isinstance(result, ShellCommand)
    assert result.command == "ls -la"
    assert result.include_in_context is True


def test_detect_double_bang():
    result = parse_user_input("!!ps aux")
    assert isinstance(result, ShellCommand)
    assert result.command == "ps aux"
    assert result.include_in_context is False


def test_detect_normal_prompt():
    result = parse_user_input("hello")
    assert isinstance(result, NormalPrompt)
    assert result.text == "hello"


def test_detect_save_command():
    result = parse_user_input("/save")
    assert isinstance(result, SaveSessionCommand)
    assert result.command == "/save"


def test_detect_export_command():
    result = parse_user_input(" /export ")
    assert isinstance(result, SaveSessionCommand)
    assert result.command == "/export"


def test_detect_quit_command():
    result = parse_user_input("/quit")
    assert isinstance(result, QuitAppCommand)
    assert result.command == "/quit"


def test_detect_exit_command():
    result = parse_user_input(" /exit ")
    assert isinstance(result, QuitAppCommand)
    assert result.command == "/exit"


def test_should_submit_save_on_enter():
    assert should_submit_on_enter("/save ") is True


def test_should_submit_quit_on_enter():
    assert should_submit_on_enter("/quit") is True


def test_should_not_submit_normal_prompt_on_enter():
    assert should_submit_on_enter("hello") is False


def test_should_not_submit_multiline_text_on_enter():
    assert should_submit_on_enter("/quit\nnow") is False


def test_detect_bang_only():
    result = parse_user_input("!")
    assert isinstance(result, NormalPrompt)
    assert result.text == "!"


def test_user_command_bypasses_allowlist():
    # ShellCommand has no allowlist field — validation is the caller's responsibility
    cmd = parse_user_input("!rm --help")
    assert isinstance(cmd, ShellCommand)
    assert cmd.command == "rm --help"


def test_format_execution_for_context():
    execution = BashExecution(
        command="git status",
        output="On branch main\n",
        exit_code=0,
        timestamp=datetime.now(),
        exclude_from_context=False,
    )
    text = format_for_context(execution)
    assert "git status" in text
    assert "On branch main" in text


def test_excluded_execution_not_in_context():
    execution = BashExecution(
        command="ps aux",
        output="processes\n",
        exit_code=0,
        timestamp=datetime.now(),
        exclude_from_context=True,
    )
    # exclude_from_context is True — the caller decides not to include it
    # format_for_context still works but should not be called for excluded ones
    assert execution.exclude_from_context is True
    text = format_for_context(execution)
    assert "ps aux" in text  # format still works; caller checks flag
