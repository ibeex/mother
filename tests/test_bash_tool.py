"""Tests for the LLM-guarded bash tool."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from mother.tools import get_default_tools
from mother.tools.bash_capture import BashResult
from mother.tools.bash_guard import BashGuardDecision
from mother.tools.bash_tool import make_bash_tool


def test_bash_tool_function_success_when_guard_is_ok():
    ok_result = BashResult(output="file.txt\n", exit_code=0)
    decision = BashGuardDecision(
        command="ls /tmp",
        label="OK",
        raw_output="LABEL: OK",
        canonical_label=True,
    )
    with (
        patch("mother.tools.bash_tool.classify_command", return_value=decision),
        patch("mother.tools.bash_tool.execute_bash", new=AsyncMock(return_value=ok_result)),
    ):
        tool = make_bash_tool(frozenset({"ls"}), Path("/tmp"))
        output = tool("ls /tmp")
    assert "file.txt" in output


def test_bash_tool_warning_blocks_and_copies_to_clipboard():
    decision = BashGuardDecision(
        command="touch notes.txt",
        label="Warning",
        raw_output="LABEL: Warning",
        canonical_label=True,
    )
    with (
        patch("mother.tools.bash_tool.classify_command", return_value=decision),
        patch("mother.tools.bash_tool.pyperclip.copy") as mock_copy,
        patch("mother.tools.bash_tool.execute_bash", new=AsyncMock()) as mock_exec,
    ):
        tool = make_bash_tool(cwd=Path("/tmp"))
        output = tool("touch notes.txt")
    mock_copy.assert_called_once_with("touch notes.txt")
    mock_exec.assert_not_called()
    assert output.startswith("Warning:")
    assert "It was not executed" in output
    assert "copied to clipboard" in output
    assert "!<command>" in output
    assert "!!<command>" in output


def test_bash_tool_fatal_blocks_and_copies_to_clipboard():
    decision = BashGuardDecision(
        command="rm -rf /",
        label="Fatal",
        raw_output="LABEL: Fatal",
        canonical_label=True,
    )
    with (
        patch("mother.tools.bash_tool.classify_command", return_value=decision),
        patch("mother.tools.bash_tool.pyperclip.copy") as mock_copy,
        patch("mother.tools.bash_tool.execute_bash", new=AsyncMock()) as mock_exec,
    ):
        tool = make_bash_tool(cwd=Path("/tmp"))
        output = tool("rm -rf /")
    mock_copy.assert_called_once_with("rm -rf /")
    mock_exec.assert_not_called()
    assert output.startswith("Fatal:")
    assert "It was not executed" in output


def test_bash_tool_guard_error_fails_closed():
    decision = BashGuardDecision(
        command="ls -al",
        label="Fatal",
        raw_output="",
        canonical_label=False,
        error="Could not parse bash guard label from model output.",
    )
    with (
        patch("mother.tools.bash_tool.classify_command", return_value=decision),
        patch("mother.tools.bash_tool.pyperclip.copy") as mock_copy,
        patch("mother.tools.bash_tool.execute_bash", new=AsyncMock()) as mock_exec,
    ):
        tool = make_bash_tool(cwd=Path("/tmp"))
        output = tool("ls -al")
    mock_copy.assert_called_once_with("ls -al")
    mock_exec.assert_not_called()
    assert "Could not parse bash guard label" in output


def test_bash_tool_function_nonzero_exit():
    fail_result = BashResult(output="no such file\n", exit_code=1)
    decision = BashGuardDecision(
        command="ls /nonexistent",
        label="OK",
        raw_output="LABEL: OK",
        canonical_label=True,
    )
    with (
        patch("mother.tools.bash_tool.classify_command", return_value=decision),
        patch("mother.tools.bash_tool.execute_bash", new=AsyncMock(return_value=fail_result)),
    ):
        tool = make_bash_tool(frozenset({"ls"}), Path("/tmp"))
        output = tool("ls /nonexistent")
    assert "exit code 1" in output
    assert "no such file" in output


def test_tool_registered_in_registry():
    registry = get_default_tools(tools_enabled=True)
    assert not registry.is_empty()
    tools = registry.tools()
    assert len(tools) == 2
    assert all(callable(tool) for tool in tools)
