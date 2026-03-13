"""Tests for the async bash subprocess execution engine."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mother.tools.bash_executor import execute_bash


def run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def test_execute_simple_command():
    result = run(execute_bash("echo hello"))
    assert result.exit_code == 0
    assert "hello" in result.output


def test_execute_exit_code():
    result = run(execute_bash("exit 42"))
    assert result.exit_code == 42


def test_execute_stderr_captured():
    result = run(execute_bash("echo oops >&2"))
    assert "oops" in result.output


def test_execute_timeout():
    with pytest.raises(TimeoutError):
        run(execute_bash("sleep 30", timeout=0.1))


def test_execute_working_directory(tmp_path: Path):
    result = run(execute_bash("pwd", cwd=tmp_path))
    assert result.exit_code == 0
    assert str(tmp_path.resolve()) in result.output


def test_execute_invalid_cwd():
    bad_cwd = Path("/nonexistent_dir_that_does_not_exist_xyz")
    with pytest.raises(FileNotFoundError):
        run(execute_bash("echo hi", cwd=bad_cwd))


def test_execute_streaming_callback():
    chunks: list[bytes] = []
    run(execute_bash("echo streaming", on_data=chunks.append))
    combined = b"".join(chunks)
    assert b"streaming" in combined


def test_execute_shell_syntax():
    result = run(execute_bash("echo foo | tr f b"))
    assert result.exit_code == 0
    assert "boo" in result.output


def test_execute_env_override():
    result = run(execute_bash("echo $MY_VAR", env={"MY_VAR": "secret42", "PATH": "/usr/bin:/bin"}))
    assert result.exit_code == 0
    assert "secret42" in result.output


def test_execute_uses_start_new_session_instead_of_preexec_fn():
    fake_stdout = AsyncMock()
    fake_stdout.read = AsyncMock(side_effect=[b"", b""])
    fake_proc = AsyncMock()
    fake_proc.stdout = fake_stdout
    fake_proc.wait = AsyncMock(return_value=0)
    fake_proc.returncode = 0
    fake_proc.pid = 12345

    with patch(
        "asyncio.create_subprocess_shell", new=AsyncMock(return_value=fake_proc)
    ) as mock_spawn:
        result = run(execute_bash("echo hello"))

    assert result.exit_code == 0
    _, kwargs = mock_spawn.call_args
    assert kwargs["start_new_session"] is True
    assert "preexec_fn" not in kwargs
