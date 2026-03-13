"""Async subprocess execution engine for the bash tool."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path

from mother.tools.bash_capture import BashResult, OutputCapture, format_truncation_notice


async def execute_bash(
    command: str,
    cwd: Path | None = None,
    timeout: float | None = None,
    on_data: Callable[[bytes], None] | None = None,
    env: dict[str, str] | None = None,
) -> BashResult:
    """Execute a shell command asynchronously, returning a BashResult.

    Args:
        command: Shell command string (executed via /bin/sh -c).
        cwd: Working directory. If given and does not exist, raises FileNotFoundError.
        timeout: Optional timeout in seconds. On timeout, kills process tree.
        on_data: Optional callback receiving raw bytes as they arrive.
        env: Optional environment dict. If None, inherits from current process.

    Returns:
        BashResult with combined stdout/stderr output and exit code.

    Raises:
        FileNotFoundError: If cwd does not exist before spawn.
        TimeoutError: If the command exceeds the timeout.
    """
    if cwd is not None and not cwd.exists():
        raise FileNotFoundError(f"Working directory does not exist: {cwd}")

    effective_env = dict(os.environ) if env is None else env

    capture = OutputCapture()

    def _handle_data(chunk: bytes) -> None:
        capture.add_bytes(chunk)
        if on_data is not None:
            on_data(chunk)

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
        env=effective_env,
        start_new_session=True,
    )

    async def _read_output() -> None:
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            _handle_data(chunk)

    try:
        if timeout is not None:
            async with asyncio.timeout(timeout):
                await _read_output()
                await proc.wait()
        else:
            await _read_output()
            await proc.wait()
    except TimeoutError as exc:
        _kill_process_group(proc)
        await proc.wait()
        trunc, full_output_path = capture.finalize()
        output = trunc.content
        if output:
            output += "\n\n"
        output += f"Command timed out after {timeout} seconds"
        raise TimeoutError(output) from exc

    trunc, full_output_path = capture.finalize()
    output = trunc.content or "(no output)"

    if trunc.truncated:
        output += format_truncation_notice(trunc, full_output_path)

    return BashResult(
        output=output,
        exit_code=proc.returncode,
        cancelled=False,
        truncated=trunc.truncated,
        full_output_path=full_output_path,
    )


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """Kill the entire process group to terminate child processes too."""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, 9)  # SIGKILL
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
