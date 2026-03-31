"""Bash tool guarded by an LLM-based command safety classifier."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from pathlib import Path

import pyperclip

from mother.tools.bash_executor import execute_bash
from mother.tools.bash_guard import BashGuardDecision, classify_command_async


def _copy_command_to_clipboard(command: str) -> tuple[bool, str]:
    try:
        pyperclip.copy(command)
    except Exception as exc:
        return False, f"Clipboard copy failed: {exc}"
    return True, "The exact command has been copied to the clipboard."


def _format_blocked_command(
    decision: BashGuardDecision,
    *,
    clipboard_copied: bool,
    clipboard_status: str,
) -> str:
    lines = [
        f"{decision.label}: bash guard blocked this command. It was not executed.",
        "",
        "Command:",
        "```bash",
        decision.command,
        "```",
        "",
        f"Guard model: {decision.model_name}",
    ]
    if decision.error is not None:
        lines.append(f"Reason: {decision.error}")
    lines.append(clipboard_status)
    if clipboard_copied:
        lines.append(
            "The user can review the clipboard copy, then paste it into a separate shell if they want to run it outside Mother."
        )
    lines.append(
        "They can also run it manually with `!<command>` to include the output in chat context or `!!<command>` to exclude it."
    )
    return "\n".join(lines)


def make_bash_tool(cwd: Path | None = None) -> Callable[..., Coroutine[object, object, str]]:
    """Factory returning a closure suitable for registration as an llm Tool."""
    effective_cwd = cwd if cwd is not None else Path.cwd()

    async def bash(command: str, timeout: float = 30.0) -> str:
        """Run a shell command on the local machine when the safety guard allows it.

        Use this for local inspection and local actions, such as listing files,
        checking processes, reading command output, or running safe diagnostics.
        Prefer read-only commands first. Risky or destructive commands may be blocked.

        Args:
            command: Shell command to classify and, if allowed, execute.
            timeout: Timeout in seconds. Default is 30.

        Returns:
            Combined stdout and stderr, or a readable guard/error message.
        """
        decision = await classify_command_async(command)
        if not decision.should_run:
            clipboard_copied, clipboard_status = _copy_command_to_clipboard(command)
            return _format_blocked_command(
                decision,
                clipboard_copied=clipboard_copied,
                clipboard_status=clipboard_status,
            )

        try:
            result = await execute_bash(command, cwd=effective_cwd, timeout=timeout)
        except TimeoutError as exc:
            return f"Error: {exc}"
        except Exception:
            raise

        if result.exit_code != 0:
            return f"Command failed (exit code {result.exit_code}):\n{result.output}"

        return result.output

    return bash
