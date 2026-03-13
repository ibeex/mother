"""Bash tool guarded by an LLM-based command safety classifier."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import pyperclip

from mother.tools.bash_executor import execute_bash
from mother.tools.bash_guard import BashGuardDecision, classify_command

# Legacy compatibility only. The old regex/allowlist guard used this value.
# The current LLM-based guard ignores it, but config/tests still import it.
DEFAULT_ALLOWLIST: frozenset[str] = frozenset({"ls", "cat"})


def _copy_command_to_clipboard(command: str) -> str:
    try:
        pyperclip.copy(command)
    except Exception as exc:
        return f"Clipboard copy failed: {exc}"
    return "Command copied to clipboard."


def _format_blocked_command(decision: BashGuardDecision, clipboard_status: str) -> str:
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
    lines.append(
        "If the user wants to override, they can run it manually with `!<command>` "
        "to include the output in chat context or `!!<command>` to exclude it."
    )
    return "\n".join(lines)


def make_bash_tool(
    allowlist: frozenset[str] = DEFAULT_ALLOWLIST,
    cwd: Path | None = None,
) -> Callable[..., str]:
    """Factory returning a closure suitable for registration as an llm Tool."""
    _ = allowlist
    effective_cwd = cwd if cwd is not None else Path.cwd()

    def bash(command: str, timeout: float = 30.0) -> str:
        """Run a shell command when the guard classifies it as safe.

        Args:
            command: The shell command to classify and maybe execute.
            timeout: Timeout in seconds (default 30).

        Returns:
            Combined stdout and stderr as a string, or a guard/error description.
        """
        decision = classify_command(command)
        if not decision.should_run:
            clipboard_status = _copy_command_to_clipboard(command)
            return _format_blocked_command(decision, clipboard_status)

        try:
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    execute_bash(command, cwd=effective_cwd, timeout=timeout)
                )
            finally:
                loop.close()
        except TimeoutError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error: {exc}"

        if result.exit_code != 0:
            return f"Command failed (exit code {result.exit_code}):\n{result.output}"

        return result.output

    return bash
