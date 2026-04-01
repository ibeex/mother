"""Direct shell-command execution helpers for MotherApp."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import datetime
from typing import Protocol, cast

from textual.containers import VerticalScroll

from mother.app_session import AppSession
from mother.bash_execution import BashExecution, format_for_context, format_for_display
from mother.interrupts import UserInterruptedError
from mother.tools.bash_capture import BashResult
from mother.user_commands import ShellCommand
from mother.widgets import OutputSection, PromptTextArea, ShellOutput


class _CancellableWorker(Protocol):
    """Minimal worker surface needed to interrupt a running shell command."""

    def cancel(self) -> None: ...


class ShellControllerHost(Protocol):
    """Minimal MotherApp surface used by the shell-command controller."""

    app_session: AppSession

    @property
    def prompt_input(self) -> PromptTextArea: ...

    def query_one(self, selector: object, expect_type: object = None) -> object: ...

    def should_follow_chat_updates(self) -> bool: ...

    def scroll_chat_to_end(self, *, force: bool = False) -> None: ...

    def reset_interrupt_escape(self) -> None: ...

    def execute_shell_command(self, command: str) -> Coroutine[object, object, BashResult]: ...


class ShellCommandController:
    """Own direct shell-command execution state and interruption logic."""

    def __init__(self, host: ShellControllerHost) -> None:
        self.host: ShellControllerHost = host
        self._active_worker: _CancellableWorker | None = None
        self._active_task: asyncio.Task[BashResult] | None = None

    def set_active_worker(self, worker: object | None) -> None:
        """Remember the worker responsible for the current shell command, if any."""
        self._active_worker = cast(_CancellableWorker | None, worker)

    def clear_worker_if_active(self, worker: object) -> bool:
        """Clear the tracked shell worker if it matches the completed worker."""
        if worker is not self._active_worker:
            return False
        self._active_worker = None
        return True

    def has_interruptible_work(self) -> bool:
        """Return whether a shell command task or shell worker is currently active."""
        task_running = self._active_task is not None and not self._active_task.done()
        worker_running = self._active_worker is not None
        return task_running or worker_running

    def interrupt_active_command(self) -> None:
        """Interrupt the active shell task if possible, or cancel its worker."""
        if self._active_task is not None and not self._active_task.done():
            _ = self._active_task.cancel()
            return
        if self._active_worker is not None:
            self._active_worker.cancel()

    @staticmethod
    def _format_interrupted_output(partial_output: str = "") -> str:
        """Format a user-facing interruption notice while preserving partial output."""
        body = partial_output.rstrip()
        if body:
            return f"{body}\n\n_Interrupted by user._"
        return "_Interrupted by user._"

    async def run_user_command(self, cmd: ShellCommand) -> None:
        """Execute a direct user shell command and display the output."""
        chat_view = cast(VerticalScroll, self.host.query_one("#chat-view"))
        should_follow = self.host.should_follow_chat_updates()
        shell_widget = ShellOutput(f"Running: {cmd.command}")
        section = OutputSection("Shell", "shell-title", shell_widget)
        _ = await chat_view.mount(section)
        if should_follow:
            self.host.scroll_chat_to_end(force=True)

        text_area = self.host.prompt_input
        text_area.read_only = True
        self._active_task = asyncio.create_task(self.host.execute_shell_command(cmd.command))

        interrupted = False
        try:
            result = await self._active_task
        except UserInterruptedError as exc:
            interrupted = True
            output = self._format_interrupted_output(exc.partial_output)
            exit_code = None
        except Exception as exc:
            output = f"Error: {exc}"
            exit_code = None
        else:
            output = result.output
            exit_code = result.exit_code
        finally:
            self._active_task = None
            self.host.reset_interrupt_escape()
            text_area.read_only = False
            _ = text_area.focus()

        execution = BashExecution(
            command=cmd.command,
            output=output,
            exit_code=exit_code,
            timestamp=datetime.now(),
            exclude_from_context=not cmd.include_in_context,
        )
        shell_widget.set_text(format_for_display(execution))
        prefix = "!" if cmd.include_in_context else "!!"
        session = self.host.app_session
        session.record_session_message("user", f"{prefix}{cmd.command}")
        session.record_session_message("assistant", format_for_context(execution))

        if interrupted:
            session.record_session_event("shell_command_interrupted", {"command": cmd.command})
            return

        session.pending_executions.append(execution)
