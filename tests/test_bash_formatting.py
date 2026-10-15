"""Optional shell formatter behavior and display-only tool trace integration."""

import asyncio
import shutil
import subprocess
from threading import Event, get_ident
from typing import override
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult

from mother.bash_formatting import format_bash_for_display
from mother.tool_trace import format_tool_event
from mother.widgets import ToolOutput


def test_missing_shfmt_preserves_command() -> None:
    command = 'echo "one; two"; pwd'
    with (
        patch("mother.bash_formatting.shutil.which", return_value=None),
        patch("mother.bash_formatting.subprocess.run") as run,
    ):
        assert format_bash_for_display(command) == command
    run.assert_not_called()


def test_shfmt_receives_bash_on_stdin_without_execution() -> None:
    command = 'echo "one; two"; pwd'
    formatted = 'echo "one; two"\npwd\n'
    with (
        patch("mother.bash_formatting.shutil.which", return_value="/bin/shfmt"),
        patch(
            "mother.bash_formatting.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, formatted, ""),
        ) as run,
    ):
        assert format_bash_for_display(command) == formatted.removesuffix("\n")
    run.assert_called_once_with(
        ["/bin/shfmt", "-ln", "bash", "-i", "2"],
        input=command,
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )


@pytest.mark.parametrize("returncode, stdout", [(1, "partial\n"), (0, ""), (0, "\n")])
def test_rejected_or_empty_formatting_preserves_command(returncode: int, stdout: str) -> None:
    command = "echo hello; pwd"
    with (
        patch("mother.bash_formatting.shutil.which", return_value="/bin/shfmt"),
        patch(
            "mother.bash_formatting.subprocess.run",
            return_value=subprocess.CompletedProcess([], returncode, stdout, "error"),
        ),
    ):
        assert format_bash_for_display(command) == command


@pytest.mark.parametrize(
    "error",
    [OSError("unavailable"), subprocess.TimeoutExpired("shfmt", 2), UnicodeError("invalid text")],
)
def test_formatter_errors_preserve_command(error: Exception) -> None:
    command = "echo hello; pwd"
    with (
        patch("mother.bash_formatting.shutil.which", return_value="/bin/shfmt"),
        patch("mother.bash_formatting.subprocess.run", side_effect=error),
    ):
        assert format_bash_for_display(command) == command


@pytest.mark.skipif(shutil.which("shfmt") is None, reason="optional shfmt executable not installed")
def test_real_shfmt_preserves_heredoc_here_string_and_quoted_semicolons() -> None:
    command = 'echo "one; two"; cat <<< "three; four"\npython - <<\'PY\'\nprint("five; six")\nPY\n'
    formatted = format_bash_for_display(command)
    assert formatted.startswith('echo "one; two"\ncat <<<"three; four"\n')
    assert "python - <<'PY'\nprint(\"five; six\")\nPY" in formatted


def test_tool_formatting_runs_off_ui_thread_and_preserves_finished_output_and_copy() -> None:
    async def run() -> None:
        command = "echo one; echo two"
        arguments: dict[str, object] = {"command": command, "timeout": 10}
        started_text = format_tool_event("bash", arguments, status="started")
        # Even output identical to the command must not be formatted.
        finished_text = format_tool_event("bash", arguments, status="finished", output=command)
        widget = ToolOutput(started_text, bash_command=command)
        ui_thread = get_ident()
        entered = Event()
        release = Event()
        formatter_threads: list[int] = []

        def format_command(value: str) -> str:
            formatter_threads.append(get_ident())
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release formatter")
            return value.replace("; ", "\n")

        class TestApp(App[None]):
            @override
            def compose(self) -> ComposeResult:
                yield widget

        app = TestApp()
        with patch(
            "mother.widgets.format_bash_for_display", side_effect=format_command
        ) as formatter:
            async with app.run_test() as pilot:
                try:
                    assert await asyncio.to_thread(entered.wait, 2)
                    widget.set_text(finished_text)
                finally:
                    release.set()
                for worker in list(app.workers):
                    await worker.wait()
                await pilot.pause()
                assert formatter_threads != [ui_thread]
                assert len(formatter_threads) == 1
                formatter.assert_called_once_with(command)
                assert "Status: finished" in widget.text
                assert "Command:\necho one\necho two" in widget.text
                assert f"Output:\n{command}" in widget.text
                assert arguments["command"] == command
                with patch("mother.widgets.pyperclip.copy") as copy:
                    widget.action_copy_output()
                copy.assert_called_once_with(finished_text)

    asyncio.run(run())


def test_tool_output_without_bash_command_does_not_format() -> None:
    async def run() -> None:
        widget = ToolOutput("Command:\nsome other tool")

        class TestApp(App[None]):
            @override
            def compose(self) -> ComposeResult:
                yield widget

        with patch("mother.widgets.format_bash_for_display") as formatter:
            app = TestApp()
            async with app.run_test():
                assert not list(app.workers)
                assert widget.text == "Command:\nsome other tool"
            formatter.assert_not_called()

    asyncio.run(run())
