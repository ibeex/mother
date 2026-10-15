"""Structured tool previews keep commands and results separately readable."""

import asyncio
from unittest.mock import patch

from textual.containers import VerticalScroll
from textual.widgets.text_area import Selection

from mother import MotherApp, MotherConfig
from mother.tool_trace import format_tool_event
from mother.widgets import ConversationTurn, ToolTrace, TraceCommand, TurnLabel


def test_tool_previews_expand_independently_and_preserve_original_copy() -> None:
    async def run() -> None:
        command = "\n".join(f"echo command_{index}" for index in range(20))
        arguments: dict[str, object] = {"command": command, "timeout": 10}
        output = "\n".join(f"result_{index}" for index in range(40))
        app = MotherApp(config=MotherConfig(model="test-model"))

        with patch("mother.widgets.format_bash_for_display", return_value=command):
            async with app.run_test(size=(120, 60)) as pilot:
                turn = ConversationTurn(prompt_text="Inspect the project", response_text="Answer")
                await app.query_one("#chat-view", VerticalScroll).mount(turn)
                app.runtime_presentation.active_turn = turn
                app.runtime_presentation.show_tool_started("bash", "call-1", arguments)
                await pilot.pause()
                trace = turn.query_one(ToolTrace)
                assert str(trace.query_one(".tool-title", TurnLabel).render()) == "bash · Running"
                assert not trace.output.display

                app.runtime_presentation.show_tool_finished("bash", "call-1", arguments, output)
                await pilot.pause()
                assert len(turn.query(ToolTrace)) == 1
                assert str(trace.query_one(".tool-title", TurnLabel).render()) == "bash · Finished"
                command_widget = trace.query(TraceCommand).first()
                assert "command_0" in command_widget.text
                assert "command_19" not in command_widget.text
                assert "result_0" in trace.output.text
                assert "result_39" not in trace.output.text
                assert trace.output.region.y > command_widget.region.bottom
                assert turn.query_one(".response-section").region.y > trace.region.bottom
                assert command_widget.styles.background.a > 0

                # With the composer focused, Ctrl+O targets the latest result region.
                await pilot.press("ctrl+o")
                assert trace.output.text == output
                assert "command_19" not in command_widget.text

                _ = command_widget.focus()
                await pilot.press("ctrl+o")
                assert command_widget.text == command
                assert trace.output.text == output
                expected = format_tool_event("bash", arguments, status="finished", output=output)
                for region in (command_widget, trace.output):
                    with patch("mother.widgets.pyperclip.copy") as copy:
                        region.action_copy_output()
                    copy.assert_called_once_with(expected)

                command_widget.selection = Selection((0, 0), (0, 4))
                with patch("mother.widgets.pyperclip.copy") as copy:
                    command_widget.action_copy_output()
                copy.assert_called_once_with("echo")

    asyncio.run(run())


def test_tool_trace_handles_empty_results_and_missing_started_event() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))
        async with app.run_test() as pilot:
            app.runtime_presentation.show_tool_finished(
                "read_file", "call-2", {"path": "README.md", "headers_json": "{}"}, ""
            )
            await pilot.pause()
            trace = app.query_one(ToolTrace)
            assert trace.output.display
            assert trace.output.text == "(no output)"
            assert len(trace.query(TraceCommand)) == 1
            arguments = trace.query_one(TraceCommand).text
            assert '"path": "README.md"' in arguments
            assert "headers_json" not in arguments
            assert "Command" not in [str(label.render()) for label in trace.query(TurnLabel)]

    asyncio.run(run())


def test_command_formatting_does_not_change_result_or_raw_trace() -> None:
    async def run() -> None:
        command = "echo one; echo two"
        arguments: dict[str, object] = {"command": command}
        trace = ToolTrace("bash", arguments)
        app = MotherApp(config=MotherConfig(model="test-model"))
        with patch("mother.widgets.format_bash_for_display", return_value="echo one\necho two"):
            async with app.run_test() as pilot:
                await app.query_one("#chat-view", VerticalScroll).mount(trace)
                trace.finish(command)
                for worker in list(app.workers):
                    await worker.wait()
                await pilot.pause()
                assert trace.query_one(TraceCommand).text == "echo one\necho two"
                assert trace.output.text == command
                assert trace.raw_text == format_tool_event(
                    "bash", arguments, status="finished", output=command
                )

    asyncio.run(run())
