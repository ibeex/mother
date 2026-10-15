"""Tests for the dedicated structured-thinking widget."""

import asyncio

from textual.containers import VerticalScroll

from mother import MotherApp, MotherConfig
from mother.widgets import ConversationTurn, ThinkingOutput


def test_thinking_output_wraps_and_shows_full_text_while_streaming_then_collapses() -> None:
    lines = [f"line {index}" for index in range(12)]
    widget = ThinkingOutput()

    assert widget.soft_wrap is True

    widget.start_streaming()
    widget.set_text("\n".join(lines))

    assert widget.display is True
    assert widget.text == "\n".join(lines)
    assert widget.styles.height is not None
    assert widget.styles.height.value == len(lines)

    widget.finish_streaming()

    assert "line 0" in widget.text
    assert "line 9" in widget.text
    assert "line 10" not in widget.text
    assert "Press Ctrl+o for rest." in widget.text

    widget.action_toggle_expanded()

    assert widget.text == "\n".join(lines)


def test_wrapped_thinking_grows_then_collapses_and_expands_with_ctrl_o() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))
        async with app.run_test(size=(100, 60)) as pilot:
            turn = ConversationTurn(prompt_text="Think", include_thinking=True)
            await app.query_one("#chat-view", VerticalScroll).mount(turn)
            widget = turn.query_one(ThinkingOutput)
            widget.start_streaming()
            widget.set_text("A long paragraph with no newlines. " * 80)
            await pilot.pause()

            assert widget.wrapped_document.height > widget.PREVIEW_LINES + 1
            assert widget.size.height == widget.wrapped_document.height
            wide_height = widget.size.height

            await pilot.resize_terminal(70, 60)
            await pilot.pause()
            assert widget.size.height > wide_height
            assert widget.size.height == widget.wrapped_document.height
            streaming_height = widget.size.height

            widget.finish_streaming()
            assert widget.size.height == streaming_height
            await pilot.pause(widget.COLLAPSE_DELAY + 0.1)
            assert widget.size.height == widget.PREVIEW_LINES + 1
            assert widget.scroll_y == 0
            assert widget.can_toggle_expanded()

            # The composer remains focused; the global binding must find thinking
            # even though the entire response is just one logical line.
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert widget.size.height == widget.wrapped_document.height
            _ = widget.focus()
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert widget.size.height == widget.PREVIEW_LINES + 1

    asyncio.run(run())


def test_new_thinking_stream_cancels_pending_collapse() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))
        async with app.run_test(size=(100, 60)) as pilot:
            widget = ThinkingOutput()
            await app.query_one("#chat-view", VerticalScroll).mount(widget)
            widget.start_streaming()
            widget.set_text("\n".join(f"step {index}" for index in range(20)))
            await pilot.pause()
            widget.finish_streaming()
            widget.start_streaming()
            await pilot.pause(widget.COLLAPSE_DELAY + 0.1)
            assert widget.size.height == 20
            assert "step 19" in widget.text
            assert not widget.can_toggle_expanded()

    asyncio.run(run())
