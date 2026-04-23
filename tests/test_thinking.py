"""Tests for the dedicated structured-thinking widget."""

from mother.widgets import ThinkingOutput


def test_thinking_output_wraps_and_shows_full_text_while_streaming_then_collapses() -> None:
    lines = [f"line {index}" for index in range(12)]
    widget = ThinkingOutput()

    assert widget.soft_wrap is True

    widget.start_streaming()
    widget.set_text("\n".join(lines))

    assert widget.display is True
    assert widget.text == "\n".join(lines)

    widget.finish_streaming()

    assert "line 0" in widget.text
    assert "line 9" in widget.text
    assert "line 10" not in widget.text
    assert "Press Ctrl+o for rest." in widget.text

    widget.action_toggle_expanded()

    assert widget.text == "\n".join(lines)
