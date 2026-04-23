"""Tests for copy behavior in shell/tool output widgets."""

from unittest.mock import patch

from textual.widgets.text_area import Selection

from mother.widgets import CopyableOutput


def test_copyable_output_copies_selected_text_when_present():
    widget = CopyableOutput("abcde\nworld")
    widget.selection = Selection((0, 0), (0, 3))

    with (
        patch("mother.widgets.pyperclip.copy") as mock_copy,
        patch.object(widget, "notify"),
    ):
        widget.action_copy_output()

    mock_copy.assert_called_once_with("abc")


def test_copyable_output_copies_full_text_when_selection_empty():
    widget = CopyableOutput("hello\nworld")

    with (
        patch("mother.widgets.pyperclip.copy") as mock_copy,
        patch.object(widget, "notify"),
    ):
        widget.action_copy_output()

    mock_copy.assert_called_once_with("hello\nworld")


def test_copyable_output_collapses_long_text_with_expand_notice():
    lines = [f"line {index}" for index in range(15)]
    raw = "\n".join(lines)
    widget = CopyableOutput(raw)

    assert "line 0" in widget.text
    assert "line 10" in widget.text
    assert "line 11" not in widget.text
    assert "hidden in UI" in widget.text
    assert "Press Ctrl+o to expand." in widget.text

    widget.action_toggle_expanded()

    assert widget.text == raw

    widget.action_toggle_expanded()

    assert "line 11" not in widget.text
    assert "Press Ctrl+o to expand." in widget.text


def test_copyable_output_preview_mentions_real_truncation_notice():
    raw = "\n".join([f"line {index}" for index in range(15)])
    raw += "\n[Showing lines 20-40 of 40. Full output: /tmp/mother_bash_abc.txt]"
    widget = CopyableOutput(raw)

    assert "Output was also truncated." in widget.text


def test_copyable_output_copy_uses_raw_text_even_when_preview_is_collapsed():
    raw = "\n".join(f"line {index}" for index in range(15))
    widget = CopyableOutput(raw)

    with (
        patch("mother.widgets.pyperclip.copy") as mock_copy,
        patch.object(widget, "notify"),
    ):
        widget.action_copy_output()

    mock_copy.assert_called_once_with(raw)
