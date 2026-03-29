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
