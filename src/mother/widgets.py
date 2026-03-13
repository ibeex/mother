"""Reusable TUI widget classes for Mother."""

from typing import ClassVar

import pyperclip
from textual.binding import BindingType
from textual.widgets import Markdown, TextArea
from textual.widgets.markdown import MarkdownBlock, MarkdownFence


class Prompt(Markdown):
    """Markdown for the user prompt."""


class CopyableOutput(TextArea):
    """Plain-text output widget with selection-aware copy support."""

    MIN_VISIBLE_LINES: ClassVar[int] = 3
    MAX_VISIBLE_LINES: ClassVar[int] = 12

    BINDINGS: ClassVar[list[BindingType]] = [
        ("c", "copy_output", "Copy"),
    ]
    can_focus: bool = True

    def __init__(self, text: str = "") -> None:
        super().__init__(
            text,
            read_only=True,
            show_cursor=True,
            soft_wrap=False,
            show_line_numbers=False,
            highlight_cursor_line=False,
        )
        self._raw: str = text
        self._sync_height(text)

    def _sync_height(self, text: str) -> None:
        """Size the widget to fit short content without making long output huge."""
        content_lines = max(1, text.count("\n") + 1)
        visible_lines = min(max(content_lines, self.MIN_VISIBLE_LINES), self.MAX_VISIBLE_LINES)
        self.styles.height = visible_lines + 2

    def set_text(self, text: str) -> None:
        """Update the rendered text while keeping a copyable raw value."""
        self._raw = text
        self.text: str = text
        self._sync_height(text)

    def action_copy_output(self) -> None:
        text = self.selected_text or self._raw
        try:
            pyperclip.copy(text)
        except Exception:
            self.app.copy_to_clipboard(text)  # pyright: ignore[reportUnknownMemberType]
        self.notify("Copied!")


class CopyableMarkdown(Markdown):
    """Markdown widget with focus and block-level copy support."""

    BINDINGS: ClassVar[list[BindingType]] = [
        ("j", "cursor_down", "Next block"),
        ("k", "cursor_up", "Prev block"),
        ("c", "copy_block", "Copy"),
    ]
    can_focus: bool = True

    def __init__(self, markdown: str = "") -> None:
        super().__init__(markdown)
        self._cursor: int = 0
        self._raw: str = markdown

    def set_markdown(self, markdown: str):
        """Update the rendered Markdown while keeping a copyable raw value."""
        self._raw = markdown
        return self.update(markdown)

    def reset_state(self, raw: str) -> None:
        """Reset the raw text and cursor position."""
        self._raw = raw
        self._cursor = 0

    def _blocks(self) -> list[MarkdownBlock]:
        return list(self.query(MarkdownBlock))

    def _refresh_highlight(self) -> None:
        for i, block in enumerate(self._blocks()):
            _ = block.set_class(i == self._cursor, "highlight")

    def on_focus(self) -> None:
        self._refresh_highlight()

    def on_blur(self) -> None:
        for block in self._blocks():
            _ = block.remove_class("highlight")

    def action_cursor_down(self) -> None:
        blocks = self._blocks()
        if self._cursor < len(blocks) - 1:
            self._cursor += 1
            self._refresh_highlight()

    def action_cursor_up(self) -> None:
        if self._cursor > 0:
            self._cursor -= 1
            self._refresh_highlight()

    def _block_text(self, block: MarkdownBlock) -> str:
        if isinstance(block, MarkdownFence):
            return block.code
        return block.source or self._raw

    def action_copy_block(self) -> None:
        text = self.screen.get_selected_text()
        if not text:
            blocks = self._blocks()
            text = self._block_text(blocks[self._cursor]) if blocks else self._raw
        try:
            pyperclip.copy(text)
        except Exception:
            self.app.copy_to_clipboard(text)  # pyright: ignore[reportUnknownMemberType]
        self.notify("Copied!")


class ShellOutput(CopyableOutput):
    """Plain-text widget for direct user shell command output."""

    BORDER_TITLE: ClassVar[str] = "Shell"


class ToolOutput(CopyableOutput):
    """Plain-text widget for agent tool execution traces."""

    BORDER_TITLE: ClassVar[str] = "Tool"


class Response(CopyableMarkdown):
    """Markdown for the reply from the LLM, with block-level copy support."""

    BORDER_TITLE: ClassVar[str] = "Mother"
