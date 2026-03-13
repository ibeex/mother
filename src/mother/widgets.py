"""Reusable TUI widget classes for Mother."""

from typing import ClassVar, final, override

import pyperclip
from textual.binding import BindingType
from textual.containers import Vertical
from textual.widgets import Label, Markdown, TextArea
from textual.widgets.markdown import MarkdownBlock, MarkdownFence


class Prompt(Markdown):
    """Markdown for the user prompt."""

    BORDER_TITLE: ClassVar[str] = "You"


class CopyableOutput(TextArea):
    """Plain-text output widget with selection-aware copy support."""

    # Override TextArea's DEFAULT_CSS `border: tall $border` — derived-class
    # DEFAULT_CSS loads last in Textual's cascade and therefore wins.
    DEFAULT_CSS: ClassVar[str] = """
    CopyableOutput {
        border: none;
        background: transparent;
        padding: 0 1;
    }
    """

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
        # Inline style (highest priority) — overrides TextArea's DEFAULT_CSS border.
        self.styles.border = ("none", "transparent")
        self._raw: str = text
        self._sync_height(text)

    def _chrome_height(self) -> int:
        """Return extra lines used by borders and padding."""
        return 0

    def _sync_height(self, text: str) -> None:
        """Size the widget to fit short content without making long output huge."""
        content_lines = max(1, text.count("\n") + 1)
        visible_lines = min(max(content_lines, self.MIN_VISIBLE_LINES), self.MAX_VISIBLE_LINES)
        self.styles.height = visible_lines + self._chrome_height()

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


@final
class ThinkingOutput(CopyableOutput):
    """Plain-text widget for model reasoning emitted inside ``<think>`` tags."""

    PREVIEW_LINES: ClassVar[int] = 10
    BORDER_TITLE: ClassVar[str] = "Mother · thinking"
    BINDINGS: ClassVar[list[BindingType]] = [
        *CopyableOutput.BINDINGS,
        ("ctrl+o", "toggle_expanded", "Rest"),
    ]

    def __init__(self, text: str = "") -> None:
        self._expanded: bool = False
        super().__init__("")
        self.set_text(text)

    def has_content(self) -> bool:
        """Return whether the widget currently contains any thinking text."""
        return bool(self._raw.strip())

    def _preview_text(self) -> str:
        lines = self._raw.splitlines()
        if len(lines) <= self.PREVIEW_LINES:
            return self._raw
        preview = "\n".join(lines[: self.PREVIEW_LINES])
        remaining = len(lines) - self.PREVIEW_LINES
        return f"{preview}\n… {remaining} more lines. Press Ctrl+O for rest."

    def _render_text(self) -> str:
        if self._expanded:
            return self._raw
        return self._preview_text()

    @override
    def set_text(self, text: str) -> None:
        """Update raw thinking text and render preview/full content."""
        self._raw = text
        visible = self.has_content()
        self.display = visible
        parent = self.parent
        if parent is not None and parent.has_class("thinking-section"):
            parent.display = visible
        rendered = self._render_text()
        self.text = rendered
        self._sync_height(rendered)

    def action_toggle_expanded(self) -> None:
        """Toggle between the preview and the full reasoning text."""
        if not self.has_content() or len(self._raw.splitlines()) <= self.PREVIEW_LINES:
            return
        self._expanded = not self._expanded
        rendered = self._render_text()
        self.text = rendered
        self._sync_height(rendered)


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


class TurnLabel(Label):
    """Section label used inside a grouped conversation turn."""

    def __init__(self, text: str, *, classes: str) -> None:
        super().__init__(text, markup=False, classes=classes)


class ConversationTurn(Vertical):
    """A grouped prompt / thinking / response block with shared outer border."""

    DEFAULT_CSS: ClassVar[str] = """
    ConversationTurn {
        border: none;
        margin: 0;
        background: transparent;
        height: auto;
    }
    """

    def __init__(
        self,
        *,
        prompt_text: str | None = None,
        response_text: str = "",
        include_thinking: bool = False,
    ) -> None:
        children: list[Vertical] = []
        has_prompt = prompt_text is not None

        self.prompt_widget: Prompt | None = None
        if prompt_text is not None:
            self.prompt_widget = Prompt(prompt_text)
            _ = self.prompt_widget.add_class("grouped")
            children.append(
                Vertical(
                    TurnLabel("You", classes="turn-title prompt-title"),
                    self.prompt_widget,
                    classes="turn-section prompt-section",
                )
            )

        self.thinking_output: ThinkingOutput | None = None
        self.thinking_section: Vertical | None = None
        if include_thinking:
            self.thinking_output = ThinkingOutput()
            _ = self.thinking_output.add_class("grouped")
            self.thinking_output.set_text("")
            self.thinking_section = Vertical(
                TurnLabel("Mother · thinking", classes="turn-title thinking-title"),
                self.thinking_output,
                classes="turn-section thinking-section separated",
            )
            self.thinking_section.display = False
            children.append(self.thinking_section)

        self.response_widget: Response = Response(response_text)
        _ = self.response_widget.add_class("grouped")
        response_classes = "turn-section response-section"
        if has_prompt or include_thinking:
            response_classes += " separated"
        children.append(
            Vertical(
                TurnLabel("Mother", classes="turn-title response-title"),
                self.response_widget,
                classes=response_classes,
            )
        )

        super().__init__(*children, classes="conversation-turn")


class StatusLine(Label):
    """Single-line status widget shown above the footer."""

    DEFAULT_CSS: ClassVar[str] = """
    StatusLine {
        width: 100%;
        height: 1;
        padding: 0 1;
        color: $footer-description-foreground;
        background: $footer-description-background;
        text-align: right;
        text-wrap: nowrap;
    }
    """

    def __init__(
        self,
        model_name: str,
        agent_mode: bool,
        context_tokens: int | None = None,
    ) -> None:
        super().__init__(
            self.format_status(model_name, agent_mode, context_tokens),
            id="status-line",
            markup=False,
        )

    @staticmethod
    def format_status(model_name: str, agent_mode: bool, context_tokens: int | None) -> str:
        """Format the text displayed in the status line."""
        model = model_name or "?"
        agent = "on" if agent_mode else "off"
        if context_tokens is None:
            context = "?"
        elif context_tokens >= 1000:
            context = f"{context_tokens / 1000:.1f}k"
        else:
            context = str(context_tokens)
        return f"{model} · {agent} · {context}"

    def set_status(
        self,
        *,
        model_name: str,
        agent_mode: bool,
        context_tokens: int | None,
    ) -> None:
        """Update the displayed model, agent mode, and context size."""
        self.update(self.format_status(model_name, agent_mode, context_tokens))


class ShellOutput(CopyableOutput):
    """Plain-text widget for direct user shell command output."""


class ToolOutput(CopyableOutput):
    """Plain-text widget for agent tool execution traces."""


@final
class OutputSection(Vertical):
    """A labeled wrapper that groups a CopyableOutput with a TurnLabel header."""

    def __init__(self, label: str, label_class: str, widget: CopyableOutput) -> None:
        _ = widget.add_class("grouped")
        super().__init__(
            TurnLabel(label, classes=f"turn-title {label_class}"),
            widget,
            classes="turn-section grouped",
        )
        self._output = widget

    @property
    def output_widget(self) -> CopyableOutput:
        return self._output


class Response(CopyableMarkdown):
    """Markdown for the reply from the LLM, with block-level copy support."""

    BORDER_TITLE: ClassVar[str] = "Mother"
