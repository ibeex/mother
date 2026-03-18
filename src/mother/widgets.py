"""Reusable TUI widget classes for Mother."""

from dataclasses import dataclass
from typing import ClassVar, Protocol, cast, final, override

import pyperclip
from textual import events
from textual.binding import BindingType
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Label, Markdown, OptionList, TextArea
from textual.widgets.markdown import MarkdownBlock, MarkdownFence
from textual.widgets.option_list import Option

from mother.model_picker import filter_available_models, get_available_models
from mother.slash_commands import SlashCommand, filter_slash_commands
from mother.user_commands import should_expand_models_query, should_submit_on_enter


class _PromptSubmitApp(Protocol):
    """Subset of app API used by the prompt input widget."""

    async def action_submit(self) -> None: ...


class Prompt(Markdown):
    """Markdown for the user prompt."""

    BORDER_TITLE: ClassVar[str] = "You"


class PromptTextArea(TextArea):
    """Main prompt input with slash-complete key handling."""

    slash_complete_active: bool = False
    model_complete_active: bool = False

    @dataclass
    class SlashNavigate(Message):
        """Request that the slash popup move its highlight."""

        text_area: "PromptTextArea"
        direction: int

        @property
        @override
        def control(self) -> "PromptTextArea":
            return self.text_area

    @dataclass
    class SlashAccept(Message):
        """Request that the slash popup accept its current selection."""

        text_area: "PromptTextArea"

        @property
        @override
        def control(self) -> "PromptTextArea":
            return self.text_area

    @dataclass
    class SlashDismiss(Message):
        """Request that the slash popup close."""

        text_area: "PromptTextArea"

        @property
        @override
        def control(self) -> "PromptTextArea":
            return self.text_area

    @dataclass
    class ModelNavigate(Message):
        """Request that the model popup move its highlight."""

        text_area: "PromptTextArea"
        direction: int

        @property
        @override
        def control(self) -> "PromptTextArea":
            return self.text_area

    @dataclass
    class ModelDismiss(Message):
        """Request that the model popup close."""

        text_area: "PromptTextArea"

        @property
        @override
        def control(self) -> "PromptTextArea":
            return self.text_area

    @dataclass
    class SlashSubmit(Message):
        """Request that the built-in slash command be submitted immediately."""

        text_area: "PromptTextArea"

        @property
        @override
        def control(self) -> "PromptTextArea":
            return self.text_area

    def _selection_is_at_end(self) -> bool:
        start, end = self.selection
        return start == end == (0, len(self.text))

    def _expand_models_query_prefix(self, character: str | None = None) -> bool:
        """Expand ``/models`` to ``/models `` and optionally append a typed character."""
        if not should_expand_models_query(self.text) or not self._selection_is_at_end():
            return False
        suffix = f" {character}" if character is not None else " "
        self.load_text(f"{self.text}{suffix}")
        self.move_cursor((0, len(self.text)), record_width=False)
        return True

    async def handle_enter_key(self) -> None:
        """Select slash completions, submit built-ins, or insert a newline."""
        if self.slash_complete_active:
            _ = self.post_message(self.SlashAccept(self))
            return
        if should_submit_on_enter(self.text):
            app = cast(_PromptSubmitApp, cast(object, self.app))
            await app.action_submit()
            return
        start, end = self.selection
        result = self.replace("\n", start, end)
        self.move_cursor(result.end_location)

    @override
    async def _on_key(self, event: events.Key) -> None:
        """Handle Enter/Escape before TextArea inserts characters."""
        if event.key == "enter":
            _ = event.stop()
            _ = event.prevent_default()
            await self.handle_enter_key()
            return
        if event.key == "tab":
            if self.slash_complete_active:
                _ = event.stop()
                _ = event.prevent_default()
                _ = self.post_message(self.SlashAccept(self))
                return
            if self._expand_models_query_prefix():
                _ = event.stop()
                _ = event.prevent_default()
                return
        if self.model_complete_active and event.key == "escape":
            _ = event.stop()
            _ = event.prevent_default()
            _ = self.post_message(self.ModelDismiss(self))
            return
        if self.slash_complete_active and event.key == "escape":
            _ = event.stop()
            _ = event.prevent_default()
            _ = self.post_message(self.SlashDismiss(self))
            return
        if event.character is not None and not event.character.isspace():
            if self._expand_models_query_prefix(event.character):
                _ = event.stop()
                _ = event.prevent_default()
                return
        await super()._on_key(event)

    @override
    def action_cursor_up(self, select: bool = False) -> None:
        if self.model_complete_active and not select:
            _ = self.post_message(self.ModelNavigate(self, -1))
            return
        if self.slash_complete_active and not select:
            _ = self.post_message(self.SlashNavigate(self, -1))
            return
        super().action_cursor_up(select=select)

    @override
    def action_cursor_down(self, select: bool = False) -> None:
        if self.model_complete_active and not select:
            _ = self.post_message(self.ModelNavigate(self, 1))
            return
        if self.slash_complete_active and not select:
            _ = self.post_message(self.SlashNavigate(self, 1))
            return
        super().action_cursor_down(select=select)


@final
class SlashComplete(OptionList):
    """Option list showing filtered slash command completions."""

    def __init__(self, commands: list[SlashCommand] | tuple[SlashCommand, ...]) -> None:
        super().__init__(id="slash-complete")
        self.commands: list[SlashCommand] = list(commands)
        self.matches: list[SlashCommand] = []

    def update_query(self, query: str) -> bool:
        """Refresh visible matches for the current slash query."""
        self.matches = filter_slash_commands(self.commands, query)
        _ = self.clear_options()
        if not self.matches:
            self.highlighted = None
            return False
        _ = self.add_options(
            Option(f"{command.command} — {command.help}", id=command.command)
            for command in self.matches
        )
        self.highlighted = 0
        return True

    def highlighted_command(self) -> SlashCommand | None:
        """Return the currently highlighted slash command, if any."""
        highlighted = self.highlighted
        if highlighted is None:
            return None
        try:
            return self.matches[highlighted]
        except IndexError:
            return None


@final
class ModelComplete(OptionList):
    """Option list showing filtered model completions for ``/models``."""

    def __init__(self) -> None:
        super().__init__(id="model-complete")
        self.matches: list[tuple[str, str]] = []

    def update_query(self, query: str, current_model: str) -> bool:
        """Refresh visible matches for the current model query."""
        self.matches = filter_available_models(query, get_available_models())
        _ = self.clear_options()
        if not self.matches:
            self.highlighted = None
            return False
        _ = self.add_options(
            Option(
                f"★ {label}" if model_id == current_model else label,
                id=model_id,
            )
            for model_id, label in self.matches
        )
        self.highlighted = 0
        return True

    def highlighted_model_id(self) -> str | None:
        """Return the currently highlighted model id, if any."""
        highlighted = self.highlighted
        if highlighted is None:
            return None
        try:
            return self.matches[highlighted][0]
        except IndexError:
            return None


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
        self._streaming: bool = False
        super().__init__("")
        self.soft_wrap = True
        self.set_text(text)

    def has_content(self) -> bool:
        """Return whether the widget currently contains any thinking text."""
        return bool(self._raw.strip())

    def start_streaming(self) -> None:
        """Show the full thinking text while it is still streaming in."""
        self._streaming = True
        self._refresh_rendered_text()

    def finish_streaming(self) -> None:
        """Collapse streamed thinking back to the preview once it is complete."""
        self._streaming = False
        self._expanded = False
        self._refresh_rendered_text()

    def _preview_text(self) -> str:
        lines = self._raw.splitlines()
        if len(lines) <= self.PREVIEW_LINES:
            return self._raw
        preview = "\n".join(lines[: self.PREVIEW_LINES])
        remaining = len(lines) - self.PREVIEW_LINES
        return f"{preview}\n… {remaining} more lines. Press Ctrl+O for rest."

    def _render_text(self) -> str:
        if self._streaming or self._expanded:
            return self._raw
        return self._preview_text()

    def _refresh_rendered_text(self) -> None:
        rendered = self._render_text()
        self.text = rendered
        self._sync_height(rendered)
        if self._streaming and rendered:
            lines = rendered.split("\n")
            self.move_cursor((len(lines) - 1, len(lines[-1])), record_width=False)
            try:
                _ = self.scroll_cursor_visible(animate=False)
            except Exception:
                pass

    @override
    def set_text(self, text: str) -> None:
        """Update raw thinking text and render preview/full content."""
        self._raw = text
        visible = self.has_content()
        self.display = visible
        parent = self.parent
        if parent is not None and parent.has_class("thinking-section"):
            parent.display = visible
        self._refresh_rendered_text()

    def action_toggle_expanded(self) -> None:
        """Toggle between the preview and the full reasoning text."""
        if (
            self._streaming
            or not self.has_content()
            or len(self._raw.splitlines()) <= self.PREVIEW_LINES
        ):
            return
        self._expanded = not self._expanded
        self._refresh_rendered_text()


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
        auto_scroll_enabled: bool = True,
    ) -> None:
        super().__init__(
            self.format_status(
                model_name,
                agent_mode,
                context_tokens,
                auto_scroll_enabled,
            ),
            id="status-line",
            markup=False,
        )

    @staticmethod
    def format_status(
        model_name: str,
        agent_mode: bool,
        context_tokens: int | None,
        auto_scroll_enabled: bool = True,
    ) -> str:
        """Format the text displayed in the status line."""
        model = model_name or "?"
        agent = "on" if agent_mode else "off"
        if context_tokens is None:
            context = "?"
        elif context_tokens >= 1000:
            context = f"{context_tokens / 1000:.1f}k"
        else:
            context = str(context_tokens)
        auto_scroll = "auto" if auto_scroll_enabled else "manual"
        return f"{model} · {agent} · {context} · {auto_scroll}"

    def set_status(
        self,
        *,
        model_name: str,
        agent_mode: bool,
        context_tokens: int | None,
        auto_scroll_enabled: bool,
    ) -> None:
        """Update the displayed model, agent mode, context size, and follow mode."""
        self.update(
            self.format_status(
                model_name,
                agent_mode,
                context_tokens,
                auto_scroll_enabled,
            )
        )


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
