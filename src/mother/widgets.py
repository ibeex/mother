"""Reusable TUI widget classes for Mother."""

from dataclasses import dataclass
from typing import ClassVar, Protocol, cast, final, override

import pyperclip
from textual import events
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Label, Markdown, OptionList, Static, TextArea
from textual.widgets.markdown import MarkdownBlock, MarkdownFence, MarkdownStream
from textual.widgets.option_list import Option

from mother.clipboard import read_clipboard_text
from mother.slash_commands import (
    SlashArgumentChoice,
    SlashCommand,
    filter_slash_commands,
    should_expand_slash_argument,
)
from mother.user_commands import should_submit_on_enter


class _PromptSubmitApp(Protocol):
    """Subset of app API used by the prompt input widget."""

    async def action_submit(self) -> None: ...


class _PromptClipboardApp(Protocol):
    """Subset of app API used for clipboard image paste support."""

    clipboard: str

    def capture_clipboard_image(self) -> str | None: ...


class _PromptInterruptApp(Protocol):
    """Subset of app API used for double-Escape interruption."""

    def handle_interrupt_escape(self) -> bool: ...


class _PromptHistoryApp(Protocol):
    """Subset of app API used for prompt-history navigation/search."""

    def action_prompt_history_previous(self) -> None: ...

    def action_prompt_history_next(self) -> None: ...

    def action_prompt_history_search(self) -> None: ...


class Prompt(Markdown):
    """Markdown for the user prompt."""

    BORDER_TITLE: ClassVar[str] = "You"


class PromptTextArea(TextArea):
    """Main prompt input with slash-complete key handling."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+r", "history_search", show=False),
    ]

    slash_complete_active: bool = False
    slash_argument_complete_active: bool = False

    @property
    def model_complete_active(self) -> bool:
        """Backward-compatible alias for generic slash-argument completion state."""
        return self.slash_argument_complete_active

    @model_complete_active.setter
    def model_complete_active(self, value: bool) -> None:
        self.slash_argument_complete_active = value

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
    class SlashArgumentNavigate(Message):
        """Request that the slash-argument popup move its highlight."""

        text_area: "PromptTextArea"
        direction: int

        @property
        @override
        def control(self) -> "PromptTextArea":
            return self.text_area

    @dataclass
    class SlashArgumentDismiss(Message):
        """Request that the slash-argument popup close."""

        text_area: "PromptTextArea"

        @property
        @override
        def control(self) -> "PromptTextArea":
            return self.text_area

    @dataclass
    class SlashArgumentAccept(Message):
        """Request that the slash-argument popup accept its current selection."""

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

    ModelNavigate: type[SlashArgumentNavigate] = SlashArgumentNavigate
    ModelDismiss: type[SlashArgumentDismiss] = SlashArgumentDismiss
    ModelAccept: type[SlashArgumentAccept] = SlashArgumentAccept

    def _selection_is_at_end(self) -> bool:
        start, end = self.selection
        return start == end == (0, len(self.text))

    def _expand_slash_argument_prefix(self, character: str | None = None) -> bool:
        """Expand a slash command with inline argument completion to include a space."""
        if not should_expand_slash_argument(self.text) or not self._selection_is_at_end():
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
        if self.slash_argument_complete_active:
            _ = self.post_message(self.SlashArgumentAccept(self))
            return
        if should_submit_on_enter(self.text):
            app = cast(_PromptSubmitApp, cast(object, self.app))
            await app.action_submit()
            return
        start, end = self.selection
        result = self.replace("\n", start, end)
        self.move_cursor(result.end_location)

    @override
    def action_paste(self) -> None:
        """Prefer clipboard images over plain-text paste when Ctrl+V is used."""
        if self.read_only:
            return
        app = cast(_PromptClipboardApp, cast(object, self.app))
        image_path = app.capture_clipboard_image()
        if image_path is not None:
            if result := self._replace_via_keyboard(image_path, *self.selection):
                self.move_cursor(result.end_location)
            return

        clipboard_text = read_clipboard_text()
        if clipboard_text is not None:
            if result := self._replace_via_keyboard(clipboard_text, *self.selection):
                self.move_cursor(result.end_location)
            return

        super().action_paste()

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
            if self.slash_argument_complete_active:
                _ = event.stop()
                _ = event.prevent_default()
                _ = self.post_message(self.SlashArgumentAccept(self))
                return
            if self._expand_slash_argument_prefix():
                _ = event.stop()
                _ = event.prevent_default()
                return
        if self.slash_argument_complete_active and event.key == "escape":
            _ = event.stop()
            _ = event.prevent_default()
            _ = self.post_message(self.SlashArgumentDismiss(self))
            return
        if self.slash_complete_active and event.key == "escape":
            _ = event.stop()
            _ = event.prevent_default()
            _ = self.post_message(self.SlashDismiss(self))
            return
        if event.key == "escape":
            app = cast(_PromptInterruptApp, cast(object, self.app))
            if app.handle_interrupt_escape():
                _ = event.stop()
                _ = event.prevent_default()
                return
        if (
            event.character is not None
            and event.character.isprintable()
            and not event.character.isspace()
        ):
            if self._expand_slash_argument_prefix(event.character):
                _ = event.stop()
                _ = event.prevent_default()
                return
        await super()._on_key(event)

    def action_history_previous(self) -> None:
        """Recall the previous submitted prompt from persistent history."""
        app = cast(_PromptHistoryApp, cast(object, self.app))
        app.action_prompt_history_previous()

    def action_history_next(self) -> None:
        """Move toward newer prompt-history entries or restore the current draft."""
        app = cast(_PromptHistoryApp, cast(object, self.app))
        app.action_prompt_history_next()

    def action_history_search(self) -> None:
        """Search backward through prompt history using the current draft as a query."""
        app = cast(_PromptHistoryApp, cast(object, self.app))
        app.action_prompt_history_search()

    @override
    def action_cursor_up(self, select: bool = False) -> None:
        if self.slash_argument_complete_active and not select:
            _ = self.post_message(self.SlashArgumentNavigate(self, -1))
            return
        if self.slash_complete_active and not select:
            _ = self.post_message(self.SlashNavigate(self, -1))
            return
        start, end = self.selection
        if not select and start == end and start[0] == 0:
            self.action_history_previous()
            return
        super().action_cursor_up(select=select)

    @override
    def action_cursor_down(self, select: bool = False) -> None:
        if self.slash_argument_complete_active and not select:
            _ = self.post_message(self.SlashArgumentNavigate(self, 1))
            return
        if self.slash_complete_active and not select:
            _ = self.post_message(self.SlashNavigate(self, 1))
            return
        start, end = self.selection
        last_row = len(self.text.split("\n")) - 1
        if not select and start == end and start[0] == last_row:
            self.action_history_next()
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
class SlashArgumentComplete(OptionList):
    """Option list showing filtered inline argument completions for slash commands."""

    def __init__(self) -> None:
        super().__init__(id="slash-argument-complete")
        self.matches: list[SlashArgumentChoice] = []

    def update_matches(
        self,
        matches: list[SlashArgumentChoice],
        current_value: str | None = None,
    ) -> bool:
        """Refresh visible matches for the active slash-argument query."""
        self.matches = matches
        _ = self.clear_options()
        if not self.matches:
            self.highlighted = None
            return False
        _ = self.add_options(
            Option(
                f"★ {choice.label}" if choice.value == current_value else choice.label,
                id=choice.value,
            )
            for choice in self.matches
        )
        self.highlighted = 0
        return True

    def highlighted_value(self) -> str | None:
        """Return the currently highlighted slash-argument value, if any."""
        highlighted = self.highlighted
        if highlighted is None:
            return None
        try:
            return self.matches[highlighted].value
        except IndexError:
            return None


ModelComplete = SlashArgumentComplete


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
    """Plain-text widget for structured model reasoning from pydantic-ai thinking parts."""

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
        """Return whether the widget currently contains any structured thinking text."""
        return bool(self._raw.strip())

    def start_streaming(self) -> None:
        """Show the full structured thinking text while it is still streaming in."""
        self._streaming = True
        self._refresh_rendered_text()

    def finish_streaming(self) -> None:
        """Collapse streamed structured thinking back to the preview once it is complete."""
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
        """Update raw structured thinking text and render preview/full content."""
        self._raw = text
        visible = self.has_content()
        self.display = visible
        parent = self.parent
        if parent is not None and parent.has_class("thinking-section"):
            parent.display = visible
        self._refresh_rendered_text()

    def action_toggle_expanded(self) -> None:
        """Toggle between the preview and the full structured reasoning text."""
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
        self._stream: MarkdownStream | None = None

    @property
    def raw_markdown(self) -> str:
        """Return the full logical markdown value, excluding transient placeholder frames."""
        return self._raw

    @property
    def stream(self) -> MarkdownStream:
        """Return the streaming helper used for incremental markdown appends."""
        if self._stream is None:
            self._stream = self.get_stream(self)
        return self._stream

    async def append_fragment(self, fragment: str) -> None:
        """Append a streamed markdown fragment without reparsing the full response."""
        if not fragment:
            return
        self._raw += fragment
        await self.stream.write(fragment)

    async def replace_markdown(self, markdown: str) -> None:
        """Replace the rendered markdown, stopping any active incremental stream first."""
        await self.stop_stream()
        self._raw = markdown
        await self.update(markdown)

    async def stop_stream(self) -> None:
        """Stop the active incremental markdown stream, flushing any pending fragments."""
        if self._stream is None:
            return
        stream = self._stream
        self._stream = None
        await stream.stop()

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
    """Compact label used inside conversation and output sections."""

    def __init__(self, text: str, *, classes: str) -> None:
        super().__init__(text, markup=False, classes=classes)


@final
class WelcomeBanner(Static):
    """UI-only startup greeting shown in the chat pane."""

    DEFAULT_TEXT: ClassVar[str] = (
        " __  __  ____  _______ _    _ ______ _____  \n"
        "|  \\/  |/ __ \\|__   __| |  | |  ____|  __ \\ \n"
        "| \\  / | |  | |  | |  | |__| | |__  | |__) |\n"
        "| |\\/| | |  | |  | |  |  __  |  __| |  _  / \n"
        "| |  | | |__| |  | |  | |  | | |____| | \\ \\ \n"
        "|_|  |_|\\____/   |_|  |_|  |_|______|_|  \\_\\\n"
        "\n"
        "INTERFACE 2037 READY\n"
        "MU-TH-UR 6000 SYSTEM"
    )

    def __init__(self, text: str | None = None) -> None:
        super().__init__(text or self.DEFAULT_TEXT, markup=False, id="welcome-banner")


class ConversationTurn(Vertical):
    """A compact prompt / thinking / response block."""

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
        children: list[Vertical | Horizontal] = []

        self.prompt_widget: Prompt | None = None
        if prompt_text is not None:
            self.prompt_widget = Prompt(prompt_text)
            _ = self.prompt_widget.add_class("grouped")
            children.append(
                Horizontal(
                    TurnLabel(">", classes="turn-gutter prompt-gutter"),
                    self.prompt_widget,
                    classes="turn-section prompt-section",
                )
            )

        self.thinking_output: ThinkingOutput | None = None
        self.thinking_section: Horizontal | None = None
        if include_thinking:
            self.thinking_output = ThinkingOutput()
            _ = self.thinking_output.add_class("grouped")
            self.thinking_output.set_text("")
            self.thinking_section = Horizontal(
                TurnLabel("…", classes="turn-gutter thinking-gutter"),
                self.thinking_output,
                classes="turn-section thinking-section",
            )
            self.thinking_section.display = False
            children.append(self.thinking_section)

        self.tool_trace_stack: Vertical = Vertical(classes="tool-trace-stack")
        self.tool_trace_stack.display = False
        children.append(self.tool_trace_stack)

        self.response_widget: Response = Response(response_text)
        _ = self.response_widget.add_class("grouped")
        children.append(
            Vertical(
                self.response_widget,
                classes="turn-section response-section",
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
        reasoning_effort: str | None = None,
        last_response_time_seconds: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_tokens: int | None = None,
        agent_label: str | None = None,
    ) -> None:
        super().__init__(
            self.format_status(
                model_name,
                agent_mode,
                context_tokens,
                auto_scroll_enabled,
                reasoning_effort,
                last_response_time_seconds,
                input_tokens,
                output_tokens,
                cached_tokens,
                agent_label,
            ),
            id="status-line",
            markup=False,
        )

    @staticmethod
    def format_response_time(seconds: float | None) -> str | None:
        """Format the last completed model-response duration for the status line."""
        if seconds is None:
            return None
        if seconds < 60:
            return f"{seconds:.1f}s"
        whole_seconds = round(seconds)
        minutes, remaining_seconds = divmod(whole_seconds, 60)
        if minutes < 60:
            return f"{minutes}m {remaining_seconds}s"
        hours, remaining_minutes = divmod(minutes, 60)
        return f"{hours}h {remaining_minutes}m {remaining_seconds}s"

    @staticmethod
    def format_token_count(tokens: int | None) -> str | None:
        """Format a token count for compact display in the status line."""
        if tokens is None:
            return None
        if tokens >= 1000:
            return f"{tokens / 1000:.1f}k"
        return str(tokens)

    @staticmethod
    def format_segment(label: str, value: str) -> str:
        """Format a labeled status-line segment."""
        return f"{label}:{value}"

    @staticmethod
    def format_optional_segment(label: str, value: str | None) -> str:
        """Format an optional labeled status-line segment."""
        if value is None:
            return ""
        return f" · {StatusLine.format_segment(label, value)}"

    @staticmethod
    def format_token_summary(
        input_tokens: int | None,
        output_tokens: int | None,
        cached_tokens: int | None,
    ) -> str | None:
        """Format token usage as input/output/cache."""
        if input_tokens is None and output_tokens is None and cached_tokens is None:
            return None
        return "/".join(
            [
                StatusLine.format_token_count(input_tokens) or "?",
                StatusLine.format_token_count(output_tokens) or "?",
                StatusLine.format_token_count(cached_tokens) or "?",
            ]
        )

    @staticmethod
    def format_status(
        model_name: str,
        agent_mode: bool,
        context_tokens: int | None,
        auto_scroll_enabled: bool = True,
        reasoning_effort: str | None = None,
        last_response_time_seconds: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_tokens: int | None = None,
        agent_label: str | None = None,
    ) -> str:
        """Format the text displayed in the status line."""
        model_segment = model_name or "?"
        resolved_agent_label = agent_label or ("on" if agent_mode else "off")
        agent_segment = StatusLine.format_segment("A", resolved_agent_label)
        context_segment = StatusLine.format_segment(
            "C", StatusLine.format_token_count(context_tokens) or "?"
        )
        token_segment = StatusLine.format_optional_segment(
            "Tok", StatusLine.format_token_summary(input_tokens, output_tokens, cached_tokens)
        )
        manual_segment = " · Man" if not auto_scroll_enabled else ""
        reasoning_segment = StatusLine.format_optional_segment("R", reasoning_effort)
        response_time = StatusLine.format_response_time(last_response_time_seconds)
        response_time_segment = f" · {response_time}" if response_time is not None else ""
        return (
            f"{model_segment} · {agent_segment} · {context_segment}"
            f"{token_segment}"
            f"{manual_segment}"
            f"{reasoning_segment}"
            f"{response_time_segment}"
        )

    def set_status(
        self,
        *,
        model_name: str,
        agent_mode: bool,
        context_tokens: int | None,
        auto_scroll_enabled: bool,
        reasoning_effort: str | None,
        last_response_time_seconds: float | None,
        input_tokens: int | None,
        output_tokens: int | None,
        cached_tokens: int | None,
        agent_label: str | None = None,
    ) -> None:
        """Update the displayed model, context, token usage, follow mode, reasoning, and last response time."""
        self.update(
            self.format_status(
                model_name,
                agent_mode,
                context_tokens,
                auto_scroll_enabled,
                reasoning_effort,
                last_response_time_seconds,
                input_tokens,
                output_tokens,
                cached_tokens,
                agent_label,
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
