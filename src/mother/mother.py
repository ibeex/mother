"""Mother TUI chatbot — a Textual interface for chatting with an LLM."""

from collections.abc import AsyncGenerator
from dataclasses import replace
from datetime import datetime
from typing import ClassVar, cast, override

import click
import llm
import pyperclip
from llm.models import Conversation, Model, Tool, ToolCall, ToolResult
from textual import work
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.command import Hit, Provider
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Markdown, OptionList, TextArea
from textual.widgets.markdown import MarkdownBlock, MarkdownFence
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerState

from mother.bash_execution import BashExecution, format_for_context
from mother.config import MotherConfig, apply_cli_overrides, load_config
from mother.tool_trace import format_tool_event
from mother.tools import get_default_tools
from mother.tools.bash_executor import execute_bash
from mother.user_commands import ShellCommand, parse_user_input


def get_available_models() -> list[tuple[str, str]]:
    """Return available models for the picker.

    Prefer custom models configured via ``extra-openai-models.yaml`` (they have
    a custom ``api_base``). If none are configured, fall back to all models.
    """
    all_models = list(llm.get_models())
    preferred_models = [model for model in all_models if getattr(model, "api_base", None)]
    source_models = preferred_models or all_models

    seen: set[str] = set()
    available_models: list[tuple[str, str]] = []
    for model in source_models:
        model_id = model.model_id
        if model_id in seen:
            continue
        seen.add(model_id)
        model_name = getattr(model, "model_name", None)
        label = f"{model_id} — {model_name}" if model_name else model_id
        available_models.append((model_id, label))
    return available_models


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


class AgentModeProvider(Provider):
    """Command palette provider for toggling agent mode."""

    @override
    async def search(self, query: str) -> AsyncGenerator[Hit, None]:
        app = cast("MotherApp", self.app)
        matcher = self.matcher(query)
        label = "Agent mode: off" if app.agent_mode else "Agent mode: on"
        score = matcher.match(label)
        if score > 0 or not query:
            yield Hit(
                score or 1.0,
                matcher.highlight(label),
                app.action_toggle_agent_mode,
                help="Toggle agent mode (tool use)",
            )


class ModelProvider(Provider):
    """Command palette provider for opening the model picker."""

    @override
    async def search(self, query: str) -> AsyncGenerator[Hit, None]:
        app = cast("MotherApp", self.app)
        matcher = self.matcher(query)
        label = "Models"
        score = matcher.match(label)
        if score > 0 or not query:
            yield Hit(
                score or 1.0,
                matcher.highlight(label),
                app.action_show_models,
                help=f"Browse and switch models (current: {app.config.model})",
            )


class ModelPickerScreen(ModalScreen[None]):
    """Modal screen for searching and selecting available models."""

    CSS: ClassVar[str] = """
    ModelPickerScreen {
        align: center middle;
        background: $background 60%;
    }

    #model-picker {
        width: 72;
        height: 24;
        border: round $primary;
        background: $surface;
        padding: 1;
    }

    #model-query {
        margin-bottom: 1;
    }

    #model-options {
        height: 1fr;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [("escape", "dismiss", "Close")]

    def __init__(self, current_model: str) -> None:
        super().__init__()
        self.current_model: str = current_model
        self._all_models: list[tuple[str, str]] = get_available_models()

    @override
    def compose(self) -> ComposeResult:
        with Container(id="model-picker"):
            with Vertical():
                yield Input(placeholder="Search models...", id="model-query")
                yield OptionList(id="model-options")

    def on_mount(self) -> None:
        self._refresh_options("")
        _ = self.query_one(Input).focus()

    def _refresh_options(self, query: str) -> None:
        option_list = self.query_one(OptionList)
        normalized_query = query.strip().lower()
        matching_models = [
            (model_id, label)
            for model_id, label in self._all_models
            if not normalized_query
            or normalized_query in model_id.lower()
            or normalized_query in label.lower()
        ]
        _ = option_list.clear_options()
        if not matching_models:
            _ = option_list.add_option(Option("No models found", disabled=True))
            option_list.highlighted = None
            return
        _ = option_list.add_options(
            Option(
                f"★ {label}" if model_id == self.current_model else label,
                id=model_id,
            )
            for model_id, label in matching_models
        )
        option_list.highlighted = 0

    def _select_highlighted_model(self) -> None:
        option_list = self.query_one(OptionList)
        if option_list.highlighted is None:
            return
        option = option_list.get_option_at_index(option_list.highlighted)
        if option.id is None:
            return
        cast("MotherApp", self.app).action_switch_model(option.id)
        _ = self.dismiss()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "model-query":
            self._refresh_options(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "model-query":
            self._select_highlighted_model()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "model-options" and event.option.id is not None:
            cast("MotherApp", self.app).action_switch_model(event.option.id)
            _ = self.dismiss()


class MotherApp(App[None]):
    """Simple app for chatting with an LLM via a conversation."""

    AUTO_FOCUS: ClassVar[str | None] = "TextArea"

    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+enter", "submit", "Send"),
    ]

    COMMANDS = App.COMMANDS | {AgentModeProvider, ModelProvider}  # pyright: ignore[reportUnannotatedClassAttribute]

    CSS: ClassVar[str] = """
    Prompt {
        background: $primary 10%;
        color: $text;
        margin: 1;
        margin-right: 8;
        padding: 1 2 0 2;
    }

    Response {
        border: wide $success;
        background: $success 10%;
        color: $text;
        margin: 1;
        margin-left: 8;
        padding: 1 2 0 2;
    }

    Response:focus {
        border: wide $warning;
    }

    MarkdownBlock.highlight {
        background: $accent 20%;
    }

    ShellOutput {
        height: auto;
        border: wide $accent;
        background: $accent 10%;
        color: $text;
        margin: 1;
        margin-right: 4;
        margin-left: 4;
        padding: 1 2 0 2;
    }

    ShellOutput:focus {
        border: wide $warning;
    }

    ToolOutput {
        height: auto;
        border: wide $secondary;
        background: $secondary 10%;
        color: $text;
        margin: 1;
        margin-right: 4;
        margin-left: 4;
        padding: 1 2 0 2;
    }

    ToolOutput:focus {
        border: wide $warning;
    }

    TextArea {
        height: 6;
        border: tall $primary;
    }

    #model-picker {
        width: 72;
        height: 24;
    }
    """

    def __init__(
        self,
        config: MotherConfig | None = None,
        model_name: str | None = None,
        system: str | None = None,
    ) -> None:
        super().__init__()
        base = config or MotherConfig()
        self.config: MotherConfig = apply_cli_overrides(base, model_name, system)
        self.agent_mode: bool = self.config.tools_enabled
        self.model: Model | None = None
        self.conversation: Conversation | None = None
        self._pending_executions: list[BashExecution] = []
        self._tool_outputs: dict[str, ToolOutput] = {}

    @override
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="chat-view"):
            yield Response("INTERFACE 2037 READY FOR INQUIRY")
        yield TextArea()
        yield Footer()

    def on_mount(self) -> None:
        self.model = llm.get_model(self.config.model)
        self.conversation = self.model.conversation()
        _ = self.query_one("#chat-view").anchor()
        self._update_subtitle()

    def action_show_models(self) -> None:
        """Open the model picker."""
        _ = self.push_screen(ModelPickerScreen(self.config.model))

    def action_switch_model(self, model_id: str) -> None:
        """Switch to a different LLM model and start a fresh conversation."""
        self.config = replace(self.config, model=model_id, tools_enabled=self.agent_mode)
        self.model = llm.get_model(model_id)
        self.conversation = self.model.conversation()
        self._update_subtitle()
        self.notify(f"Switched to {model_id}", title="Model changed")

    def action_toggle_agent_mode(self) -> None:
        """Toggle agent mode (enables/disables tool use)."""
        self.agent_mode = not self.agent_mode
        self._update_subtitle()
        state = "enabled" if self.agent_mode else "disabled"
        self.notify(f"Agent mode {state}", title="Agent mode")

    def _update_subtitle(self) -> None:
        """Update subtitle to show model and agent mode indicator."""
        sub_title: str = f"{self.config.model} [AGENT]" if self.agent_mode else self.config.model
        self.sub_title = sub_title  # pyright: ignore[reportUnannotatedClassAttribute]

    async def action_submit(self) -> None:
        """When the user hits Ctrl+Enter."""
        text_area = self.query_one(TextArea)
        value = text_area.text.strip()
        if not value:
            return
        _ = text_area.clear()

        parsed = parse_user_input(value)
        if isinstance(parsed, ShellCommand):
            await self.run_user_command(parsed)
            return

        chat_view = self.query_one("#chat-view")
        text_area.read_only = True

        # Flush any pending shell executions as context before the LLM prompt
        context_parts: list[str] = []
        for execution in self._pending_executions:
            if not execution.exclude_from_context:
                context_parts.append(format_for_context(execution))
        self._pending_executions.clear()

        prompt = value
        if context_parts:
            prompt = "\n\n".join(context_parts) + "\n\n" + value

        _ = await chat_view.mount(Prompt(value))
        _ = await chat_view.mount(response := Response())
        _ = self.send_prompt(prompt, response)

    async def run_user_command(self, cmd: ShellCommand) -> None:
        """Execute a direct user shell command and display the output."""
        chat_view = self.query_one("#chat-view")
        shell_widget = ShellOutput(f"Running: {cmd.command}")
        _ = await chat_view.mount(shell_widget)

        try:
            result = await execute_bash(cmd.command)
        except Exception as exc:
            output = f"Error: {exc}"
            exit_code = None
        else:
            output = result.output
            exit_code = result.exit_code

        shell_widget.set_text(output)

        execution = BashExecution(
            command=cmd.command,
            output=output,
            exit_code=exit_code,
            timestamp=datetime.now(),
            exclude_from_context=not cmd.include_in_context,
        )
        self._pending_executions.append(execution)

    def _tool_output_key(
        self,
        tool_name: str,
        tool_call_id: str | None,
        arguments: dict[str, object],
    ) -> str:
        """Build a stable key for a tool execution trace widget."""
        if tool_call_id:
            return tool_call_id
        command = arguments.get("command")
        if isinstance(command, str) and command:
            return f"{tool_name}:{command}"
        return tool_name

    def _show_tool_started(
        self,
        tool_name: str,
        tool_call_id: str | None,
        arguments: dict[str, object],
    ) -> None:
        """Mount a widget showing that a tool call has started."""
        key = self._tool_output_key(tool_name, tool_call_id, arguments)
        widget = ToolOutput(format_tool_event(tool_name, arguments, status="started"))
        self._tool_outputs[key] = widget
        chat_view = self.query_one("#chat-view", VerticalScroll)
        _ = chat_view.mount(widget)
        _ = chat_view.scroll_end(animate=False)

    def _show_tool_finished(
        self,
        tool_name: str,
        tool_call_id: str | None,
        arguments: dict[str, object],
        output: str,
    ) -> None:
        """Update a tool trace widget when a tool call finishes."""
        key = self._tool_output_key(tool_name, tool_call_id, arguments)
        widget = self._tool_outputs.pop(key, None)
        chat_view = self.query_one("#chat-view", VerticalScroll)
        if widget is None:
            widget = ToolOutput()
            _ = chat_view.mount(widget)
        widget.set_text(format_tool_event(tool_name, arguments, status="finished", output=output))
        _ = chat_view.scroll_end(animate=False)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Re-enable input when the worker finishes."""
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            text_area = self.query_one(TextArea)
            text_area.read_only = False
            _ = text_area.focus()

    @work(thread=True)
    def send_prompt(self, prompt: str, response: Response) -> None:
        """Get the response in a thread, maintaining conversation history."""
        _ = self.call_from_thread(response.update, "*Thinking...*")
        conversation = self.conversation
        if conversation is None:
            error_text = "**Error:** Conversation is not initialized."
            _ = self.call_from_thread(response.update, error_text)
            response.reset_state(error_text)
            return
        tool_registry = get_default_tools(
            tools_enabled=self.agent_mode, allowlist=self.config.allowlist
        )
        tools = tool_registry.tools() if not tool_registry.is_empty() else None
        system = self.config.system_prompt

        def before_tool_call(tool: Tool | None, tool_call: ToolCall) -> None:
            tool_name = tool.name if tool is not None else tool_call.name
            arguments: dict[str, object] = cast(dict[str, object], dict(tool_call.arguments))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            self.call_from_thread(
                self._show_tool_started,
                tool_name,
                tool_call.tool_call_id,
                arguments,
            )

        def after_tool_call(tool: Tool, tool_call: ToolCall, tool_result: ToolResult) -> None:
            arguments: dict[str, object] = cast(dict[str, object], dict(tool_call.arguments))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            self.call_from_thread(
                self._show_tool_finished,
                tool.name,
                tool_call.tool_call_id,
                arguments,
                tool_result.output,
            )

        try:
            if tools is not None:
                llm_response = conversation.chain(  # pyright: ignore[reportUnknownMemberType]
                    prompt,
                    system=system,
                    tools=tools,
                    before_call=before_tool_call,
                    after_call=after_tool_call,
                )
            else:
                llm_response = conversation.prompt(prompt, system=system)  # pyright: ignore[reportUnknownMemberType]
            full_text = ""
            for chunk in llm_response:
                full_text += chunk
                _ = self.call_from_thread(response.update, full_text)
        except Exception as exc:
            if "does not support tools" in str(exc) and tools is not None:
                _ = self.call_from_thread(self._disable_agent_mode_unsupported)
                try:
                    llm_response = conversation.prompt(prompt, system=system)  # pyright: ignore[reportUnknownMemberType]
                    full_text = ""
                    for chunk in llm_response:
                        full_text += chunk
                        _ = self.call_from_thread(response.update, full_text)
                except Exception as exc2:
                    error_text = f"**Error:** {exc2}"
                    _ = self.call_from_thread(response.update, error_text)
                    response.reset_state(error_text)
                    return
            else:
                error_text = f"**Error:** {exc}"
                _ = self.call_from_thread(response.update, error_text)
                response.reset_state(error_text)
                return
        response.reset_state(full_text)

    def _disable_agent_mode_unsupported(self) -> None:
        """Disable agent mode and notify user that the model doesn't support tools."""
        self.agent_mode = False
        self._update_subtitle()
        self.notify(
            f"{self.config.model} does not support tools — agent mode disabled",
            title="Agent mode",
            severity="warning",
        )


@click.command()
@click.option("--model", "-m", default=None, help="LLM model to use.")
@click.option("--system", "-s", default=None, help="System prompt.")
def cli(model: str | None, system: str | None) -> None:
    """Launch the Mother TUI chatbot."""
    config = load_config()
    config = apply_cli_overrides(config, model, system)
    app = MotherApp(config=config)
    app.run()
