"""Mother TUI chatbot — a Textual interface for chatting with an LLM."""

from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import datetime
from pathlib import Path, PurePath
from typing import ClassVar, cast, override

import click
import llm
from llm.models import (
    AfterCallSync,
    BeforeCallSync,
    Conversation,
    Model,
    Tool,
    ToolCall,
    ToolDef,
    ToolResult,
)
from textual import work
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import BindingType
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Footer, Header, TextArea
from textual.worker import Worker, WorkerState

from mother.bash_execution import BashExecution, format_for_context
from mother.config import MotherConfig, apply_cli_overrides, load_config
from mother.model_picker import AgentModeProvider, ModelPickerScreen, ModelProvider
from mother.session import SessionManager
from mother.system_prompt import build_system_prompt
from mother.thinking import ThinkTagStreamParser
from mother.tool_trace import format_tool_event
from mother.tools import get_default_tools
from mother.tools.bash_executor import execute_bash
from mother.user_commands import SaveSessionCommand, ShellCommand, parse_user_input
from mother.widgets import (
    ConversationTurn,
    OutputSection,
    Response,
    ShellOutput,
    StatusLine,
    ThinkingOutput,
    ToolOutput,
)

CSS_DIR = Path(__file__).resolve().parent / "css"
APP_CSS_PATHS: list[str | PurePath] = [
    CSS_DIR / "chat.tcss",
    CSS_DIR / "output.tcss",
    CSS_DIR / "input.tcss",
]


class MotherApp(App[None]):
    """Simple app for chatting with an LLM via a conversation."""

    AUTO_FOCUS: ClassVar[str | None] = "TextArea"

    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+enter", "submit", "Send"),
        ("ctrl+o", "toggle_thinking_widget", "Thoughts"),
        ("ctrl+g", "toggle_auto_scroll", "Autoscroll"),
        ("end", "scroll_to_bottom", "Bottom"),
        ("shift+g", "scroll_to_bottom_from_chat", "Bottom"),
        ("ctrl+s", "save_session", "Save"),
    ]

    COMMANDS = App.COMMANDS | {AgentModeProvider, ModelProvider}  # pyright: ignore[reportUnannotatedClassAttribute]
    CSS_PATH: ClassVar[str | PurePath | list[str | PurePath] | None] = APP_CSS_PATHS

    def __init__(
        self,
        config: MotherConfig | None = None,
        model_name: str | None = None,
        system: str | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        super().__init__()
        base = config or MotherConfig()
        self.config: MotherConfig = apply_cli_overrides(base, model_name, system)
        self.agent_mode: bool = self.config.tools_enabled
        self.model: Model | None = None
        self.conversation: Conversation | None = None
        self.session_manager: SessionManager | None = session_manager
        self._pending_executions: list[BashExecution] = []
        self._tool_outputs: dict[str, ToolOutput] = {}
        self._last_context_tokens: int | None = None
        self.auto_scroll_enabled: bool = True

    @override
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="chat-view"):
            yield ConversationTurn(response_text="INTERFACE 2037 READY FOR INQUIRY")
        yield TextArea()
        yield StatusLine(
            self.config.model,
            self.agent_mode,
            self._last_context_tokens,
            self.auto_scroll_enabled,
        )
        yield Footer()

    def on_mount(self) -> None:
        self.model = llm.get_model(self.config.model)
        self.conversation = self.model.conversation()
        _ = self.query_one("#chat-view").anchor()
        self._update_subtitle()
        self._update_statusline()

    def action_show_models(self) -> None:
        """Open the model picker."""
        _ = self.push_screen(ModelPickerScreen(self.config.model))

    def action_switch_model(self, model_id: str) -> None:
        """Switch to a different LLM model and start a fresh conversation."""
        previous_model = self.config.model
        self.config = replace(self.config, model=model_id, tools_enabled=self.agent_mode)
        self.model = llm.get_model(model_id)
        self.conversation = self.model.conversation()
        self._last_context_tokens = None
        self._record_session_event(
            "model_change",
            {"from": previous_model, "model": model_id},
        )
        self._update_subtitle()
        self._update_statusline()
        self.notify(f"Switched to {model_id}", title="Model changed")

    def action_toggle_agent_mode(self) -> None:
        """Toggle agent mode (enables/disables tool use)."""
        self.agent_mode = not self.agent_mode
        self._record_session_event(
            "agent_mode_change",
            {"enabled": self.agent_mode},
        )
        self._update_subtitle()
        state = "enabled" if self.agent_mode else "disabled"
        self._update_statusline()
        self.notify(f"Agent mode {state}", title="Agent mode")

    def action_toggle_thinking_widget(self) -> None:
        """Expand or collapse the latest visible thinking widget."""
        focused = self.focused
        if isinstance(focused, ThinkingOutput) and focused.has_content():
            focused.action_toggle_expanded()
            return

        thinking_widgets = [widget for widget in self.query(ThinkingOutput) if widget.has_content()]
        if thinking_widgets:
            thinking_widgets[-1].action_toggle_expanded()

    def action_toggle_auto_scroll(self) -> None:
        """Toggle whether new chat output should keep the view pinned to the bottom."""
        self.auto_scroll_enabled = not self.auto_scroll_enabled
        self._update_statusline()
        if self.auto_scroll_enabled:
            self._scroll_chat_to_end(force=True)
        state = "enabled" if self.auto_scroll_enabled else "disabled"
        self.notify(f"Autoscroll {state}", title="Chat view")

    def action_scroll_to_bottom(self) -> None:
        """Jump the chat view to the latest content."""
        self._scroll_chat_to_end(force=True)

    def action_scroll_to_bottom_from_chat(self) -> None:
        """Jump to the bottom only when focus is outside the input field."""
        if isinstance(self.focused, TextArea):
            return
        self.action_scroll_to_bottom()

    def _update_subtitle(self) -> None:
        """Update subtitle to show model and agent mode indicator."""
        sub_title: str = f"{self.config.model} [AGENT]" if self.agent_mode else self.config.model
        self.sub_title = sub_title  # pyright: ignore[reportUnannotatedClassAttribute]

    def _update_statusline(self) -> None:
        """Update the single-line status bar above the footer."""
        try:
            status_line = self.query_one(StatusLine)
        except (NoMatches, ScreenStackError):
            return
        status_line.set_status(
            model_name=self.config.model,
            agent_mode=self.agent_mode,
            context_tokens=self._last_context_tokens,
            auto_scroll_enabled=self.auto_scroll_enabled,
        )

    def _refresh_context_size(self) -> None:
        """Capture the latest context token count, if the provider reports one."""
        conversation = self.conversation
        if conversation is None or not conversation.responses:
            self._last_context_tokens = None
        else:
            self._last_context_tokens = conversation.responses[-1].input_tokens
        _ = self.call_from_thread(self._update_statusline)

    def _flush_pending_context(self, value: str) -> str:
        """Build prompt with any pending shell output prepended, then clear the queue."""
        context_parts: list[str] = []
        for execution in self._pending_executions:
            if not execution.exclude_from_context:
                context_parts.append(format_for_context(execution))
        self._pending_executions.clear()
        if context_parts:
            return "\n\n".join(context_parts) + "\n\n" + value
        return value

    def _record_session_message(self, role: str, content: str) -> None:
        """Append a message to the active session if session persistence is enabled."""
        if self.session_manager is None:
            return
        self.session_manager.append(role, content)

    def _record_session_event(self, name: str, details: dict[str, object] | None = None) -> None:
        """Append a structured lifecycle event to the active session."""
        if self.session_manager is None:
            return
        self.session_manager.record_event(name, details)

    def _record_prompt_context(
        self,
        *,
        user_text: str,
        prompt_text: str,
        system_prompt: str,
        tool_names: list[str],
    ) -> None:
        """Record the exact prompt context sent to the model for this turn."""
        if self.session_manager is None:
            return
        self.session_manager.record_prompt(
            user_text=user_text,
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            agent_mode=self.agent_mode,
            tool_names=tool_names,
        )

    def _start_new_session(self) -> None:
        """Rotate to a fresh transient session after a successful save."""
        self.session_manager = SessionManager.create(
            markdown_dir=Path(self.config.session_markdown_dir),
            model_name=self.config.model,
        )

    def action_save_session(self) -> None:
        """Export the current session to markdown, overwriting the same file for this session."""
        if self.session_manager is None:
            self.notify("Session saving is unavailable.", title="Session", severity="warning")
            return

        try:
            output_path = self.session_manager.save_as_markdown()
        except RuntimeError as exc:
            self.notify(str(exc), title="Session", severity="warning")
            return
        except Exception as exc:
            self.notify(f"Failed to save session: {exc}", title="Session", severity="error")
            return

        self.notify(f"Saved to {output_path}", title="Session")

    async def action_submit(self) -> None:
        """When the user hits Ctrl+Enter."""
        text_area = self.query_one(TextArea)
        value = text_area.text.strip()
        if not value:
            return
        _ = text_area.clear()

        parsed = parse_user_input(value)
        if isinstance(parsed, SaveSessionCommand):
            self.action_save_session()
            return
        if isinstance(parsed, ShellCommand):
            await self.run_user_command(parsed)
            return

        self._record_session_message("user", value)
        chat_view = self.query_one("#chat-view")
        should_follow = self._should_follow_chat_updates()
        text_area.read_only = True
        prompt = self._flush_pending_context(value)
        turn = ConversationTurn(prompt_text=value, include_thinking=True)
        _ = await chat_view.mount(turn)
        if should_follow:
            self._scroll_chat_to_end(force=True)
        thinking_output = turn.thinking_output
        if thinking_output is None:
            text_area.read_only = False
            return
        _ = self.send_prompt(prompt, value, turn.response_widget, thinking_output)

    async def run_user_command(self, cmd: ShellCommand) -> None:
        """Execute a direct user shell command and display the output."""
        chat_view = self.query_one("#chat-view")
        should_follow = self._should_follow_chat_updates()
        shell_widget = ShellOutput(f"Running: {cmd.command}")
        section = OutputSection("Shell", "shell-title", shell_widget)
        _ = await chat_view.mount(section)
        if should_follow:
            self._scroll_chat_to_end(force=True)

        try:
            result = await execute_bash(cmd.command)
        except Exception as exc:
            output = f"Error: {exc}"
            exit_code = None
        else:
            output = result.output
            exit_code = result.exit_code

        shell_widget.set_text(output)
        prefix = "!" if cmd.include_in_context else "!!"
        self._record_session_message("user", f"{prefix}{cmd.command}")
        self._record_session_message("assistant", f"$ {cmd.command}\n\n{output}")

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
        should_follow = self._should_follow_chat_updates()
        widget = ToolOutput(format_tool_event(tool_name, arguments, status="started"))
        self._tool_outputs[key] = widget
        section = OutputSection("Tool", "tool-title", widget)
        chat_view = self.query_one("#chat-view", VerticalScroll)
        _ = chat_view.mount(section)
        if should_follow:
            self._scroll_chat_to_end(force=True)

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
        should_follow = self._should_follow_chat_updates()
        chat_view = self.query_one("#chat-view", VerticalScroll)
        if widget is None:
            widget = ToolOutput()
            section = OutputSection("Tool", "tool-title", widget)
            _ = chat_view.mount(section)
        widget.set_text(format_tool_event(tool_name, arguments, status="finished", output=output))
        if should_follow:
            self._scroll_chat_to_end(force=True)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Re-enable input when the worker finishes."""
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            text_area = self.query_one(TextArea)
            text_area.read_only = False
            _ = text_area.focus()

    def _chat_is_near_end(self, margin: int = 1) -> bool:
        """Return whether the chat view is already at (or very near) the bottom."""
        try:
            chat_view = self.query_one("#chat-view", VerticalScroll)
        except (NoMatches, ScreenStackError):
            return True
        return (chat_view.max_scroll_y - chat_view.scroll_y) <= margin

    def _should_follow_chat_updates(self) -> bool:
        """Return whether the next chat update should keep following new output."""
        return self.auto_scroll_enabled and self._chat_is_near_end()

    def _scroll_chat_to_end(self, *, force: bool = False) -> None:
        """Keep the chat view pinned to the latest streamed content."""
        if not force and not self.auto_scroll_enabled:
            return
        try:
            chat_view = self.query_one("#chat-view", VerticalScroll)
        except (NoMatches, ScreenStackError):
            return
        _ = chat_view.scroll_end(animate=False)

    def _update_response_output(self, response: Response, text: str) -> None:
        """Render response text and keep following only when the user is at the bottom."""
        should_follow = self._should_follow_chat_updates()
        _ = response.update(text)
        if should_follow:
            self._scroll_chat_to_end(force=True)

    def _start_thinking_output(self, thinking_output: ThinkingOutput) -> None:
        """Switch the thinking widget into live streaming mode."""
        should_follow = self._should_follow_chat_updates()
        thinking_output.start_streaming()
        if should_follow:
            self._scroll_chat_to_end(force=True)

    def _update_thinking_output(self, thinking_output: ThinkingOutput, text: str) -> None:
        """Render streamed reasoning text in the dedicated thinking widget."""
        should_follow = self._should_follow_chat_updates()
        thinking_output.set_text(text)
        if should_follow:
            self._scroll_chat_to_end(force=True)

    def _finish_thinking_output(self, thinking_output: ThinkingOutput) -> None:
        """Collapse the thinking widget back to its preview after streaming ends."""
        should_follow = self._should_follow_chat_updates()
        thinking_output.finish_streaming()
        if should_follow:
            self._scroll_chat_to_end(force=True)

    def _stream_response(
        self,
        llm_response: Iterable[str],
        response: Response,
        thinking_output: ThinkingOutput | None = None,
    ) -> str:
        """Stream chunks from an LLM response into the visible widgets; return final reply text."""
        full_text = ""
        if thinking_output is None:
            for chunk in llm_response:
                full_text += chunk
                _ = self.call_from_thread(self._update_response_output, response, full_text)
            return full_text

        thinking_parser = ThinkTagStreamParser()
        thinking_text = ""
        thinking_streaming = False

        for chunk in llm_response:
            thinking_delta, response_delta = thinking_parser.feed(chunk)
            if thinking_delta:
                thinking_text += thinking_delta
                if not thinking_streaming:
                    _ = self.call_from_thread(self._start_thinking_output, thinking_output)
                    thinking_streaming = True
                _ = self.call_from_thread(
                    self._update_thinking_output, thinking_output, thinking_text
                )
            if thinking_streaming and not thinking_parser.in_think:
                _ = self.call_from_thread(self._finish_thinking_output, thinking_output)
                thinking_streaming = False
            if response_delta:
                full_text += response_delta
                _ = self.call_from_thread(self._update_response_output, response, full_text)

        thinking_delta, response_delta = thinking_parser.flush()
        if thinking_delta:
            thinking_text += thinking_delta
            if not thinking_streaming:
                _ = self.call_from_thread(self._start_thinking_output, thinking_output)
                thinking_streaming = True
            _ = self.call_from_thread(self._update_thinking_output, thinking_output, thinking_text)
        if thinking_streaming:
            _ = self.call_from_thread(self._finish_thinking_output, thinking_output)
        if response_delta:
            full_text += response_delta
            _ = self.call_from_thread(self._update_response_output, response, full_text)

        return full_text

    @staticmethod
    def _tool_call_arguments(tool_call: ToolCall) -> dict[str, object]:
        """Convert tool call arguments into a regular dictionary."""
        arguments = cast(
            dict[object, object],
            tool_call.arguments,
        )
        return {str(key): value for key, value in arguments.items()}

    def _before_tool_call(self, tool: Tool | None, tool_call: ToolCall) -> None:
        """Show a trace entry when an agent tool call starts."""
        tool_name = tool.name if tool is not None else tool_call.name
        arguments = self._tool_call_arguments(tool_call)
        if self.session_manager is not None:
            self.session_manager.record_tool_call(
                tool_name=tool_name,
                tool_call_id=tool_call.tool_call_id,
                arguments=arguments,
            )
        _ = self.call_from_thread(
            self._show_tool_started,
            tool_name,
            tool_call.tool_call_id,
            arguments,
        )

    def _after_tool_call(self, tool: Tool, tool_call: ToolCall, tool_result: ToolResult) -> None:
        """Update the trace entry when an agent tool call finishes."""
        arguments = self._tool_call_arguments(tool_call)
        if self.session_manager is not None:
            self.session_manager.record_tool_result(
                tool_name=tool.name,
                tool_call_id=tool_call.tool_call_id,
                arguments=arguments,
                output=tool_result.output,
            )
        _ = self.call_from_thread(
            self._show_tool_finished,
            tool.name,
            tool_call.tool_call_id,
            arguments,
            tool_result.output,
        )

    def _get_enabled_tools(self) -> list[ToolDef] | None:
        """Return the active tool definitions, if any are enabled and available."""
        tool_registry = get_default_tools(
            tools_enabled=self.agent_mode,
            allowlist=self.config.allowlist,
            ca_bundle_path=self.config.ca_bundle_path,
        )
        if tool_registry.is_empty():
            return None
        return tool_registry.tools()

    @staticmethod
    def _tool_names(tools: list[ToolDef] | None) -> list[str]:
        """Extract stable prompt-friendly tool names from tool definitions."""
        if tools is None:
            return []

        names: list[str] = []
        seen: set[str] = set()
        for tool in tools:
            raw_name = getattr(tool, "name", None)
            if not isinstance(raw_name, str) or not raw_name:
                raw_name = getattr(tool, "__name__", tool.__class__.__name__)
            if raw_name in seen:
                continue
            seen.add(raw_name)
            names.append(raw_name)
        return names

    def _build_system_prompt(
        self,
        tools: list[ToolDef] | None,
        *,
        agent_mode: bool | None = None,
    ) -> str:
        """Build the runtime system prompt for the current model turn."""
        effective_agent_mode = self.agent_mode if agent_mode is None else agent_mode
        return build_system_prompt(
            self.config.system_prompt,
            agent_mode=effective_agent_mode,
            cwd=Path.cwd(),
            tool_names=self._tool_names(tools),
        )

    def _build_llm_response(
        self,
        conversation: Conversation,
        prompt: str,
        system: str | None,
        tools: list[ToolDef] | None,
        before_tool_call: BeforeCallSync | None,
        after_tool_call: AfterCallSync | None,
    ) -> Iterable[str]:
        """Build the LLM response stream, with tool callbacks when agent mode is active."""
        if tools is None:
            prompt_fn = cast(Callable[..., Iterable[str]], conversation.prompt)
            return prompt_fn(prompt, system=system)

        chain_fn = cast(Callable[..., Iterable[str]], conversation.chain)
        return chain_fn(
            prompt,
            system=system,
            tools=tools,
            chain_limit=3,
            before_call=before_tool_call,
            after_call=after_tool_call,
        )

    @staticmethod
    def _is_unsupported_tools_error(error: Exception) -> bool:
        """Return whether the model error indicates tool support is unavailable."""
        return "does not support tools" in str(error)

    def _request_llm_response(
        self,
        conversation: Conversation,
        prompt: str,
        system: str | None,
        tools: list[ToolDef] | None,
        response: Response,
    ) -> Iterable[str] | None:
        """Request an LLM response stream, falling back if the model rejects tools."""
        before_tool_call: BeforeCallSync | None = None
        after_tool_call: AfterCallSync | None = None
        if tools is not None:
            tool_calls_this_turn = 0

            def limited_before_tool_call(tool: Tool | None, tool_call: ToolCall) -> None:
                nonlocal tool_calls_this_turn
                if tool_calls_this_turn >= 1:
                    raise llm.CancelToolCall(
                        "Only one tool call is allowed per turn. Report findings and wait for the user before continuing."
                    )
                tool_calls_this_turn += 1
                self._before_tool_call(tool, tool_call)

            before_tool_call = limited_before_tool_call
            after_tool_call = self._after_tool_call

        try:
            return self._build_llm_response(
                conversation,
                prompt,
                system,
                tools,
                before_tool_call,
                after_tool_call,
            )
        except Exception as exc:
            if tools is None or not self._is_unsupported_tools_error(exc):
                self._show_error(response, f"**Error:** {exc}")
                return None

        _ = self.call_from_thread(self._disable_agent_mode_unsupported)
        prompt_fn = cast(Callable[..., Iterable[str]], conversation.prompt)
        fallback_system = self._build_system_prompt(None, agent_mode=False)
        try:
            return prompt_fn(prompt, system=fallback_system)
        except Exception as exc:
            self._show_error(response, f"**Error:** {exc}")
            return None

    def _stream_llm_response(
        self,
        llm_response: Iterable[str],
        response: Response,
        thinking_output: ThinkingOutput | None = None,
    ) -> str | None:
        """Stream an LLM response into the widget and return the final reply text."""
        try:
            full_text = self._stream_response(llm_response, response, thinking_output)
        except Exception as exc:
            self._show_error(response, f"**Error:** {exc}")
            return None

        response.reset_state(full_text)
        self._refresh_context_size()
        return full_text

    def _show_error(self, response: Response, error_text: str) -> None:
        """Display an error in the response widget and reset its state."""
        _ = self.call_from_thread(self._update_response_output, response, error_text)
        response.reset_state(error_text)

    @work(thread=True)
    def send_prompt(
        self,
        prompt: str,
        user_text: str,
        response: Response,
        thinking_output: ThinkingOutput | None = None,
    ) -> None:
        """Get the response in a thread, maintaining conversation history."""
        _ = self.call_from_thread(self._update_response_output, response, "*Thinking...*")
        conversation = self.conversation
        if conversation is None:
            self._show_error(response, "**Error:** Conversation is not initialized.")
            return

        tools = self._get_enabled_tools()
        tool_names = self._tool_names(tools)
        system = self._build_system_prompt(tools)
        self._record_prompt_context(
            user_text=user_text,
            prompt_text=prompt,
            system_prompt=system,
            tool_names=tool_names,
        )
        llm_response = self._request_llm_response(
            conversation=conversation,
            prompt=prompt,
            system=system,
            tools=tools,
            response=response,
        )
        if llm_response is None:
            return

        full_text = self._stream_llm_response(llm_response, response, thinking_output)
        if full_text is not None:
            self._record_session_message("assistant", full_text)

    def _disable_agent_mode_unsupported(self) -> None:
        """Disable agent mode and notify user that the model doesn't support tools."""
        self.agent_mode = False
        self._record_session_event(
            "agent_mode_disabled_unsupported",
            {"model": self.config.model},
        )
        self._update_subtitle()
        self.notify(
            f"{self.config.model} does not support tools — agent mode disabled",
            title="Agent mode",
            severity="warning",
        )


@click.command()
@click.option("--model", "-m", default=None, help="LLM model to use.")
@click.option("--system", "-s", default=None, help="System prompt.")
@click.option("--save", "save_last", is_flag=True, help="Save the last unsaved session and exit.")
def cli(model: str | None, system: str | None, save_last: bool) -> None:
    """Launch the Mother TUI chatbot."""
    config = load_config()
    config = apply_cli_overrides(config, model, system)

    if save_last:
        output_path = SessionManager.save_last(markdown_dir=Path(config.session_markdown_dir))
        if output_path is None:
            click.echo("No unsaved session found.")
            return
        click.echo(f"Saved: {output_path}")
        return

    session_manager = SessionManager.create(
        markdown_dir=Path(config.session_markdown_dir),
        model_name=config.model,
    )
    app = MotherApp(config=config, session_manager=session_manager)
    app.run()
