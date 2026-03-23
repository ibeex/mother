"""Mother TUI chatbot — a Textual interface for chatting with an LLM."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path, PurePath
from random import choice
from time import monotonic
from typing import ClassVar, cast, override

import click
from pydantic_ai import Tool
from textual import events, on, work
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Footer, Header, OptionList, TextArea
from textual.worker import Worker, WorkerState

from mother.agent_modes import (
    DEFAULT_AGENT_PROFILE,
    AgentProfile,
    RuntimeMode,
    format_agent_profile,
    format_agent_status,
    normalize_agent_profile,
    resolve_runtime_mode,
)
from mother.bash_execution import BashExecution, format_for_context
from mother.clipboard import ClipboardImageError, save_clipboard_image
from mother.config import MotherConfig, apply_cli_overrides, load_config
from mother.conversation import ConversationState
from mother.interrupts import UserInterruptedError
from mother.model_picker import (
    AgentModeProvider,
    ModelPickerScreen,
    ModelProvider,
    ModelSwitchConfirmScreen,
)
from mother.models import ModelEntry, resolve_model_entry
from mother.reasoning import (
    REASONING_EFFORT_HELP,
    build_reasoning_options,
    format_reasoning_effort,
    normalize_reasoning_effort,
    supported_reasoning_efforts,
    supports_openai_reasoning_summary,
    supports_reasoning_effort,
)
from mother.runtime import ChatRuntime, RuntimeToolEvent
from mother.session import SessionManager, format_markdown_export
from mother.slash_commands import (
    SLASH_COMMANDS,
    SlashCommand,
    current_slash_argument_query,
    current_slash_query,
    get_slash_argument_spec,
)
from mother.stats import SessionUsage, TurnUsage
from mother.system_prompt import build_system_prompt
from mother.tool_trace import format_tool_event
from mother.tools import get_default_tools
from mother.tools.bash_capture import BashResult
from mother.tools.bash_executor import execute_bash
from mother.user_commands import (
    AgentModeCommand,
    ModelsCommand,
    QuitAppCommand,
    ReasoningCommand,
    SaveSessionCommand,
    ShellCommand,
    parse_user_input,
)
from mother.widgets import (
    ConversationTurn,
    ModelComplete,
    OutputSection,
    PromptTextArea,
    Response,
    ShellOutput,
    SlashArgumentComplete,
    SlashComplete,
    StatusLine,
    ThinkingOutput,
    ToolOutput,
    TurnLabel,
    WelcomeBanner,
)

CSS_DIR = Path(__file__).resolve().parent / "css"
APP_CSS_PATHS: list[str | PurePath] = [
    CSS_DIR / "chat.tcss",
    CSS_DIR / "output.tcss",
    CSS_DIR / "input.tcss",
]


@dataclass
class _ResponseWaitingAnimation:
    """Track the lightweight response placeholder animation for a single turn."""

    response: Response
    message: str
    frame_index: int = 0


class MotherApp(App[None]):
    """Simple app for chatting with an LLM via a conversation."""

    AUTO_FOCUS: ClassVar[str | None] = "#prompt-input"
    RESPONSE_WAITING_MESSAGES: ClassVar[tuple[str, ...]] = (
        "WEYLAND-YUTANI SYSTEMS ONLINE",
        "DIRECTIVE REVIEW IN PROGRESS",
        "PRIORITY PROTOCOL ENGAGED",
        "CREW QUERY ACKNOWLEDGED",
        "INTERNAL SYSTEMS RESPONDING",
        "TRANSMISSION RECEIVED. PROCESSING",
        "COMMAND INTERFACE ACTIVE",
        "DATA RELAY IN PROGRESS",
        "OPERATIONAL ANALYSIS UNDERWAY",
        "SPECIAL ORDER PARAMETERS DETECTED",
    )
    DOUBLE_ESCAPE_WINDOW_SECONDS: ClassVar[float] = 0.4

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
        self.theme = self.config.theme  # pyright: ignore[reportUnannotatedClassAttribute]
        self.agent_mode: bool = self.config.tools_enabled
        self.agent_profile: AgentProfile = DEFAULT_AGENT_PROFILE
        self.current_model_entry: ModelEntry = resolve_model_entry(
            self.config.model,
            self.config.models,
        )
        self.conversation_state: ConversationState = ConversationState()
        self.session_manager: SessionManager | None = session_manager
        self._pending_executions: list[BashExecution] = []
        self._pending_image_attachments: dict[str, Path] = {}
        self._active_turn: ConversationTurn | None = None
        self._tool_outputs: dict[str, ToolOutput] = {}
        self._session_usage: SessionUsage = SessionUsage()
        self._last_turn_usage: TurnUsage | None = None
        self._last_context_tokens: int | None = None
        self._session_input_tokens: int | None = None
        self._session_output_tokens: int | None = None
        self._session_cached_tokens: int | None = None
        self._last_response_time_seconds: float | None = None
        self._suppress_prompt_completion_once: str | None = None
        self.auto_scroll_enabled: bool = True
        self._response_waiting_animations: dict[int, _ResponseWaitingAnimation] = {}
        self._last_interrupt_escape_at: float | None = None
        self._active_prompt_worker: Worker[None] | None = None
        self._active_shell_worker: Worker[None] | None = None
        self._active_runtime_loop: asyncio.AbstractEventLoop | None = None
        self._active_runtime_task: asyncio.Task[str | None] | None = None
        self._active_shell_task: asyncio.Task[BashResult] | None = None

    @override
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="main-pane"):
            with VerticalScroll(id="chat-view"):
                yield WelcomeBanner()
            with Vertical(id="prompt-area"):
                slash_complete = SlashComplete(SLASH_COMMANDS)
                slash_complete.display = False
                yield slash_complete
                slash_argument_complete = SlashArgumentComplete()
                slash_argument_complete.display = False
                yield slash_argument_complete
                with Horizontal(id="prompt-row"):
                    yield TurnLabel(">", classes="turn-gutter input-gutter")
                    yield PromptTextArea(id="prompt-input")
        yield StatusLine(
            self.config.model,
            self.agent_mode,
            self._last_context_tokens,
            self.auto_scroll_enabled,
            self._status_reasoning_effort(),
            self._last_response_time_seconds,
            self._session_input_tokens,
            self._session_output_tokens,
            self._session_cached_tokens,
            agent_label=self._status_agent_label(),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.current_model_entry = resolve_model_entry(self.config.model, self.config.models)
        self.conversation_state = ConversationState()
        _ = self.query_one("#chat-view").anchor()
        _ = self.set_interval(0.12, self._tick_response_waiting_animations)
        self._update_subtitle()
        self._update_statusline()

    def on_key(self, event: events.Key) -> None:
        """Handle global Escape interruption even when focus isn't in the prompt."""
        if event.key != "escape":
            return
        if self.handle_interrupt_escape():
            _ = event.stop()
            _ = event.prevent_default()

    @property
    def prompt_input(self) -> PromptTextArea:
        """Return the main editable prompt input widget."""
        return self.query_one("#prompt-input", PromptTextArea)

    @property
    def slash_complete(self) -> SlashComplete:
        """Return the slash-command autocomplete popup widget."""
        return self.query_one(SlashComplete)

    @property
    def slash_argument_complete(self) -> SlashArgumentComplete:
        """Return the inline slash-argument autocomplete popup widget."""
        return self.query_one(SlashArgumentComplete)

    @property
    def model_complete(self) -> ModelComplete:
        """Backward-compatible alias for the inline slash-argument popup widget."""
        return self.query_one(ModelComplete)

    def _hide_slash_complete(self) -> None:
        """Hide slash-command autocomplete and restore normal prompt keys."""
        self.slash_complete.display = False
        self.prompt_input.slash_complete_active = False

    def _hide_slash_argument_complete(self) -> None:
        """Hide inline slash-argument autocomplete and restore normal prompt keys."""
        self.slash_argument_complete.display = False
        self.prompt_input.slash_argument_complete_active = False

    def _current_slash_argument_value(self, command: str) -> str | None:
        """Return the active value to highlight for a slash command argument."""
        normalized_command = command.lower()
        if normalized_command == "/agent":
            if not self.agent_mode:
                return None
            return format_agent_profile(self.agent_profile)
        if normalized_command == "/models":
            return self.config.model
        if normalized_command == "/reasoning":
            return format_reasoning_effort(self.config.reasoning_effort)
        return None

    def _refresh_prompt_completions(self, text: str) -> None:
        """Show, filter, or hide prompt helpers for slash commands and arguments."""
        argument_query = current_slash_argument_query(text)
        if argument_query is not None:
            spec = get_slash_argument_spec(argument_query.command)
            if spec is not None:
                self._hide_slash_complete()
                matches = spec.complete(argument_query.query)
                has_matches = self.slash_argument_complete.update_matches(
                    matches,
                    self._current_slash_argument_value(argument_query.command),
                )
                self.slash_argument_complete.display = has_matches
                self.prompt_input.slash_argument_complete_active = has_matches
                return

        self._hide_slash_argument_complete()
        query = current_slash_query(text)
        if query is None or get_slash_argument_spec(query) is not None:
            self._hide_slash_complete()
            return
        has_matches = self.slash_complete.update_query(query)
        self.slash_complete.display = has_matches
        self.prompt_input.slash_complete_active = has_matches

    def _selected_slash_command(self) -> SlashCommand | None:
        """Return the highlighted slash command from the popup, if any."""
        return self.slash_complete.highlighted_command()

    def _selected_slash_argument_value(self, command: str | None = None) -> str | None:
        """Return the highlighted inline slash-argument value, if any."""
        if not self.slash_argument_complete.display:
            return None
        active_query = current_slash_argument_query(self.prompt_input.text)
        if active_query is None:
            return None
        if command is not None and active_query.command.lower() != command.lower():
            return None
        return self.slash_argument_complete.highlighted_value()

    def _apply_slash_completion(self, command: SlashCommand | None = None) -> None:
        """Insert the selected slash command back into the prompt input."""
        selected_command = command or self._selected_slash_command()
        if selected_command is None:
            return

        completed_text = f"{selected_command.command} "
        self.prompt_input.load_text(completed_text)
        self.prompt_input.move_cursor((0, len(completed_text)), record_width=False)
        self._hide_slash_complete()
        self._refresh_prompt_completions(completed_text)
        _ = self.prompt_input.focus()

    def _apply_slash_argument_completion(self, value: str | None = None) -> None:
        """Insert the selected inline slash-argument completion into the prompt input."""
        active_query = current_slash_argument_query(self.prompt_input.text)
        if active_query is None:
            return
        selected_value = value or self._selected_slash_argument_value(active_query.command)
        if selected_value is None:
            return

        completed_text = f"{active_query.command} {selected_value}"
        self._suppress_prompt_completion_once = completed_text
        self.prompt_input.load_text(completed_text)
        self.prompt_input.move_cursor((0, len(completed_text)), record_width=False)
        self._hide_slash_argument_complete()
        _ = self.prompt_input.focus()

    def _conversation_has_history(self) -> bool:
        """Return whether the active conversation already contains model-visible history."""
        return self.conversation_state.has_history

    def action_show_models(self) -> None:
        """Open the model picker."""

        def on_model_selected(model_id: str | None) -> None:
            if model_id is not None:
                self.action_switch_model(model_id)

        _ = self.push_screen(ModelPickerScreen(self.config.model), on_model_selected)

    def _resolve_slash_argument_query(self, command: str, query: str) -> str | None:
        """Resolve a slash-command argument query to a concrete completion value."""
        highlighted_value = self._selected_slash_argument_value(command)
        if highlighted_value is not None:
            return highlighted_value

        spec = get_slash_argument_spec(command)
        if spec is None:
            return None
        return spec.resolve(query)

    def _reasoning_options(self) -> dict[str, object]:
        """Return supported model options for reasoning-capable models."""
        return build_reasoning_options(
            self.current_model_entry,
            self.config.reasoning_effort,
            self.config.openai_reasoning_summary,
        )

    def _show_reasoning_status(self) -> None:
        """Notify the user of the current reasoning setting and model support."""
        configured = format_reasoning_effort(self.config.reasoning_effort)
        summary_suffix = ""
        if supports_openai_reasoning_summary(self.current_model_entry):
            summary_suffix = f" · summary {self.config.openai_reasoning_summary}"
        supported = supported_reasoning_efforts(self.current_model_entry)
        if supported:
            if (
                self.config.reasoning_effort != "auto"
                and self.config.reasoning_effort not in supported
            ):
                supported_text = "|".join(format_reasoning_effort(value) for value in supported)
                self.notify(
                    (
                        f"{self.config.model} reasoning: {configured}{summary_suffix} "
                        f"(not supported here). Supported: {supported_text}"
                    ),
                    title="Reasoning",
                    severity="warning",
                )
                return
            self.notify(
                f"{self.config.model} reasoning: {configured}{summary_suffix}",
                title="Reasoning",
            )
            return
        self.notify(
            (
                f"{self.config.model} does not expose reasoning control. "
                f"Configured default: {configured}"
            ),
            title="Reasoning",
            severity="warning",
        )

    def _set_reasoning_effort(self, effort: str) -> None:
        """Update the configured reasoning effort for future model requests."""
        normalized = normalize_reasoning_effort(effort)
        if normalized is None:
            self.notify(
                f"Use /reasoning {REASONING_EFFORT_HELP}",
                title="Reasoning",
                severity="warning",
            )
            return

        supported = supported_reasoning_efforts(self.current_model_entry)
        if supported and normalized != "auto" and normalized not in supported:
            supported_text = "|".join(format_reasoning_effort(value) for value in supported)
            self.notify(
                f"{self.config.model} supports: {supported_text}",
                title="Reasoning",
                severity="warning",
            )
            return

        previous = self.config.reasoning_effort
        self.config = replace(
            self.config,
            reasoning_effort=normalized,
            tools_enabled=self.agent_mode,
        )
        self._record_session_event(
            "reasoning_effort_change",
            {
                "from": previous,
                "reasoning_effort": normalized,
                "model": self.config.model,
            },
        )
        self._update_statusline()
        configured = format_reasoning_effort(normalized)
        if supports_reasoning_effort(self.current_model_entry):
            self.notify(
                f"Reasoning set to {configured} for {self.config.model}",
                title="Reasoning",
            )
            return
        self.notify(
            (
                f"Reasoning set to {configured}. {self.config.model} does not expose "
                "reasoning control, so this will apply when you switch models."
            ),
            title="Reasoning",
            severity="warning",
        )

    def _apply_model_switch(self, model_id: str) -> None:
        """Switch to a different LLM model and start a fresh conversation."""
        previous_model = self.config.model
        self.config = replace(self.config, model=model_id, tools_enabled=self.agent_mode)
        self.current_model_entry = resolve_model_entry(model_id, self.config.models)
        self.conversation_state = ConversationState()
        self._last_context_tokens = None
        self._last_response_time_seconds = None
        self._record_session_event(
            "model_change",
            {"from": previous_model, "model": model_id},
        )
        self._update_subtitle()
        self._update_statusline()
        self.notify(f"Switched to {model_id}", title="Model changed")

    def action_switch_model(self, model_id: str) -> None:
        """Switch to a different LLM model, asking first if that will clear context."""
        if model_id == self.config.model:
            return

        if not self._conversation_has_history():
            self._apply_model_switch(model_id)
            return

        def on_confirm(confirmed: bool | None) -> None:
            if confirmed:
                self._apply_model_switch(model_id)

        _ = self.push_screen(ModelSwitchConfirmScreen(model_id), on_confirm)

    def _runtime_mode(self) -> RuntimeMode:
        """Return the effective runtime mode for prompts, sessions, and tool limits."""
        return resolve_runtime_mode(
            agent_enabled=self.agent_mode,
            agent_profile=self.agent_profile,
        )

    def _status_agent_label(self) -> str:
        """Return the compact status-line label for the current agent state."""
        return format_agent_status(self.agent_mode, self.agent_profile)

    def _set_agent_mode(
        self,
        *,
        enabled: bool,
        profile: AgentProfile | None = None,
    ) -> tuple[bool, AgentProfile]:
        """Update agent state, refresh UI, and record the transition."""
        previous_enabled = self.agent_mode
        previous_profile = self.agent_profile
        if profile is not None:
            self.agent_profile = profile
        self.agent_mode = enabled
        self._record_session_event(
            "agent_mode_change",
            {
                "enabled": self.agent_mode,
                "profile": self.agent_profile,
                "previous_enabled": previous_enabled,
                "previous_profile": previous_profile,
            },
        )
        self._update_subtitle()
        self._update_statusline()
        return previous_enabled, previous_profile

    def action_set_agent_profile(self, profile: AgentProfile) -> None:
        """Enable agent mode with a specific profile."""
        previous_enabled, previous_profile = self._set_agent_mode(enabled=True, profile=profile)
        if profile == "deep_research":
            message = "Deep research mode enabled"
            if previous_enabled and previous_profile != profile:
                message = "Switched to deep research mode"
        else:
            message = "Agent mode enabled"
            if previous_enabled and previous_profile != profile:
                message = f"Switched to {format_agent_profile(profile)} agent mode"
        self.notify(message, title="Agent mode")

    def action_toggle_agent_mode(self) -> None:
        """Toggle standard agent mode on or disable the current agent mode."""
        if self.agent_mode:
            _ = self._set_agent_mode(enabled=False)
            self.notify("Agent mode disabled", title="Agent mode")
            return
        self.action_set_agent_profile("standard")

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
        if isinstance(self.focused, PromptTextArea):
            return
        self.action_scroll_to_bottom()

    def _update_subtitle(self) -> None:
        """Update subtitle to show model and the active runtime mode."""
        if not self.agent_mode:
            sub_title = self.config.model
        elif self.agent_profile == "deep_research":
            sub_title = f"{self.config.model} [RESEARCH]"
        else:
            sub_title = f"{self.config.model} [AGENT]"
        self.sub_title = sub_title  # pyright: ignore[reportUnannotatedClassAttribute]

    def _status_reasoning_effort(self) -> str | None:
        """Return the visible reasoning setting for reasoning-capable models."""
        if not supports_reasoning_effort(self.current_model_entry):
            return None
        label = format_reasoning_effort(self.config.reasoning_effort)
        if supports_openai_reasoning_summary(self.current_model_entry):
            summary = self.config.openai_reasoning_summary
            if summary != "auto":
                return f"{label}/{summary}"
            return label
        if self.current_model_entry.api_type == "anthropic":
            if label not in {"auto", "off"}:
                return f"{label}/thinking"
        return label

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
            reasoning_effort=self._status_reasoning_effort(),
            last_response_time_seconds=self._last_response_time_seconds,
            input_tokens=self._session_input_tokens,
            output_tokens=self._session_output_tokens,
            cached_tokens=self._session_cached_tokens,
            agent_label=self._status_agent_label(),
        )

    def _remember_last_response_time(self, duration_seconds: float) -> None:
        """Store the most recent successful model-response duration and refresh the status line."""
        self._last_response_time_seconds = max(0.0, duration_seconds)
        self._update_statusline()

    def _apply_turn_usage(self, usage: TurnUsage) -> None:
        """Accumulate normalized turn statistics and refresh the status line."""
        self._last_turn_usage = usage
        self._session_usage.add_turn(usage)
        self._last_context_tokens = self._session_usage.last_context_tokens
        self._last_response_time_seconds = self._session_usage.last_response_time_seconds
        self._session_input_tokens = self._session_usage.request_tokens
        self._session_output_tokens = self._session_usage.response_tokens
        self._session_cached_tokens = self._session_usage.cache_read_tokens
        self._update_statusline()

    def _refresh_context_size(self) -> None:
        """Refresh the status line for the latest normalized usage state."""
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

    def capture_clipboard_image(self) -> str | None:
        """Save a clipboard image to a temp file and register it as a pending attachment."""
        try:
            path = save_clipboard_image()
        except ClipboardImageError as exc:
            self.notify(str(exc), title="Clipboard", severity="warning")
            return None

        if path is None:
            return None

        self._pending_image_attachments[str(path)] = path
        self.notify(f"Attached image: {path.name}", title="Clipboard")
        return str(path)

    def _consume_attachments_for_text(self, text: str) -> list[Path]:
        """Return and clear any pending image attachments referenced by the prompt text."""
        attachment_paths = [path for path in self._pending_image_attachments if path in text]
        return [self._pending_image_attachments.pop(path) for path in attachment_paths]

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
        attachment_paths: list[str],
    ) -> None:
        """Record the exact prompt context sent to the model for this turn."""
        if self.session_manager is None:
            return
        self.session_manager.record_prompt(
            user_text=user_text,
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            agent_mode=self.agent_mode,
            mode=self._runtime_mode(),
            tool_names=tool_names,
            attachment_paths=attachment_paths,
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

        notice = format_markdown_export(output_path)
        if notice is None:
            return
        if notice.severity is None:
            self.notify(notice.message, title="Session")
            return
        self.notify(notice.message, title="Session", severity=notice.severity)

    def action_quit_app(self) -> None:
        """Close the application immediately."""
        self.exit()

    @on(TextArea.Changed)
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Refresh prompt autocomplete helpers as the main prompt changes."""
        if event.text_area is not self.prompt_input:
            return
        if event.text_area.text == self._suppress_prompt_completion_once:
            self._suppress_prompt_completion_once = None
            self._hide_slash_argument_complete()
            self._hide_slash_complete()
            return
        self._suppress_prompt_completion_once = None
        self._refresh_prompt_completions(event.text_area.text)

    @on(PromptTextArea.SlashNavigate)
    def on_prompt_text_area_slash_navigate(self, event: PromptTextArea.SlashNavigate) -> None:
        """Move the slash-command highlight while the prompt retains focus."""
        if not event.text_area.slash_complete_active:
            return
        if event.direction < 0:
            self.slash_complete.action_cursor_up()
        else:
            self.slash_complete.action_cursor_down()

    @on(PromptTextArea.SlashArgumentNavigate)
    def on_prompt_text_area_slash_argument_navigate(
        self,
        event: PromptTextArea.SlashArgumentNavigate,
    ) -> None:
        """Move the inline slash-argument highlight while the prompt retains focus."""
        if not event.text_area.slash_argument_complete_active:
            return
        if event.direction < 0:
            self.slash_argument_complete.action_cursor_up()
        else:
            self.slash_argument_complete.action_cursor_down()

    @on(PromptTextArea.SlashAccept)
    def on_prompt_text_area_slash_accept(self, event: PromptTextArea.SlashAccept) -> None:
        """Insert the currently highlighted slash command into the prompt."""
        if event.text_area.slash_complete_active:
            self._apply_slash_completion()

    @on(PromptTextArea.SlashDismiss)
    def on_prompt_text_area_slash_dismiss(self, event: PromptTextArea.SlashDismiss) -> None:
        """Dismiss slash-command autocomplete without changing prompt text."""
        if event.text_area.slash_complete_active:
            self._hide_slash_complete()

    @on(PromptTextArea.SlashArgumentAccept)
    def on_prompt_text_area_slash_argument_accept(
        self,
        event: PromptTextArea.SlashArgumentAccept,
    ) -> None:
        """Insert the currently highlighted slash-argument completion into the prompt."""
        if event.text_area.slash_argument_complete_active:
            self._apply_slash_argument_completion()

    @on(PromptTextArea.SlashArgumentDismiss)
    def on_prompt_text_area_slash_argument_dismiss(
        self,
        event: PromptTextArea.SlashArgumentDismiss,
    ) -> None:
        """Dismiss inline slash-argument autocomplete without changing prompt text."""
        if event.text_area.slash_argument_complete_active:
            self._hide_slash_argument_complete()

    @on(PromptTextArea.SlashSubmit)
    async def on_prompt_text_area_slash_submit(self, event: PromptTextArea.SlashSubmit) -> None:
        """Submit built-in slash commands with Enter instead of inserting a newline."""
        if event.text_area is self.prompt_input:
            await self.action_submit()

    @on(OptionList.OptionSelected)
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle prompt popup selections from slash and argument helpers."""
        if event.option_list is self.slash_complete and event.option.id is not None:
            selected = next(
                (
                    command
                    for command in self.slash_complete.matches
                    if command.command == event.option.id
                ),
                None,
            )
            self._apply_slash_completion(selected)
            return

        if event.option_list is self.slash_argument_complete and event.option.id is not None:
            self._apply_slash_argument_completion(event.option.id)

    async def action_submit(self) -> None:
        """When the user hits Ctrl+Enter."""
        text_area = self.prompt_input
        value = text_area.text.strip()
        if not value:
            return
        _ = text_area.clear()

        parsed = parse_user_input(value)
        if isinstance(parsed, SaveSessionCommand):
            self.action_save_session()
            return
        if isinstance(parsed, QuitAppCommand):
            self.action_quit_app()
            return
        if isinstance(parsed, AgentModeCommand):
            if parsed.mode is None:
                self.action_toggle_agent_mode()
                return
            resolved_profile = normalize_agent_profile(parsed.mode)
            if resolved_profile is None:
                self.notify(
                    "Use /agent, /agent standard, or /agent deep research",
                    title="Agent mode",
                    severity="warning",
                )
                return
            self.action_set_agent_profile(resolved_profile)
            return
        if isinstance(parsed, ModelsCommand):
            if parsed.query is None:
                self.action_show_models()
                return
            model_id = self._resolve_slash_argument_query(parsed.command, parsed.query)
            if model_id is None:
                self.notify(
                    f"No models found for '{parsed.query}'",
                    title="Models",
                    severity="warning",
                )
                return
            self.action_switch_model(model_id)
            return
        if isinstance(parsed, ReasoningCommand):
            if parsed.effort is None:
                self._show_reasoning_status()
                return
            resolved_effort = (
                self._resolve_slash_argument_query(parsed.command, parsed.effort) or parsed.effort
            )
            self._set_reasoning_effort(resolved_effort)
            return
        if isinstance(parsed, ShellCommand):
            text_area.read_only = True
            self._active_shell_worker = self.run_worker(
                self.run_user_command(parsed),
                name="shell-command",
                group="shell-command",
                exit_on_error=False,
            )
            return

        attachments = self._consume_attachments_for_text(value)
        if attachments and not self.current_model_entry.supports_images:
            self.notify(
                f"{self.config.model} does not support image attachments — sending text only",
                title="Images",
                severity="warning",
            )
            attachments = []
        self._record_session_message("user", value)
        chat_view = self.query_one("#chat-view")
        should_follow = self._should_follow_chat_updates()
        text_area.read_only = True
        prompt = self._flush_pending_context(value)
        turn = ConversationTurn(prompt_text=value, include_thinking=True)
        self._active_turn = turn
        _ = await chat_view.mount(turn)
        if should_follow:
            self._scroll_chat_to_end(force=True)
        thinking_output = turn.thinking_output
        if thinking_output is None:
            text_area.read_only = False
            return
        self._active_prompt_worker = self.send_prompt(
            prompt,
            value,
            turn.response_widget,
            thinking_output,
            attachments,
        )

    def _reset_interrupt_escape(self) -> None:
        """Forget any pending first Escape press."""
        self._last_interrupt_escape_at = None

    def _has_interruptible_work(self) -> bool:
        """Return whether an active shell command or prompt request can be interrupted."""
        shell_running = self._active_shell_task is not None and not self._active_shell_task.done()
        shell_worker_running = self._active_shell_worker is not None
        runtime_running = (
            self._active_runtime_task is not None and not self._active_runtime_task.done()
        )
        return (
            shell_running
            or shell_worker_running
            or runtime_running
            or self._active_prompt_worker is not None
        )

    def handle_interrupt_escape(self) -> bool:
        """Handle Escape presses used to interrupt the active request."""
        if not self._has_interruptible_work():
            self._reset_interrupt_escape()
            return False
        now = monotonic()
        previous = self._last_interrupt_escape_at
        self._last_interrupt_escape_at = now
        if previous is None or (now - previous) > self.DOUBLE_ESCAPE_WINDOW_SECONDS:
            self.notify("Press Esc again quickly to interrupt", title="Interrupt")
            return True
        self._reset_interrupt_escape()
        self._interrupt_active_request()
        return True

    def _interrupt_active_request(self) -> None:
        """Interrupt the active runtime request and any running shell process."""
        self._record_session_event("interrupt_requested", {})
        if self._active_shell_task is not None and not self._active_shell_task.done():
            _ = self._active_shell_task.cancel()
        elif self._active_shell_worker is not None:
            self._active_shell_worker.cancel()
        loop = self._active_runtime_loop
        task = self._active_runtime_task
        if loop is not None and task is not None and not task.done():
            _ = loop.call_soon_threadsafe(task.cancel)

    async def run_user_command(self, cmd: ShellCommand) -> None:
        """Execute a direct user shell command and display the output."""
        chat_view = self.query_one("#chat-view")
        should_follow = self._should_follow_chat_updates()
        shell_widget = ShellOutput(f"Running: {cmd.command}")
        section = OutputSection("Shell", "shell-title", shell_widget)
        _ = await chat_view.mount(section)
        if should_follow:
            self._scroll_chat_to_end(force=True)

        text_area = self.prompt_input
        text_area.read_only = True
        self._active_shell_task = asyncio.create_task(execute_bash(cmd.command))

        interrupted = False
        try:
            result = await self._active_shell_task
        except UserInterruptedError as exc:
            interrupted = True
            output = self._format_interrupted_output(exc.partial_output)
            exit_code = None
        except Exception as exc:
            output = f"Error: {exc}"
            exit_code = None
        else:
            output = result.output
            exit_code = result.exit_code
        finally:
            self._active_shell_task = None
            self._reset_interrupt_escape()
            text_area.read_only = False
            _ = text_area.focus()

        shell_widget.set_text(output)
        prefix = "!" if cmd.include_in_context else "!!"
        self._record_session_message("user", f"{prefix}{cmd.command}")
        self._record_session_message("assistant", f"$ {cmd.command}\n\n{output}")

        if interrupted:
            self._record_session_event("shell_command_interrupted", {"command": cmd.command})
            return

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
        active_turn = self._active_turn
        if active_turn is not None:
            active_turn.tool_trace_stack.display = True
            _ = active_turn.tool_trace_stack.mount(section)
        else:
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
        if widget is None:
            widget = ToolOutput()
            section = OutputSection("Tool", "tool-title", widget)
            active_turn = self._active_turn
            if active_turn is not None:
                active_turn.tool_trace_stack.display = True
                _ = active_turn.tool_trace_stack.mount(section)
            else:
                chat_view = self.query_one("#chat-view", VerticalScroll)
                _ = chat_view.mount(section)
        widget.set_text(format_tool_event(tool_name, arguments, status="finished", output=output))
        if should_follow:
            self._scroll_chat_to_end(force=True)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Re-enable input when the active worker finishes."""
        if event.state not in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            return
        worker = cast(Worker[None], event.worker)
        if worker is self._active_prompt_worker:
            self._active_prompt_worker = None
            self._active_turn = None
        if worker is self._active_shell_worker:
            self._active_shell_worker = None
        self._reset_interrupt_escape()
        text_area = self.prompt_input
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

    def _waiting_response_positions(self, message: str | None = None) -> tuple[int, ...]:
        """Return the character positions used by the animated waiting wave."""
        active_message = message or self.RESPONSE_WAITING_MESSAGES[0]
        return tuple(index for index, character in enumerate(active_message) if character.isalnum())

    def _waiting_response_highlight_position(
        self,
        frame_index: int,
        message: str | None = None,
    ) -> int:
        """Return the current highlighted character index for the waiting wave."""
        positions = self._waiting_response_positions(message)
        if len(positions) <= 1:
            return positions[0] if positions else 0
        cycle_length = (len(positions) * 2) - 2
        step = frame_index % cycle_length
        if step < len(positions):
            return positions[step]
        return positions[cycle_length - step]

    def _waiting_response_text(self, frame_index: int, message: str | None = None) -> str:
        """Render the waiting message with a single highlighted character that sweeps back and forth."""
        active_message = message or self.RESPONSE_WAITING_MESSAGES[0]
        highlight_index = self._waiting_response_highlight_position(frame_index, active_message)
        return (
            f"{active_message[:highlight_index]}`{active_message[highlight_index]}`"
            f"{active_message[highlight_index + 1 :]}"
        )

    def _render_response_waiting_frame(self, animation: _ResponseWaitingAnimation) -> None:
        """Show the current animated waiting frame on a response widget."""
        _ = animation.response.add_class("response-awaiting")
        _ = animation.response.update(
            self._waiting_response_text(animation.frame_index, animation.message)
        )

    def _start_response_waiting_animation(self, response: Response) -> None:
        """Begin animating the temporary MU-TH-UR-style waiting line."""
        animation = _ResponseWaitingAnimation(
            response=response,
            message=choice(self.RESPONSE_WAITING_MESSAGES),
        )
        self._response_waiting_animations[id(response)] = animation
        self._render_response_waiting_frame(animation)
        if self._should_follow_chat_updates():
            self._scroll_chat_to_end(force=True)

    def _clear_response_waiting_animation(self, response: Response) -> None:
        """Remove waiting animation classes and stop updating the response widget."""
        _ = self._response_waiting_animations.pop(id(response), None)
        _ = response.remove_class("response-awaiting")

    def _tick_response_waiting_animations(self) -> None:
        """Advance the lightweight waiting animation for any pending responses."""
        if not self._response_waiting_animations:
            return
        should_follow = self._should_follow_chat_updates()
        for animation in self._response_waiting_animations.values():
            animation.frame_index += 1
            self._render_response_waiting_frame(animation)
        if should_follow:
            self._scroll_chat_to_end(force=True)

    async def _update_response_output(self, response: Response, text: str) -> None:
        """Render response text incrementally and keep following only when the user is at the bottom."""
        had_waiting_animation = id(response) in self._response_waiting_animations
        self._clear_response_waiting_animation(response)
        should_follow = self._should_follow_chat_updates()
        current_text = response.raw_markdown
        if had_waiting_animation:
            await response.replace_markdown(text)
        elif text.startswith(current_text):
            await response.append_fragment(text[len(current_text) :])
        else:
            await response.replace_markdown(text)
        if should_follow:
            self._scroll_chat_to_end(force=True)

    def _start_thinking_output(self, thinking_output: ThinkingOutput) -> None:
        """Switch the structured thinking widget into live streaming mode."""
        should_follow = self._should_follow_chat_updates()
        thinking_output.start_streaming()
        if should_follow:
            self._scroll_chat_to_end(force=True)

    def _update_thinking_output(self, thinking_output: ThinkingOutput, text: str) -> None:
        """Render streamed pydantic-ai thinking text in the dedicated thinking widget."""
        should_follow = self._should_follow_chat_updates()
        thinking_output.set_text(text)
        if should_follow:
            self._scroll_chat_to_end(force=True)

    def _finish_thinking_output(self, thinking_output: ThinkingOutput) -> None:
        """Collapse the structured thinking widget back to its preview after streaming ends."""
        should_follow = self._should_follow_chat_updates()
        thinking_output.finish_streaming()
        if should_follow:
            self._scroll_chat_to_end(force=True)

    def _handle_runtime_tool_event(self, event: RuntimeToolEvent) -> None:
        """Mirror runtime tool events into the TUI and session log."""
        if event.phase == "started":
            if self.session_manager is not None:
                self.session_manager.record_tool_call(
                    tool_name=event.tool_name,
                    tool_call_id=event.tool_call_id,
                    arguments=event.arguments,
                )
            _ = self.call_from_thread(
                self._show_tool_started,
                event.tool_name,
                event.tool_call_id,
                event.arguments,
            )
            return

        if self.session_manager is not None:
            self.session_manager.record_tool_result(
                tool_name=event.tool_name,
                tool_call_id=event.tool_call_id,
                arguments=event.arguments,
                output=event.output or "",
                is_error=event.is_error,
            )
        _ = self.call_from_thread(
            self._show_tool_finished,
            event.tool_name,
            event.tool_call_id,
            event.arguments,
            event.output or "",
        )

    def _get_enabled_tools(self) -> list[Tool[None]]:
        """Return the active tool definitions, if any are enabled and available."""
        tool_registry = get_default_tools(
            tools_enabled=self.agent_mode,
            ca_bundle_path=self.config.ca_bundle_path,
            agent_profile=self.agent_profile,
        )
        if tool_registry.is_empty():
            return []
        return tool_registry.tools()

    @staticmethod
    def _tool_names(tools: list[Tool[None]]) -> list[str]:
        """Extract stable prompt-friendly tool names from tool definitions."""
        if not tools:
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
        tools: list[Tool[None]],
        *,
        agent_mode: bool | None = None,
    ) -> str:
        """Build the runtime system prompt for the current model turn."""
        effective_agent_mode = self.agent_mode if agent_mode is None else agent_mode
        effective_mode = self._runtime_mode() if effective_agent_mode else "chat"
        return build_system_prompt(
            self.config.system_prompt,
            mode=effective_mode,
            agent_mode=effective_agent_mode,
            agent_profile=self.agent_profile,
            cwd=Path.cwd(),
            tool_names=self._tool_names(tools),
        )

    def _tool_call_limit(self) -> int | None:
        """Return the per-turn tool-call limit for the active runtime mode."""
        if not self.agent_mode:
            return None
        if self._runtime_mode() == "deep_research":
            return None
        return 1

    async def _run_runtime_request(
        self,
        prompt: str,
        response: Response,
        system: str,
        tools: list[Tool[None]],
        attachments: list[Path],
        thinking_output: ThinkingOutput | None,
    ) -> str | None:
        runtime = ChatRuntime(self.current_model_entry)
        visible_text = ""
        thinking_text = ""
        thinking_streaming = False

        def on_text_update(text: str) -> None:
            nonlocal visible_text, thinking_streaming
            if thinking_output is not None and thinking_streaming and text != visible_text:
                _ = self.call_from_thread(self._finish_thinking_output, thinking_output)
                thinking_streaming = False
            visible_text = text
            _ = self.call_from_thread(self._update_response_output, response, visible_text)

        def on_thinking_update(text: str) -> None:
            nonlocal thinking_text, thinking_streaming
            if thinking_output is None:
                return
            thinking_text = text
            if not thinking_streaming:
                _ = self.call_from_thread(self._start_thinking_output, thinking_output)
                thinking_streaming = True
            _ = self.call_from_thread(self._update_thinking_output, thinking_output, thinking_text)

        try:
            runtime_response = await runtime.run_stream(
                prompt_text=prompt,
                system_prompt=system,
                message_history=self.conversation_state.message_history,
                attachments=attachments,
                tools=tools,
                model_settings=self._reasoning_options(),
                tool_call_limit=self._tool_call_limit(),
                on_text_update=on_text_update,
                on_thinking_update=on_thinking_update if thinking_output is not None else None,
                on_tool_event=self._handle_runtime_tool_event,
            )
        except UserInterruptedError as exc:
            if thinking_streaming:
                assert thinking_output is not None
                _ = self.call_from_thread(self._finish_thinking_output, thinking_output)
            interrupted_text = self._format_interrupted_output(visible_text or exc.partial_output)
            self._show_error(response, interrupted_text)
            self._record_session_event("turn_interrupted", {"agent_mode": self.agent_mode})
            return None
        except Exception as exc:
            self._show_error(response, f"**Error:** {exc}")
            return None

        if thinking_streaming:
            assert thinking_output is not None
            _ = self.call_from_thread(self._finish_thinking_output, thinking_output)

        if (
            runtime_response.text != visible_text
            or id(response) in self._response_waiting_animations
        ):
            visible_text = runtime_response.text
            _ = self.call_from_thread(self._update_response_output, response, visible_text)

        _ = self.call_from_thread(response.stop_stream)
        self.conversation_state.message_history = list(runtime_response.all_messages)
        response.reset_state(visible_text)
        self._record_session_event("turn_usage", runtime_response.usage.to_event_details())
        _ = self.call_from_thread(self._apply_turn_usage, runtime_response.usage)
        if tools and not runtime_response.agent_mode_used:
            _ = self.call_from_thread(self._disable_agent_mode_unsupported)
        return visible_text

    @staticmethod
    def _format_interrupted_output(partial_output: str = "") -> str:
        """Format the user-facing interruption notice, preserving partial output."""
        body = partial_output.rstrip()
        if body:
            return f"{body}\n\n_Interrupted by user._"
        return "_Interrupted by user._"

    def _show_error(self, response: Response, error_text: str) -> None:
        """Display an error in the response widget and reset its state."""
        _ = self.call_from_thread(self._update_response_output, response, error_text)
        _ = self.call_from_thread(response.stop_stream)
        response.reset_state(error_text)

    @work(thread=True)
    def send_prompt(
        self,
        prompt: str,
        user_text: str,
        response: Response,
        thinking_output: ThinkingOutput | None = None,
        attachments: list[Path] | None = None,
    ) -> None:
        """Get the response in a thread, maintaining conversation history."""
        _ = self.call_from_thread(self._start_response_waiting_animation, response)

        tools = self._get_enabled_tools()
        tool_names = self._tool_names(tools)
        system = self._build_system_prompt(tools)
        attachment_paths = [str(path) for path in attachments or []]
        self._record_prompt_context(
            user_text=user_text,
            prompt_text=prompt,
            system_prompt=system,
            tool_names=tool_names,
            attachment_paths=attachment_paths,
        )

        loop = asyncio.new_event_loop()
        task: asyncio.Task[str | None] | None = None
        self._active_runtime_loop = loop
        try:
            asyncio.set_event_loop(loop)
            task = loop.create_task(
                self._run_runtime_request(
                    prompt,
                    response,
                    system,
                    tools,
                    attachments or [],
                    thinking_output,
                )
            )
            self._active_runtime_task = task
            full_text = loop.run_until_complete(task)
        finally:
            self._active_runtime_task = None
            self._active_runtime_loop = None
            loop.close()
            asyncio.set_event_loop(None)

        if full_text is not None:
            self._record_session_message("assistant", full_text)

    def _disable_agent_mode_unsupported(self) -> None:
        """Disable agent mode and notify user that the model doesn't support tools."""
        _ = self._set_agent_mode(enabled=False)
        self._record_session_event(
            "agent_mode_disabled_unsupported",
            {"model": self.config.model, "profile": self.agent_profile},
        )
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
        try:
            output_path = SessionManager.save_last(markdown_dir=Path(config.session_markdown_dir))
        except RuntimeError as exc:
            click.echo(str(exc))
            return
        if output_path is None:
            click.echo("No unsaved session found.")
            return
        click.echo(f"Saved: {output_path}")
        notice = format_markdown_export(output_path)
        if notice is not None:
            click.echo(notice.message)
        return

    if not config.models:
        click.echo(
            'No models configured. Edit ~/.config/mother/config.toml, add at least one [[models]] entry, and set model = "...".'
        )
        return

    if not config.model:
        click.echo('No default model selected. Set model = "..." in ~/.config/mother/config.toml.')
        return

    if resolve_model_entry(config.model, config.models).id != config.model and not any(
        entry.id == config.model for entry in config.models
    ):
        click.echo(
            f"Configured default model {config.model!r} was not found in ~/.config/mother/config.toml."
        )
        return

    session_manager = SessionManager.create(
        markdown_dir=Path(config.session_markdown_dir),
        model_name=config.model,
    )
    app = MotherApp(config=config, session_manager=session_manager)
    app.run()
