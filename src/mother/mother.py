"""Mother TUI chatbot — a Textual interface for chatting with an LLM."""

from __future__ import annotations

from collections.abc import Coroutine
from pathlib import Path, PurePath
from time import monotonic
from typing import ClassVar, cast, override

import click
from pydantic_ai import Tool
from textual import events, on, work
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding, BindingType
from textual.css.query import NoMatches
from textual.widgets import Footer, Header, OptionList, Static, TextArea
from textual.worker import Worker, WorkerState

from mother.agent_modes import AgentProfile
from mother.app_chrome import (
    StatusLineState,
    build_status_line_state,
    subtitle_text,
    update_status_line,
)
from mother.app_interaction import decide_interrupt_escape
from mother.app_session import AppSession, CouncilModelResolutionError
from mother.app_shell import build_main_pane, build_status_line
from mother.app_wiring import (
    build_runtime_coordinator_callbacks,
    build_settings_controller_callbacks,
    build_submission_controller_callbacks,
)
from mother.clipboard import ClipboardImageError, save_clipboard_image
from mother.config import MotherConfig, apply_cli_overrides, load_config
from mother.conversation import ConversationState
from mother.council import CouncilResult
from mother.history import PromptHistory
from mother.model_picker import AgentModeProvider, ModelPickerScreen, ModelProvider
from mother.models import ModelEntry, resolve_model_entry
from mother.prompt_controller import PromptController, PromptControllerHost
from mother.runtime import RuntimeToolEvent
from mother.runtime_coordinator import RuntimeCoordinator
from mother.runtime_presentation import RuntimePresentationController, RuntimePresentationHost
from mother.runtime_tool_events import handle_runtime_tool_event
from mother.session import SessionManager, format_markdown_export
from mother.session_save import save_session_markdown
from mother.settings_controller import SettingsController
from mother.shell_controller import ShellCommandController, ShellControllerHost
from mother.stats import TurnUsage
from mother.submission_controller import SubmissionController
from mother.tools.bash_capture import BashResult
from mother.tools.bash_executor import execute_bash
from mother.widgets import (
    ConversationTurn,
    CopyableOutput,
    ModelComplete,
    PromptHistoryComplete,
    PromptTextArea,
    Response,
    SlashArgumentComplete,
    SlashComplete,
    StatusLine,
    ThinkingOutput,
)

PromptNavigateEvent = (
    PromptTextArea.SlashNavigate
    | PromptTextArea.SlashArgumentNavigate
    | PromptTextArea.HistorySearchNavigate
)
PromptPopupEvent = (
    PromptTextArea.SlashAccept
    | PromptTextArea.SlashDismiss
    | PromptTextArea.SlashArgumentAccept
    | PromptTextArea.SlashArgumentDismiss
    | PromptTextArea.HistorySearchAccept
    | PromptTextArea.HistorySearchDismiss
)

CSS_DIR = Path(__file__).resolve().parent / "css"
APP_CSS_PATHS: list[str | PurePath] = [
    CSS_DIR / "chat.tcss",
    CSS_DIR / "output.tcss",
    CSS_DIR / "input.tcss",
]


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
        Binding("ctrl+o", "toggle_thinking_widget", "Expand", priority=True),
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
        prompt_history: PromptHistory | None = None,
    ) -> None:
        super().__init__()
        base = config or MotherConfig()
        resolved_config = apply_cli_overrides(base, model_name, system)
        self.app_session: AppSession = AppSession(
            resolved_config,
            session_manager=session_manager,
        )
        self.theme = self.config.theme  # pyright: ignore[reportUnannotatedClassAttribute]
        self.prompt_history: PromptHistory = prompt_history or PromptHistory()
        self.prompt_controller: PromptController = PromptController(
            cast(PromptControllerHost, cast(object, self)),
            prompt_history=self.prompt_history,
        )
        self.auto_scroll_enabled: bool = True
        self.runtime_presentation: RuntimePresentationController = RuntimePresentationController(
            cast(RuntimePresentationHost, cast(object, self)),
            waiting_messages=self.RESPONSE_WAITING_MESSAGES,
        )
        self.runtime_coordinator: RuntimeCoordinator = RuntimeCoordinator(
            build_runtime_coordinator_callbacks(self)
        )
        self.shell_controller: ShellCommandController = ShellCommandController(
            cast(ShellControllerHost, cast(object, self))
        )
        self.settings_controller: SettingsController = SettingsController(
            build_settings_controller_callbacks(self)
        )
        self.submission_controller: SubmissionController = SubmissionController(
            build_submission_controller_callbacks(self)
        )
        self._last_interrupt_escape_at: float | None = None
        self._active_prompt_worker: Worker[None] | None = None

    @property
    def config(self) -> MotherConfig:
        return self.app_session.config

    @config.setter
    def config(self, value: MotherConfig) -> None:
        self.app_session.config = value

    @property
    def current_model_entry(self) -> ModelEntry:
        return self.app_session.current_model_entry

    @current_model_entry.setter
    def current_model_entry(self, value: ModelEntry) -> None:
        self.app_session.current_model_entry = value

    @property
    def conversation_state(self) -> ConversationState:
        return self.app_session.conversation_state

    @conversation_state.setter
    def conversation_state(self, value: ConversationState) -> None:
        self.app_session.conversation_state = value

    @property
    def _active_turn(self) -> ConversationTurn | None:
        return self.runtime_presentation.active_turn

    @_active_turn.setter
    def _active_turn(self, value: ConversationTurn | None) -> None:
        self.runtime_presentation.active_turn = value

    def set_active_turn(self, turn: ConversationTurn | None) -> None:
        """Public adapter used by submission wiring to set the active chat turn."""
        self._active_turn = turn

    def set_active_prompt_worker(self, worker: Worker[None] | None) -> None:
        """Public adapter used by submission wiring to track the active prompt worker."""
        self._active_prompt_worker = worker

    @property
    def session_manager(self) -> SessionManager | None:
        return self.app_session.session_manager

    @session_manager.setter
    def session_manager(self, value: SessionManager | None) -> None:
        self.app_session.session_manager = value

    @property
    def agent_mode(self) -> bool:
        return self.app_session.agent_mode

    @agent_mode.setter
    def agent_mode(self, value: bool) -> None:
        self.app_session.agent_mode = value

    @property
    def agent_profile(self) -> AgentProfile:
        return self.app_session.agent_profile

    @agent_profile.setter
    def agent_profile(self, value: AgentProfile) -> None:
        self.app_session.agent_profile = value

    @override
    def compose(self) -> ComposeResult:
        yield Header()
        yield build_main_pane()
        yield build_status_line(self._status_line_state())
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

    @property
    def prompt_history_help(self) -> Static:
        """Return the prompt-history helper label shown above the picker."""
        return self.query_one("#prompt-history-help", Static)

    @property
    def prompt_history_complete(self) -> PromptHistoryComplete:
        """Return the fuzzy prompt-history popup widget."""
        return self.query_one(PromptHistoryComplete)

    @property
    def prompt_council_help(self) -> Static:
        """Return the multiline ``/council`` helper label shown above the prompt."""
        return self.query_one("#prompt-council-help", Static)

    def action_prompt_history_previous(self) -> None:
        """Recall the previous submitted prompt from persistent history."""
        self.prompt_controller.action_prompt_history_previous()

    def action_prompt_history_next(self) -> None:
        """Move toward newer prompt-history entries or restore the current draft."""
        self.prompt_controller.action_prompt_history_next()

    def action_prompt_history_search(self) -> None:
        """Open fuzzy prompt-history search for the current input or an empty query."""
        self.prompt_controller.action_prompt_history_search()

    def _conversation_has_visible_turns(self) -> bool:
        """Return whether the chat pane currently shows any conversation turns."""
        try:
            chat_view = self.query_one("#chat-view")
        except (NoMatches, ScreenStackError):
            return False
        return any(isinstance(child, ConversationTurn) for child in chat_view.children)

    def _conversation_has_history(self) -> bool:
        """Return whether switching models would discard visible or stored chat context."""
        return (
            self.app_session.has_history
            or self._active_turn is not None
            or self._conversation_has_visible_turns()
        )

    def action_show_models(self) -> None:
        """Open the model picker."""

        def on_model_selected(model_id: str | None) -> None:
            if model_id is not None:
                self.action_switch_model(model_id)

        _ = self.push_screen(ModelPickerScreen(self.config.model), on_model_selected)

    def _resolve_slash_argument_query(self, command: str, query: str) -> str | None:
        """Resolve a slash-command argument query to a concrete completion value."""
        return self.prompt_controller.resolve_slash_argument_query(command, query)

    def action_switch_model(self, model_id: str) -> None:
        """Switch to a different LLM model, asking first if that will clear context."""
        self.settings_controller.action_switch_model(model_id)

    def action_set_agent_profile(self, profile: AgentProfile) -> None:
        """Enable agent mode with a specific profile."""
        self.settings_controller.action_set_agent_profile(profile)

    def action_toggle_agent_mode(self) -> None:
        """Toggle standard agent mode on or disable the current agent mode."""
        self.settings_controller.action_toggle_agent_mode()

    def action_toggle_thinking_widget(self) -> None:
        """Expand focused output, or fall back to the latest expandable output widget."""
        focused = self.focused
        if isinstance(focused, CopyableOutput) and focused.can_toggle_expanded():
            focused.action_toggle_expanded()
            return

        output_widgets = [
            widget for widget in self.query(CopyableOutput) if widget.can_toggle_expanded()
        ]
        if output_widgets:
            output_widgets[-1].action_toggle_expanded()

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
        sub_title = subtitle_text(
            model_name=self.config.model,
            agent_mode=self.agent_mode,
            agent_profile=self.agent_profile,
        )
        self.sub_title = sub_title  # pyright: ignore[reportUnannotatedClassAttribute]

    def _status_reasoning_effort(self) -> str | None:
        """Return the visible reasoning setting for reasoning-capable models."""
        return self.app_session.status_reasoning_effort()

    def _status_line_state(self) -> StatusLineState:
        """Return the current normalized status-line values for the app chrome."""
        return build_status_line_state(
            self.app_session,
            auto_scroll_enabled=self.auto_scroll_enabled,
        )

    def _update_statusline(self) -> None:
        """Update the single-line status bar above the footer."""
        try:
            status_line = self.query_one(StatusLine)
        except (NoMatches, ScreenStackError):
            return
        update_status_line(status_line, self._status_line_state())

    def _apply_turn_usage(self, usage: TurnUsage) -> None:
        """Accumulate normalized turn statistics and refresh the status line."""
        self.app_session.apply_turn_usage(usage)
        self._update_statusline()

    def capture_clipboard_image(self) -> str | None:
        """Save a clipboard image to a temp file and register it as a pending attachment."""
        try:
            path = save_clipboard_image()
        except ClipboardImageError as exc:
            self.notify(str(exc), title="Clipboard", severity="warning")
            return None

        if path is None:
            return None

        self.app_session.pending_image_attachments[str(path)] = path
        self.notify(f"Attached image: {path.name}", title="Clipboard")
        return str(path)

    def _resolve_council_models(self) -> tuple[tuple[ModelEntry, ...], ModelEntry] | None:
        """Resolve configured council members and judge from Mother's model registry."""
        try:
            return self.app_session.resolve_council_models()
        except CouncilModelResolutionError as exc:
            self.notify(str(exc), title="Council", severity="warning")
            return None

    def action_save_session(self) -> None:
        """Export the current session to markdown, overwriting the same file for this session."""
        result = save_session_markdown(
            self.session_manager,
            format_export=format_markdown_export,
        )
        for notification in result.notifications:
            if notification.severity is None:
                self.notify(notification.message, title="Session")
                continue
            self.notify(
                notification.message,
                title="Session",
                severity=notification.severity,
            )

    def action_quit_app(self) -> None:
        """Close the application immediately."""
        self.exit()

    @on(TextArea.Changed)
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Refresh prompt autocomplete helpers as the main prompt changes."""
        if event.text_area is not self.prompt_input:
            return
        self.prompt_controller.handle_text_changed(event.text_area.text)

    @on(PromptTextArea.SlashNavigate)
    @on(PromptTextArea.SlashArgumentNavigate)
    @on(PromptTextArea.HistorySearchNavigate)
    def on_prompt_text_area_navigate(self, event: PromptNavigateEvent) -> None:
        """Route prompt-navigation events to the appropriate popup controller."""
        if isinstance(event, PromptTextArea.SlashNavigate):
            self.prompt_controller.navigate_slash(event.direction)
            return
        if isinstance(event, PromptTextArea.SlashArgumentNavigate):
            self.prompt_controller.navigate_slash_argument(event.direction)
            return
        self.prompt_controller.navigate_history_search(event.direction)

    @on(PromptTextArea.SlashAccept)
    @on(PromptTextArea.SlashDismiss)
    @on(PromptTextArea.SlashArgumentAccept)
    @on(PromptTextArea.SlashArgumentDismiss)
    @on(PromptTextArea.HistorySearchAccept)
    @on(PromptTextArea.HistorySearchDismiss)
    def on_prompt_text_area_popup_event(self, event: PromptPopupEvent) -> None:
        """Route prompt popup accept/dismiss events to the appropriate controller action."""
        text_area = event.text_area
        if isinstance(event, PromptTextArea.SlashAccept) and text_area.slash_complete_active:
            self.prompt_controller.apply_slash_completion()
            return
        if isinstance(event, PromptTextArea.SlashDismiss) and text_area.slash_complete_active:
            self.prompt_controller.hide_slash_complete()
            return
        if (
            isinstance(event, PromptTextArea.SlashArgumentAccept)
            and text_area.slash_argument_complete_active
        ):
            self.prompt_controller.apply_slash_argument_completion()
            return
        if (
            isinstance(event, PromptTextArea.SlashArgumentDismiss)
            and text_area.slash_argument_complete_active
        ):
            self.prompt_controller.hide_slash_argument_complete()
            return
        if (
            isinstance(event, PromptTextArea.HistorySearchAccept)
            and text_area.history_search_active
        ):
            self.prompt_controller.accept_prompt_history_search()
            return
        if (
            isinstance(event, PromptTextArea.HistorySearchDismiss)
            and text_area.history_search_active
        ):
            self.prompt_controller.dismiss_prompt_history_search()
            return

    @on(PromptTextArea.SlashSubmit)
    async def on_prompt_text_area_slash_submit(self, event: PromptTextArea.SlashSubmit) -> None:
        """Submit built-in slash commands with Enter instead of inserting a newline."""
        if event.text_area is self.prompt_input:
            await self.action_submit()

    @on(OptionList.OptionSelected)
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle prompt popup selections from slash and argument helpers."""
        _ = self.prompt_controller.handle_option_selected(event)

    async def action_submit(self) -> None:
        """When the user hits Ctrl+Enter."""
        await self.submission_controller.submit_current_prompt()

    def _reset_interrupt_escape(self) -> None:
        """Forget any pending first Escape press."""
        self._last_interrupt_escape_at = None

    def reset_interrupt_escape(self) -> None:
        """Public adapter used by shell execution to clear pending interrupt state."""
        self._reset_interrupt_escape()

    def execute_shell_command(self, command: str) -> Coroutine[object, object, BashResult]:
        """Public adapter used by shell execution to run one shell command."""
        return execute_bash(command)

    def _has_interruptible_work(self) -> bool:
        """Return whether an active shell command or prompt request can be interrupted."""
        shell_running = self.shell_controller.has_interruptible_work()
        runtime_running = self.runtime_coordinator.has_active_request()
        return shell_running or runtime_running or self._active_prompt_worker is not None

    def handle_interrupt_escape(self) -> bool:
        """Handle Escape presses used to interrupt the active request."""
        decision = decide_interrupt_escape(
            has_interruptible_work=self._has_interruptible_work(),
            now=monotonic(),
            previous_escape_at=self._last_interrupt_escape_at,
            double_escape_window_seconds=self.DOUBLE_ESCAPE_WINDOW_SECONDS,
        )
        self._last_interrupt_escape_at = decision.next_escape_at
        if not decision.handled:
            return False
        if decision.should_notify:
            self.notify("Press Esc again quickly to interrupt", title="Interrupt")
            return True
        if decision.should_interrupt:
            self._interrupt_active_request()
        return True

    def _interrupt_active_request(self) -> None:
        """Interrupt the active runtime request and any running shell process."""
        self.app_session.record_session_event("interrupt_requested", {})
        self.shell_controller.interrupt_active_command()
        self.runtime_coordinator.interrupt_active_request()

    def _show_council_trace(self, result: CouncilResult) -> None:
        """Mount inspectable council stage traces within the active conversation turn."""
        self.runtime_presentation.show_council_trace(result)

    def _finish_worker(self, worker: Worker[None]) -> None:
        """Clear tracked worker state and restore prompt interactivity after completion."""
        if worker is self._active_prompt_worker:
            self._active_prompt_worker = None
            self._active_turn = None
        _ = self.shell_controller.clear_worker_if_active(worker)
        self._reset_interrupt_escape()
        text_area = self.prompt_input
        text_area.read_only = False
        _ = text_area.focus()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Re-enable input when the active worker finishes."""
        if event.state not in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            return
        self._finish_worker(cast(Worker[None], event.worker))

    def _should_follow_chat_updates(self) -> bool:
        """Return whether the next chat update should keep following new output."""
        return self.runtime_presentation.should_follow_chat_updates()

    def _scroll_chat_to_end(self, *, force: bool = False) -> None:
        """Keep the chat view pinned to the latest streamed content."""
        self.runtime_presentation.scroll_chat_to_end(force=force)

    def should_follow_chat_updates(self) -> bool:
        """Public adapter used by presentation helpers that need patchable follow checks."""
        return self._should_follow_chat_updates()

    def scroll_chat_to_end(self, *, force: bool = False) -> None:
        """Public adapter used by presentation helpers that need patchable scrolling."""
        self._scroll_chat_to_end(force=force)

    def _start_response_waiting_animation(
        self,
        response: Response,
        message: str | None = None,
    ) -> None:
        """Begin animating the temporary MU-TH-UR-style waiting line."""
        self.runtime_presentation.start_response_waiting_animation(response, message)

    def _set_response_waiting_message(self, response: Response, message: str) -> None:
        """Update the animated waiting line for an in-flight response."""
        self.runtime_presentation.set_response_waiting_message(response, message)

    def _tick_response_waiting_animations(self) -> None:
        """Advance the lightweight waiting animation for any pending responses."""
        self.runtime_presentation.tick_response_waiting_animations()

    async def _update_response_output(self, response: Response, text: str) -> None:
        """Render response text incrementally and keep following only when the user is at the bottom."""
        await self.runtime_presentation.update_response_output(response, text)

    def _start_thinking_output(self, thinking_output: ThinkingOutput) -> None:
        """Switch the structured thinking widget into live streaming mode."""
        self.runtime_presentation.start_thinking_output(thinking_output)

    def _update_thinking_output(self, thinking_output: ThinkingOutput, text: str) -> None:
        """Render streamed pydantic-ai thinking text in the dedicated thinking widget."""
        self.runtime_presentation.update_thinking_output(thinking_output, text)

    def _finish_thinking_output(self, thinking_output: ThinkingOutput) -> None:
        """Collapse the structured thinking widget back to its preview after streaming ends."""
        self.runtime_presentation.finish_thinking_output(thinking_output)

    def _handle_runtime_tool_event(self, event: RuntimeToolEvent) -> None:
        """Mirror runtime tool events into the TUI and session log."""
        handle_runtime_tool_event(
            event=event,
            session_manager=self.session_manager,
            call_from_thread=self.call_from_thread,
            show_tool_started=self.runtime_presentation.show_tool_started,
            show_tool_finished=self.runtime_presentation.show_tool_finished,
        )

    async def _run_runtime_request(
        self,
        prompt: str,
        response: Response,
        system: str,
        tools: list[Tool[None]],
        attachments: list[Path],
        thinking_output: ThinkingOutput | None,
    ) -> str | None:
        """Run one chat runtime request outside the Textual shell class."""
        return await self.runtime_coordinator.run_runtime_request(
            prompt,
            response,
            system,
            tools,
            attachments,
            thinking_output,
        )

    async def _run_council_request(
        self,
        *,
        user_question: str,
        response: Response,
        conversation_context: str,
        supplemental_context: str,
        council_members: tuple[ModelEntry, ...],
        council_judge: ModelEntry,
    ) -> str | None:
        """Run one /council request outside the Textual shell class."""
        return await self.runtime_coordinator.run_council_request(
            user_question=user_question,
            response=response,
            conversation_context=conversation_context,
            supplemental_context=supplemental_context,
            council_members=council_members,
            council_judge=council_judge,
        )

    @work(thread=True)
    def send_prompt(
        self,
        prompt: str,
        user_text: str,
        response: Response,
        thinking_output: ThinkingOutput | None = None,
        attachments: list[Path] | None = None,
    ) -> None:
        """Get the response in a worker thread while preserving runtime/session state."""
        self.runtime_coordinator.send_prompt(
            prompt,
            user_text,
            response,
            thinking_output,
            attachments,
        )

    @work(thread=True)
    def send_council(
        self,
        *,
        user_question: str,
        response: Response,
        conversation_context: str,
        supplemental_context: str,
        council_members: tuple[ModelEntry, ...],
        council_judge: ModelEntry,
    ) -> None:
        """Run /council in a worker thread without polluting main model history."""
        self.runtime_coordinator.send_council(
            user_question=user_question,
            response=response,
            conversation_context=conversation_context,
            supplemental_context=supplemental_context,
            council_members=council_members,
            council_judge=council_judge,
        )

    def _disable_agent_mode_unsupported(self) -> None:
        """Disable agent mode and notify user that the model doesn't support tools."""
        self.settings_controller.disable_agent_mode_unsupported()


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
