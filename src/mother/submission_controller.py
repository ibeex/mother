"""Prompt submission and slash-command dispatch helpers for MotherApp."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from textual.containers import VerticalScroll

from mother.agent_modes import AgentProfile, normalize_agent_profile
from mother.app_session import AppSession
from mother.history import PromptHistory
from mother.models import ModelEntry
from mother.user_commands import (
    AgentModeCommand,
    CouncilCommand,
    ModelsCommand,
    QuitAppCommand,
    ReasoningCommand,
    SaveSessionCommand,
    ShellCommand,
    parse_user_input,
)
from mother.widgets import ConversationTurn, PromptTextArea


@dataclass(frozen=True, slots=True)
class SubmissionControllerCallbacks:
    """Explicit callbacks and state used by prompt-submission workflows."""

    app_session: AppSession
    prompt_history: PromptHistory
    current_model_entry: Callable[[], ModelEntry]
    prompt_input: Callable[[], PromptTextArea]
    notify: Callable[..., None]
    query_one: Callable[..., object]
    should_follow_chat_updates: Callable[[], bool]
    scroll_chat_to_end: Callable[..., None]
    set_active_turn: Callable[[ConversationTurn | None], None]
    set_active_prompt_worker: Callable[[object | None], None]
    set_active_shell_worker: Callable[[object | None], None]
    action_save_session: Callable[[], None]
    action_quit_app: Callable[[], None]
    action_toggle_agent_mode: Callable[[], None]
    action_set_agent_profile: Callable[[AgentProfile], None]
    action_show_models: Callable[[], None]
    action_switch_model: Callable[[str], None]
    resolve_slash_argument_query: Callable[[str, str], str | None]
    resolve_council_models: Callable[[], tuple[tuple[ModelEntry, ...], ModelEntry] | None]
    show_reasoning_status: Callable[[], None]
    set_reasoning_effort: Callable[[str], None]
    run_user_command: Callable[[ShellCommand], object]
    run_worker: Callable[..., object]
    send_prompt: Callable[..., object]
    send_council: Callable[..., object]


class SubmissionController:
    """Encapsulate prompt submission, command dispatch, and turn setup."""

    def __init__(self, callbacks: SubmissionControllerCallbacks) -> None:
        self.callbacks: SubmissionControllerCallbacks = callbacks

    def handle_immediate_submission(self, parsed: object) -> bool:
        """Handle slash commands that only mutate app state and return immediately."""
        if isinstance(parsed, SaveSessionCommand):
            self.callbacks.action_save_session()
            return True
        if isinstance(parsed, QuitAppCommand):
            self.callbacks.action_quit_app()
            return True
        if isinstance(parsed, AgentModeCommand):
            if parsed.mode is None:
                self.callbacks.action_toggle_agent_mode()
                return True
            resolved_profile = normalize_agent_profile(parsed.mode)
            if resolved_profile is None:
                self.callbacks.notify(
                    "Use /agent, /agent standard, or /agent deep research",
                    title="Agent mode",
                    severity="warning",
                )
                return True
            self.callbacks.action_set_agent_profile(resolved_profile)
            return True
        if isinstance(parsed, ModelsCommand):
            if parsed.query is None:
                self.callbacks.action_show_models()
                return True
            model_id = self.callbacks.resolve_slash_argument_query(parsed.command, parsed.query)
            if model_id is None:
                self.callbacks.notify(
                    f"No models found for '{parsed.query}'",
                    title="Models",
                    severity="warning",
                )
                return True
            self.callbacks.action_switch_model(model_id)
            return True
        if isinstance(parsed, ReasoningCommand):
            if parsed.effort is None:
                self.callbacks.show_reasoning_status()
                return True
            resolved_effort = (
                self.callbacks.resolve_slash_argument_query(parsed.command, parsed.effort)
                or parsed.effort
            )
            self.callbacks.set_reasoning_effort(resolved_effort)
            return True
        return False

    def prepare_chat_attachments(self, value: str) -> list[Path]:
        """Resolve referenced image attachments for a normal chat turn."""
        attachments = self.callbacks.app_session.consume_attachments_for_text(value)
        if attachments and not self.callbacks.current_model_entry().supports_images:
            self.callbacks.notify(
                f"{self.callbacks.app_session.config.model} does not support image attachments — sending text only",
                title="Images",
                severity="warning",
            )
            return []
        return attachments

    async def submit_council_command(self, value: str, command: CouncilCommand) -> None:
        """Submit a /council request after validation and UI setup."""
        if command.prompt is None:
            self.callbacks.notify(
                "Usage: /council [question] — add the question inline or on the next line, then press Ctrl+Enter",
                title="Council",
                severity="warning",
            )
            return

        session = self.callbacks.app_session
        council_models = self.callbacks.resolve_council_models()
        if council_models is None:
            return
        council_members, council_judge = council_models

        self.callbacks.prompt_history.append(value)
        discarded_attachments = session.consume_attachments_for_text(value)
        if discarded_attachments:
            self.callbacks.notify(
                "/council does not yet support image attachments — ignoring them",
                title="Council",
                severity="warning",
            )
        session.record_session_message("user", value)
        session.record_session_event(
            "council_invoked",
            {
                "members": [entry.id for entry in council_members],
                "judge": council_judge.id,
                "question": command.prompt,
            },
        )
        chat_view = cast(VerticalScroll, self.callbacks.query_one("#chat-view"))
        should_follow = self.callbacks.should_follow_chat_updates()
        prompt_input = self.callbacks.prompt_input()
        prompt_input.read_only = True
        supplemental_context = session.drain_pending_context_text()
        conversation_context = session.build_council_context()
        turn = ConversationTurn(prompt_text=value, include_thinking=False)
        self.callbacks.set_active_turn(turn)
        _ = await chat_view.mount(turn)
        if should_follow:
            self.callbacks.scroll_chat_to_end(force=True)
        self.callbacks.set_active_prompt_worker(
            self.callbacks.send_council(
                user_question=command.prompt,
                response=turn.response_widget,
                conversation_context=conversation_context,
                supplemental_context=supplemental_context,
                council_members=council_members,
                council_judge=council_judge,
            )
        )

    def submit_shell_command(self, value: str, command: ShellCommand) -> None:
        """Run a direct shell command through the dedicated worker path."""
        if command.include_in_context:
            self.callbacks.prompt_history.append(value)
        prompt_input = self.callbacks.prompt_input()
        prompt_input.read_only = True
        self.callbacks.set_active_shell_worker(
            self.callbacks.run_worker(
                self.callbacks.run_user_command(command),
                name="shell-command",
                group="shell-command",
                exit_on_error=False,
            )
        )

    async def submit_chat_turn(self, value: str) -> None:
        """Submit a normal chat or agent turn."""
        session = self.callbacks.app_session
        self.callbacks.prompt_history.append(value)
        attachments = self.prepare_chat_attachments(value)
        session.record_session_message("user", value)
        chat_view = cast(VerticalScroll, self.callbacks.query_one("#chat-view"))
        should_follow = self.callbacks.should_follow_chat_updates()
        prompt_input = self.callbacks.prompt_input()
        prompt_input.read_only = True
        prompt = session.flush_pending_context(value)
        turn = ConversationTurn(prompt_text=value, include_thinking=True)
        self.callbacks.set_active_turn(turn)
        _ = await chat_view.mount(turn)
        if should_follow:
            self.callbacks.scroll_chat_to_end(force=True)
        thinking_output = turn.thinking_output
        if thinking_output is None:
            prompt_input.read_only = False
            return
        self.callbacks.set_active_prompt_worker(
            self.callbacks.send_prompt(
                prompt,
                value,
                turn.response_widget,
                thinking_output,
                attachments,
            )
        )

    async def submit_current_prompt(self) -> None:
        """Submit the current prompt input using Mother's command-routing rules."""
        prompt_input = self.callbacks.prompt_input()
        value = prompt_input.text.strip()
        if not value:
            return
        _ = prompt_input.clear()

        parsed = parse_user_input(value)
        if self.handle_immediate_submission(parsed):
            return
        if isinstance(parsed, CouncilCommand):
            await self.submit_council_command(value, parsed)
            return
        if isinstance(parsed, ShellCommand):
            self.submit_shell_command(value, parsed)
            return
        await self.submit_chat_turn(value)
