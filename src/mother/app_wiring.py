"""Constructor wiring helpers for MotherApp controller callback bundles."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from mother.agent_modes import AgentProfile
from mother.app_session import AppSession
from mother.council import CouncilResult
from mother.history import PromptHistory
from mother.model_picker import ModelSwitchConfirmScreen
from mother.models import ModelEntry
from mother.runtime import RuntimeToolEvent
from mother.runtime_coordinator import RuntimeCoordinatorCallbacks
from mother.runtime_presentation import RuntimePresentationController
from mother.settings_controller import SettingsController, SettingsControllerCallbacks
from mother.shell_controller import ShellCommandController
from mother.stats import TurnUsage
from mother.submission_controller import SubmissionControllerCallbacks
from mother.user_commands import ShellCommand
from mother.widgets import ConversationTurn, PromptTextArea, Response, ThinkingOutput


class MotherAppWiringHost(Protocol):
    """Minimal MotherApp surface needed to build controller callback bundles."""

    @property
    def app_session(self) -> AppSession: ...

    @property
    def prompt_history(self) -> PromptHistory: ...

    @property
    def runtime_presentation(self) -> RuntimePresentationController: ...

    @property
    def shell_controller(self) -> ShellCommandController: ...

    @property
    def settings_controller(self) -> SettingsController: ...

    @property
    def current_model_entry(self) -> ModelEntry: ...

    @property
    def prompt_input(self) -> PromptTextArea: ...


def _call_app_method(
    app: object,
    method_name: str,
    /,
    *args: object,
    **kwargs: object,
) -> object:
    """Call an app method by name so patched methods are resolved at call time."""
    method = cast(Callable[..., object], getattr(app, method_name))
    return method(*args, **kwargs)


def _notify_app(
    app: object,
    /,
    *args: object,
    **kwargs: object,
) -> None:
    """Call ``app.notify`` with late-bound method lookup for test patchability."""
    _ = _call_app_method(app, "notify", *args, **kwargs)


def _query_one_app(
    app: object,
    selector: object,
    expect_type: object | None = None,
) -> object:
    """Call ``app.query_one`` while preserving its optional expected-type overload."""
    if expect_type is None:
        return _call_app_method(app, "query_one", selector)
    return _call_app_method(app, "query_one", selector, expect_type)


def build_runtime_coordinator_callbacks(app: MotherAppWiringHost) -> RuntimeCoordinatorCallbacks:
    """Build runtime-orchestration callbacks with late-bound app method lookup."""

    def call_from_thread(callback: object, *args: object) -> object:
        return _call_app_method(app, "call_from_thread", callback, *args)

    def update_response_output(response: Response, text: str) -> object:
        return _call_app_method(app, "_update_response_output", response, text)

    def start_response_waiting_animation(
        response: Response,
        message: str | None = None,
    ) -> None:
        _ = _call_app_method(app, "_start_response_waiting_animation", response, message)

    def set_response_waiting_message(response: Response, message: str) -> None:
        _ = _call_app_method(app, "_set_response_waiting_message", response, message)

    def start_thinking_output(thinking_output: ThinkingOutput) -> None:
        _ = _call_app_method(app, "_start_thinking_output", thinking_output)

    def update_thinking_output(thinking_output: ThinkingOutput, text: str) -> None:
        _ = _call_app_method(app, "_update_thinking_output", thinking_output, text)

    def finish_thinking_output(thinking_output: ThinkingOutput) -> None:
        _ = _call_app_method(app, "_finish_thinking_output", thinking_output)

    def show_council_trace(result: CouncilResult) -> None:
        _ = _call_app_method(app, "_show_council_trace", result)

    def handle_runtime_tool_event(event: RuntimeToolEvent) -> None:
        _ = _call_app_method(app, "_handle_runtime_tool_event", event)

    def apply_turn_usage(usage: TurnUsage) -> None:
        _ = _call_app_method(app, "_apply_turn_usage", usage)

    def disable_agent_mode_unsupported() -> None:
        _ = _call_app_method(app, "_disable_agent_mode_unsupported")

    return RuntimeCoordinatorCallbacks(
        app_session=app.app_session,
        runtime_presentation=app.runtime_presentation,
        call_from_thread=call_from_thread,
        update_response_output=update_response_output,
        start_response_waiting_animation=start_response_waiting_animation,
        set_response_waiting_message=set_response_waiting_message,
        start_thinking_output=start_thinking_output,
        update_thinking_output=update_thinking_output,
        finish_thinking_output=finish_thinking_output,
        show_council_trace=show_council_trace,
        handle_runtime_tool_event=handle_runtime_tool_event,
        apply_turn_usage=apply_turn_usage,
        disable_agent_mode_unsupported=disable_agent_mode_unsupported,
    )


def build_settings_controller_callbacks(app: MotherAppWiringHost) -> SettingsControllerCallbacks:
    """Build settings-action callbacks with late-bound app method lookup."""

    def notify(*args: object, **kwargs: object) -> None:
        _notify_app(app, *args, **kwargs)

    def update_subtitle() -> None:
        _ = _call_app_method(app, "_update_subtitle")

    def update_statusline() -> None:
        _ = _call_app_method(app, "_update_statusline")

    def conversation_has_history() -> bool:
        return cast(bool, _call_app_method(app, "_conversation_has_history"))

    def push_model_switch_confirm(
        model_id: str,
        callback: Callable[[bool | None], None],
    ) -> object:
        return _call_app_method(
            app,
            "push_screen",
            ModelSwitchConfirmScreen(model_id),
            callback,
        )

    return SettingsControllerCallbacks(
        app_session=app.app_session,
        notify=notify,
        update_subtitle=update_subtitle,
        update_statusline=update_statusline,
        conversation_has_history=conversation_has_history,
        push_model_switch_confirm=push_model_switch_confirm,
    )


def build_submission_controller_callbacks(
    app: MotherAppWiringHost,
) -> SubmissionControllerCallbacks:
    """Build prompt-submission callbacks with late-bound app method lookup."""

    def current_model_entry() -> ModelEntry:
        return app.current_model_entry

    def prompt_input() -> PromptTextArea:
        return app.prompt_input

    def notify(*args: object, **kwargs: object) -> None:
        _notify_app(app, *args, **kwargs)

    def query_one(selector: object, expect_type: object | None = None) -> object:
        return _query_one_app(app, selector, expect_type)

    def should_follow_chat_updates() -> bool:
        return cast(bool, _call_app_method(app, "_should_follow_chat_updates"))

    def scroll_chat_to_end(*, force: bool = False) -> None:
        _ = _call_app_method(app, "_scroll_chat_to_end", force=force)

    def set_active_turn(turn: ConversationTurn | None) -> None:
        _ = _call_app_method(app, "set_active_turn", turn)

    def set_active_prompt_worker(worker: object | None) -> None:
        _ = _call_app_method(app, "set_active_prompt_worker", worker)

    def set_active_shell_worker(worker: object | None) -> None:
        app.shell_controller.set_active_worker(worker)

    def action_save_session() -> None:
        _ = _call_app_method(app, "action_save_session")

    def action_quit_app() -> None:
        _ = _call_app_method(app, "action_quit_app")

    def action_toggle_agent_mode() -> None:
        _ = _call_app_method(app, "action_toggle_agent_mode")

    def action_set_agent_profile(profile: AgentProfile) -> None:
        _ = _call_app_method(app, "action_set_agent_profile", profile)

    def action_show_models() -> None:
        _ = _call_app_method(app, "action_show_models")

    def action_switch_model(model_id: str) -> None:
        _ = _call_app_method(app, "action_switch_model", model_id)

    def resolve_slash_argument_query(command: str, query: str) -> str | None:
        return cast(
            str | None, _call_app_method(app, "_resolve_slash_argument_query", command, query)
        )

    def resolve_council_models() -> tuple[tuple[ModelEntry, ...], ModelEntry] | None:
        return cast(
            tuple[tuple[ModelEntry, ...], ModelEntry] | None,
            _call_app_method(app, "_resolve_council_models"),
        )

    def show_reasoning_status() -> None:
        app.settings_controller.show_reasoning_status()

    def set_reasoning_effort(effort: str) -> None:
        app.settings_controller.set_reasoning_effort(effort)

    def run_user_command(cmd: ShellCommand) -> object:
        return app.shell_controller.run_user_command(cmd)

    def run_worker(
        work: object,
        *,
        name: str,
        group: str,
        exit_on_error: bool,
    ) -> object:
        return _call_app_method(
            app,
            "run_worker",
            work,
            name=name,
            group=group,
            exit_on_error=exit_on_error,
        )

    def send_prompt(
        prompt: str,
        user_text: str,
        response: Response,
        thinking_output: ThinkingOutput | None = None,
        attachments: list[Path] | None = None,
    ) -> object:
        return _call_app_method(
            app,
            "send_prompt",
            prompt,
            user_text,
            response,
            thinking_output,
            attachments,
        )

    def send_council(
        *,
        user_question: str,
        response: Response,
        conversation_context: str,
        supplemental_context: str,
        council_members: tuple[ModelEntry, ...],
        council_judge: ModelEntry,
    ) -> object:
        return _call_app_method(
            app,
            "send_council",
            user_question=user_question,
            response=response,
            conversation_context=conversation_context,
            supplemental_context=supplemental_context,
            council_members=council_members,
            council_judge=council_judge,
        )

    return SubmissionControllerCallbacks(
        app_session=app.app_session,
        prompt_history=app.prompt_history,
        current_model_entry=current_model_entry,
        prompt_input=prompt_input,
        notify=notify,
        query_one=query_one,
        should_follow_chat_updates=should_follow_chat_updates,
        scroll_chat_to_end=scroll_chat_to_end,
        set_active_turn=set_active_turn,
        set_active_prompt_worker=set_active_prompt_worker,
        set_active_shell_worker=set_active_shell_worker,
        action_save_session=action_save_session,
        action_quit_app=action_quit_app,
        action_toggle_agent_mode=action_toggle_agent_mode,
        action_set_agent_profile=action_set_agent_profile,
        action_show_models=action_show_models,
        action_switch_model=action_switch_model,
        resolve_slash_argument_query=resolve_slash_argument_query,
        resolve_council_models=resolve_council_models,
        show_reasoning_status=show_reasoning_status,
        set_reasoning_effort=set_reasoning_effort,
        run_user_command=run_user_command,
        run_worker=run_worker,
        send_prompt=send_prompt,
        send_council=send_council,
    )
