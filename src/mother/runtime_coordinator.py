"""Runtime request and council orchestration helpers for MotherApp."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TypeVar, cast

from pydantic_ai import Tool

from mother.app_session import AppSession
from mother.council import CouncilProgressUpdate, CouncilResult, CouncilRunner
from mother.interrupts import UserInterruptedError
from mother.models import ModelEntry
from mother.runtime import ChatRuntime, RuntimePartialRunError, RuntimeToolEvent
from mother.runtime_presentation import RuntimePresentationController
from mother.stats import TurnUsage
from mother.widgets import Response, ThinkingOutput

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RuntimeCoordinatorCallbacks:
    """Explicit callbacks and state used by runtime request orchestration."""

    app_session: AppSession
    runtime_presentation: RuntimePresentationController
    call_from_thread: Callable[..., object]
    update_response_output: Callable[[Response, str], object]
    start_response_waiting_animation: Callable[[Response, str | None], None]
    set_response_waiting_message: Callable[[Response, str], None]
    start_thinking_output: Callable[[ThinkingOutput], None]
    update_thinking_output: Callable[[ThinkingOutput, str], None]
    finish_thinking_output: Callable[[ThinkingOutput], None]
    show_council_trace: Callable[[CouncilResult], None]
    handle_runtime_tool_event: Callable[[RuntimeToolEvent], None]
    apply_turn_usage: Callable[[TurnUsage], None]
    disable_agent_mode_unsupported: Callable[[], None]


@dataclass(slots=True)
class _RuntimeStreamState:
    """Track mutable state for one streamed runtime request."""

    response: Response
    thinking_output: ThinkingOutput | None
    visible_text: str = ""
    thinking_text: str = ""
    thinking_streaming: bool = False
    blocked_bash_notice: str | None = None


class RuntimeCoordinator:
    """Own async runtime and council request orchestration outside the UI shell."""

    def __init__(self, callbacks: RuntimeCoordinatorCallbacks) -> None:
        self.callbacks: RuntimeCoordinatorCallbacks = callbacks
        self._active_runtime_loop: asyncio.AbstractEventLoop | None = None
        self._active_runtime_task: asyncio.Task[object] | None = None

    def has_active_request(self) -> bool:
        """Return whether a runtime or council request task is currently active."""
        return self._active_runtime_task is not None and not self._active_runtime_task.done()

    def interrupt_active_request(self) -> None:
        """Cancel the active runtime request from the main thread if one is running."""
        loop = self._active_runtime_loop
        task = self._active_runtime_task
        if loop is None or task is None or task.done():
            return
        _ = loop.call_soon_threadsafe(task.cancel)

    @staticmethod
    def _blocked_bash_clipboard_notice(event: RuntimeToolEvent) -> str | None:
        """Return a deterministic user-facing notice for blocked bash commands."""
        if event.phase != "finished" or event.tool_name != "bash":
            return None

        output = event.output or ""
        if "bash guard blocked this command. It was not executed." not in output:
            return None
        if "The exact command has been copied to the clipboard." not in output:
            return None

        return (
            "Note: the exact blocked command is already in your clipboard. "
            "Review it, then either paste it into a separate shell, or run it here with "
            "`!<command>` to include the output in chat context or `!!<command>` to exclude it."
        )

    @staticmethod
    def _append_notice_if_missing(text: str, notice: str) -> str:
        """Append a notice unless the response already mentions the clipboard."""
        stripped = text.rstrip()
        if "clipboard" in stripped.casefold():
            return stripped
        if not stripped:
            return notice
        return f"{stripped}\n\n{notice}"

    @staticmethod
    def _format_interrupted_output(partial_output: str = "") -> str:
        """Format the user-facing interruption notice, preserving partial output."""
        body = partial_output.rstrip()
        if body:
            return f"{body}\n\n_Interrupted by user._"
        return "_Interrupted by user._"

    def _runtime_on_text_update(self, stream_state: _RuntimeStreamState, text: str) -> None:
        """Handle streamed visible-text updates for a runtime request."""
        if (
            stream_state.thinking_output is not None
            and stream_state.thinking_streaming
            and text != stream_state.visible_text
        ):
            _ = self.callbacks.call_from_thread(
                self.callbacks.finish_thinking_output,
                stream_state.thinking_output,
            )
            stream_state.thinking_streaming = False
        stream_state.visible_text = text
        _ = self.callbacks.call_from_thread(
            self.callbacks.update_response_output,
            stream_state.response,
            stream_state.visible_text,
        )

    def _runtime_on_thinking_update(
        self,
        stream_state: _RuntimeStreamState,
        text: str,
    ) -> None:
        """Handle streamed reasoning/thinking updates for a runtime request."""
        if stream_state.thinking_output is None:
            return
        stream_state.thinking_text = text
        if not stream_state.thinking_streaming:
            _ = self.callbacks.call_from_thread(
                self.callbacks.start_thinking_output,
                stream_state.thinking_output,
            )
            stream_state.thinking_streaming = True
        _ = self.callbacks.call_from_thread(
            self.callbacks.update_thinking_output,
            stream_state.thinking_output,
            stream_state.thinking_text,
        )

    def _runtime_on_tool_event(
        self,
        stream_state: _RuntimeStreamState,
        event: RuntimeToolEvent,
    ) -> None:
        """Handle tool events emitted while a runtime request is running."""
        if stream_state.blocked_bash_notice is None:
            stream_state.blocked_bash_notice = self._blocked_bash_clipboard_notice(event)
        self.callbacks.handle_runtime_tool_event(event)

    def _finish_runtime_thinking_if_needed(self, stream_state: _RuntimeStreamState) -> None:
        """Finalize any in-progress thinking widget stream."""
        if not stream_state.thinking_streaming or stream_state.thinking_output is None:
            return
        _ = self.callbacks.call_from_thread(
            self.callbacks.finish_thinking_output,
            stream_state.thinking_output,
        )
        stream_state.thinking_streaming = False

    def _show_error(self, response: Response, error_text: str) -> None:
        """Display an error in the response widget and reset its state."""
        _ = self.callbacks.call_from_thread(
            self.callbacks.update_response_output, response, error_text
        )
        _ = self.callbacks.call_from_thread(response.stop_stream)
        response.reset_state(error_text)

    async def run_runtime_request(
        self,
        prompt: str,
        response: Response,
        system: str,
        tools: list[Tool[None]],
        attachments: list[Path],
        thinking_output: ThinkingOutput | None,
    ) -> str | None:
        """Run one chat runtime request and coordinate streamed UI/state updates."""
        session = self.callbacks.app_session
        model_entry = session.current_model_entry
        runtime = ChatRuntime(model_entry)
        stream_state = _RuntimeStreamState(response, thinking_output)

        try:
            runtime_response = await runtime.run_stream(
                prompt_text=prompt,
                system_prompt=system,
                message_history=session.conversation_state.message_history,
                attachments=attachments,
                tools=tools,
                model_settings=session.reasoning_options(),
                tool_call_limit=session.tool_call_limit(),
                on_text_update=partial(self._runtime_on_text_update, stream_state),
                on_thinking_update=(
                    partial(self._runtime_on_thinking_update, stream_state)
                    if thinking_output is not None
                    else None
                ),
                on_tool_event=partial(self._runtime_on_tool_event, stream_state),
            )
        except UserInterruptedError as exc:
            self._finish_runtime_thinking_if_needed(stream_state)
            interrupted_text = self._format_interrupted_output(
                stream_state.visible_text or exc.partial_output
            )
            self._show_error(response, interrupted_text)
            session.record_session_event("turn_interrupted", {"agent_mode": session.agent_mode})
            return None
        except RuntimePartialRunError as exc:
            self._finish_runtime_thinking_if_needed(stream_state)
            session.conversation_state.message_history = list(exc.partial_messages)
            self._show_error(response, f"**Error:** {exc.cause}")
            logger.exception(
                "Runtime request preserved partial history after failure for model %s",
                model_entry.id,
            )
            return None
        except Exception as exc:
            self._finish_runtime_thinking_if_needed(stream_state)
            self._show_error(response, f"**Error:** {exc}")
            logger.exception(
                "Runtime request failed for model %s (tools=%s attachments=%d)",
                model_entry.id,
                [tool.name for tool in tools],
                len(attachments),
            )
            return None

        self._finish_runtime_thinking_if_needed(stream_state)

        final_text = runtime_response.text
        if stream_state.blocked_bash_notice is not None:
            final_text = self._append_notice_if_missing(
                final_text,
                stream_state.blocked_bash_notice,
            )

        if (
            final_text != stream_state.visible_text
            or self.callbacks.runtime_presentation.has_waiting_animation(response)
        ):
            stream_state.visible_text = final_text
            _ = self.callbacks.call_from_thread(
                self.callbacks.update_response_output,
                response,
                stream_state.visible_text,
            )

        _ = self.callbacks.call_from_thread(response.stop_stream)
        session.conversation_state.message_history = list(runtime_response.all_messages)
        response.reset_state(stream_state.visible_text)
        if runtime_response.tool_limit_recovery_used:
            session.record_session_event(
                "tool_limit_recovery",
                {
                    "strategy": "text_only",
                    "model": model_entry.id,
                    "mode": session.runtime_mode(),
                    "profile": session.agent_profile,
                    "tool_call_limit": session.tool_call_limit(),
                    "tool_calls_started": runtime_response.usage.tool_calls_started,
                    "tool_calls_finished": runtime_response.usage.tool_calls_finished,
                },
            )
        session.record_session_event("turn_usage", runtime_response.usage.to_event_details())
        _ = self.callbacks.call_from_thread(self.callbacks.apply_turn_usage, runtime_response.usage)
        if tools and not runtime_response.agent_mode_used:
            _ = self.callbacks.call_from_thread(self.callbacks.disable_agent_mode_unsupported)
        return stream_state.visible_text

    async def run_council_request(
        self,
        *,
        user_question: str,
        response: Response,
        conversation_context: str,
        supplemental_context: str,
        council_members: tuple[ModelEntry, ...],
        council_judge: ModelEntry,
    ) -> str | None:
        """Run one /council request and synthesize its main-turn result."""
        session = self.callbacks.app_session
        config = session.config

        def on_progress(update: CouncilProgressUpdate) -> None:
            _ = self.callbacks.call_from_thread(
                self.callbacks.set_response_waiting_message,
                response,
                update.status_text(),
            )

        runner = CouncilRunner(
            members=council_members,
            judge=council_judge,
            base_system_prompt=config.system_prompt,
            reasoning_effort=config.reasoning_effort,
            openai_reasoning_summary=config.openai_reasoning_summary,
            cwd=Path.cwd(),
            on_progress=on_progress,
        )
        try:
            result = await runner.run(
                user_question=user_question,
                conversation_context=conversation_context,
                supplemental_context=supplemental_context,
            )
        except UserInterruptedError:
            interrupted_text = self._format_interrupted_output()
            self._show_error(response, interrupted_text)
            session.record_session_event(
                "council_interrupted",
                {
                    "members": [entry.id for entry in council_members],
                    "judge": council_judge.id,
                    "question": user_question,
                },
            )
            return None
        except Exception as exc:
            self._show_error(response, f"**Error:** {exc}")
            logger.exception(
                "Council request failed (judge=%s members=%s)",
                council_judge.id,
                [entry.id for entry in council_members],
            )
            session.record_session_event(
                "council_failed",
                {
                    "members": [entry.id for entry in council_members],
                    "judge": council_judge.id,
                    "question": user_question,
                    "error": str(exc),
                },
            )
            return None

        final_text = result.final_text
        if final_text or self.callbacks.runtime_presentation.has_waiting_animation(response):
            _ = self.callbacks.call_from_thread(
                self.callbacks.update_response_output, response, final_text
            )
        _ = self.callbacks.call_from_thread(self.callbacks.show_council_trace, result)
        _ = self.callbacks.call_from_thread(response.stop_stream)
        response.reset_state(final_text)
        session.conversation_state.append_synthetic_turn(user_question, final_text)
        session.record_session_message("assistant", final_text)
        session.record_session_event("council_completed", result.to_event_details())
        return final_text

    def _run_async_request(self, request: Coroutine[object, object, T]) -> T:
        """Run one async request inside a dedicated worker-thread event loop."""
        loop = asyncio.new_event_loop()
        task: asyncio.Task[T] | None = None
        self._active_runtime_loop = loop
        try:
            asyncio.set_event_loop(loop)
            task = loop.create_task(request)
            self._active_runtime_task = cast(asyncio.Task[object] | None, task)
            return loop.run_until_complete(task)
        finally:
            self._active_runtime_task = None
            self._active_runtime_loop = None
            loop.close()
            asyncio.set_event_loop(None)

    def send_prompt(
        self,
        prompt: str,
        user_text: str,
        response: Response,
        thinking_output: ThinkingOutput | None = None,
        attachments: list[Path] | None = None,
    ) -> None:
        """Run a chat/agent turn inside the worker thread and persist the result."""
        session = self.callbacks.app_session
        resolved_attachments = attachments or []

        _ = self.callbacks.call_from_thread(
            self.callbacks.start_response_waiting_animation,
            response,
        )

        prompt_text = session.expand_prompt_fetch_directives(prompt, user_text)
        tools = session.enabled_tools()
        tool_names = session.tool_names(tools)
        system = session.build_system_prompt(tools)
        attachment_paths = [str(path) for path in resolved_attachments]
        session.record_prompt_context(
            user_text=user_text,
            prompt_text=prompt_text,
            system_prompt=system,
            tool_names=tool_names,
            attachment_paths=attachment_paths,
        )

        full_text = self._run_async_request(
            self.run_runtime_request(
                prompt_text,
                response,
                system,
                tools,
                resolved_attachments,
                thinking_output,
            )
        )

        if full_text is not None:
            session.conversation_state.append_transcript_turn(user_text, full_text)
            session.record_session_message("assistant", full_text)

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
        """Run /council inside the worker thread without polluting model history."""
        initial_progress = CouncilProgressUpdate.stage1(0, len(council_members))
        _ = self.callbacks.call_from_thread(
            self.callbacks.start_response_waiting_animation,
            response,
            initial_progress.status_text(),
        )

        _ = self._run_async_request(
            self.run_council_request(
                user_question=user_question,
                response=response,
                conversation_context=conversation_context,
                supplemental_context=supplemental_context,
                council_members=council_members,
                council_judge=council_judge,
            )
        )
