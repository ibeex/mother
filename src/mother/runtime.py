"""pydantic-ai runtime adapter used by Mother's TUI."""

from __future__ import annotations

import asyncio
import mimetypes
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import wraps
from inspect import Parameter, isawaitable, signature
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

from pydantic_ai import Agent, AgentRunResultEvent, Tool, capture_run_messages
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolReturnPart,
    UserContent,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from mother.interrupts import UserInterruptedError
from mother.models import ModelEntry, create_pydantic_model
from mother.stats import TurnUsage

_TEXT_ONLY_TOOL_LIMIT_RECOVERY_PROMPT = (
    "Reply to the user in plain text only using the completed tool result. Do not call tools."
)


@dataclass(frozen=True, slots=True)
class RuntimeToolEvent:
    phase: Literal["started", "finished"]
    tool_name: str
    tool_call_id: str | None
    arguments: dict[str, object]
    output: str | None = None
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeResponse:
    text: str
    all_messages: list[ModelMessage]
    usage: TurnUsage
    agent_mode_used: bool
    tool_limit_recovery_used: bool = False


@dataclass(slots=True)
class _ToolState:
    sequence: int = 0
    started: int = 0
    finished: int = 0
    errors: int = 0

    def next_call_id(self, tool_name: str) -> str:
        self.sequence += 1
        return f"{tool_name}-{self.sequence}"


@dataclass(slots=True)
class _StreamProgress:
    text: str = ""
    thinking: str = ""

    def append_text(
        self,
        content: str | None,
        callback: Callable[[str], None] | None,
    ) -> None:
        if not content:
            return
        self.text += content
        if callback is not None:
            callback(self.text)

    def append_thinking(
        self,
        content: str | None,
        callback: Callable[[str], None] | None,
    ) -> None:
        if not content:
            return
        self.thinking += content
        if callback is not None:
            callback(self.thinking)


@dataclass(slots=True)
class _CapturedRunMessages:
    messages: list[ModelMessage] = field(default_factory=list)


class RuntimePartialRunError(Exception):
    """Runtime error carrying safe partial history from a failed run."""

    cause: Exception
    partial_messages: list[ModelMessage]

    def __init__(self, cause: Exception, partial_messages: list[ModelMessage]) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.partial_messages = partial_messages


class ChatRuntime:
    """Thin adapter around pydantic-ai streaming APIs."""

    def __init__(self, model_entry: ModelEntry) -> None:
        self.model_entry: ModelEntry = model_entry

    @staticmethod
    def _guess_media_type(path: Path) -> str:
        guessed, _ = mimetypes.guess_type(path.name)
        return guessed or "application/octet-stream"

    def _build_user_prompt(
        self,
        prompt_text: str,
        attachments: list[Path],
    ) -> str | list[UserContent]:
        if not attachments:
            return prompt_text

        parts: list[UserContent] = [prompt_text]
        for path in attachments:
            parts.append(
                BinaryContent(
                    path.read_bytes(),
                    media_type=self._guess_media_type(path),
                    identifier=path.name,
                )
            )
        return parts

    @staticmethod
    def tool_arguments(
        tool: Tool[None], args: tuple[object, ...], kwargs: dict[str, object]
    ) -> dict[str, object]:
        tool_signature = signature(tool.function)
        bound = tool_signature.bind_partial(*args, **kwargs)
        raw_arguments = cast(dict[str, object], bound.arguments)
        arguments = dict(raw_arguments)
        if tool.takes_ctx:
            _ = arguments.pop("ctx", None)

        filtered_arguments: dict[str, object] = {}
        for name, value in arguments.items():
            parameter = tool_signature.parameters.get(name)
            if parameter is None:
                filtered_arguments[name] = value
                continue
            default_value = cast(object, parameter.default)
            if default_value is not Parameter.empty and value == default_value:
                continue
            filtered_arguments[name] = value
        return filtered_arguments

    @staticmethod
    async def _call_tool_function(
        original: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        result = original(*args, **kwargs)
        if isawaitable(result):
            return await cast(Awaitable[object], result)
        return result

    @staticmethod
    def _tool_output(result: object) -> str:
        return result if isinstance(result, str) else str(result)

    @staticmethod
    def _emit_tool_event(
        on_tool_event: Callable[[RuntimeToolEvent], None] | None,
        *,
        phase: Literal["started", "finished"],
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, object],
        output: str | None = None,
        is_error: bool = False,
    ) -> None:
        if on_tool_event is None:
            return
        _ = on_tool_event(
            RuntimeToolEvent(
                phase=phase,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                output=output,
                is_error=is_error,
            )
        )

    @staticmethod
    def _wrapped_tool(
        tool: Tool[None],
        *,
        on_tool_event: Callable[[RuntimeToolEvent], None] | None,
        tool_state: _ToolState,
    ) -> Tool[None]:
        original = cast(Callable[..., object], tool.function)

        @wraps(original)
        async def wrapped(*args: object, **kwargs: object) -> object:
            arguments = ChatRuntime.tool_arguments(tool, args, kwargs)
            call_id = tool_state.next_call_id(tool.name)
            tool_state.started += 1
            ChatRuntime._emit_tool_event(
                on_tool_event,
                phase="started",
                tool_name=tool.name,
                tool_call_id=call_id,
                arguments=arguments,
            )
            try:
                result = await ChatRuntime._call_tool_function(original, *args, **kwargs)
            except Exception as exc:
                tool_state.errors += 1
                tool_state.finished += 1
                ChatRuntime._emit_tool_event(
                    on_tool_event,
                    phase="finished",
                    tool_name=tool.name,
                    tool_call_id=call_id,
                    arguments=arguments,
                    output=str(exc),
                    is_error=True,
                )
                raise

            tool_state.finished += 1
            ChatRuntime._emit_tool_event(
                on_tool_event,
                phase="finished",
                tool_name=tool.name,
                tool_call_id=call_id,
                arguments=arguments,
                output=ChatRuntime._tool_output(result),
            )
            return result

        return Tool(
            wrapped,
            takes_ctx=tool.takes_ctx,
            name=tool.name,
            description=tool.description,
            max_retries=tool.max_retries,
            docstring_format=tool.docstring_format,
            require_parameter_descriptions=tool.require_parameter_descriptions,
            strict=tool.strict,
            sequential=tool.sequential,
            requires_approval=tool.requires_approval,
            metadata=tool.metadata,
            timeout=tool.timeout,
        )

    @staticmethod
    def _is_unsupported_tools_error(error: Exception) -> bool:
        message = str(error).lower()
        return "does not support tools" in message or (
            "tool" in message and ("unsupported" in message or "not support" in message)
        )

    def _effective_model_settings(
        self,
        *,
        model_settings: dict[str, object],
        wrapped_tools: list[Tool[None]],
    ) -> dict[str, object]:
        effective_settings = dict(model_settings)
        if not wrapped_tools:
            return effective_settings
        if self.model_entry.api_type not in {"openai-chat", "openai-responses"}:
            return effective_settings
        effective_settings["parallel_tool_calls"] = False
        return effective_settings

    @staticmethod
    def _has_retryable_tool_result(message: ModelRequest) -> bool:
        return any(isinstance(part, ToolReturnPart | RetryPromptPart) for part in message.parts)

    @staticmethod
    def _has_tool_return(message: ModelRequest) -> bool:
        return any(isinstance(part, ToolReturnPart) for part in message.parts)

    @staticmethod
    def _preserve_partial_messages(
        messages: list[ModelMessage], error: Exception
    ) -> list[ModelMessage] | None:
        if not isinstance(error, UsageLimitExceeded):
            return None

        preserved = list(messages)
        while preserved:
            last_message = preserved[-1]
            if isinstance(last_message, ModelResponse) and last_message.tool_calls:
                _ = preserved.pop()
                continue
            break

        if not preserved:
            return None

        last_message = preserved[-1]
        if not isinstance(last_message, ModelRequest):
            return None
        if not ChatRuntime._has_retryable_tool_result(last_message):
            return None
        return preserved

    @staticmethod
    def _should_retry_text_only_after_tool_limit(
        error: Exception,
        *,
        partial_messages: list[ModelMessage] | None,
        wrapped_tools: list[Tool[None]],
        tool_call_limit: int | None,
    ) -> bool:
        if not isinstance(error, UsageLimitExceeded):
            return False
        if not wrapped_tools or tool_call_limit != 1 or not partial_messages:
            return False

        last_message = partial_messages[-1]
        if not isinstance(last_message, ModelRequest):
            return False
        return ChatRuntime._has_tool_return(last_message)

    @staticmethod
    def _process_part_start(
        event: PartStartEvent,
        *,
        progress: _StreamProgress,
        on_text_update: Callable[[str], None] | None,
        on_thinking_update: Callable[[str], None] | None,
    ) -> None:
        if isinstance(event.part, ThinkingPart):
            progress.append_thinking(event.part.content, on_thinking_update)
            return
        if isinstance(event.part, TextPart):
            progress.append_text(event.part.content, on_text_update)

    @staticmethod
    def _process_part_delta(
        event: PartDeltaEvent,
        *,
        progress: _StreamProgress,
        on_text_update: Callable[[str], None] | None,
        on_thinking_update: Callable[[str], None] | None,
    ) -> None:
        if isinstance(event.delta, ThinkingPartDelta):
            progress.append_thinking(event.delta.content_delta, on_thinking_update)
            return
        if isinstance(event.delta, TextPartDelta):
            progress.append_text(event.delta.content_delta, on_text_update)

    @staticmethod
    def _process_stream_event(
        event: object,
        *,
        progress: _StreamProgress,
        on_text_update: Callable[[str], None] | None,
        on_thinking_update: Callable[[str], None] | None,
    ) -> AgentRunResultEvent[str] | None:
        if isinstance(event, AgentRunResultEvent):
            return cast(AgentRunResultEvent[str], event)
        if isinstance(event, PartStartEvent):
            ChatRuntime._process_part_start(
                event,
                progress=progress,
                on_text_update=on_text_update,
                on_thinking_update=on_thinking_update,
            )
            return None
        if isinstance(event, PartDeltaEvent):
            ChatRuntime._process_part_delta(
                event,
                progress=progress,
                on_text_update=on_text_update,
                on_thinking_update=on_thinking_update,
            )
        return None

    async def _collect_stream_result(
        self,
        *,
        agent: Agent,
        user_prompt: str | list[UserContent],
        message_history: list[ModelMessage],
        effective_model_settings: dict[str, object],
        usage_limits: UsageLimits | None,
        captured_messages_ref: _CapturedRunMessages,
        on_text_update: Callable[[str], None] | None,
        on_thinking_update: Callable[[str], None] | None,
    ) -> tuple[AgentRunResultEvent[str], _StreamProgress]:
        progress = _StreamProgress()

        with capture_run_messages() as captured_messages:
            captured_messages_ref.messages = captured_messages
            async for event in agent.run_stream_events(
                user_prompt,
                message_history=message_history,
                model_settings=cast(ModelSettings, cast(object, effective_model_settings)),
                usage_limits=usage_limits,
            ):
                final_result = self._process_stream_event(
                    event,
                    progress=progress,
                    on_text_update=on_text_update,
                    on_thinking_update=on_thinking_update,
                )
                if final_result is not None:
                    return final_result, progress

        raise RuntimeError("Agent stream ended without a final result")

    @staticmethod
    def _flush_final_updates(
        *,
        final_text: str,
        final_thinking: str,
        progress: _StreamProgress,
        on_text_update: Callable[[str], None] | None,
        on_thinking_update: Callable[[str], None] | None,
    ) -> None:
        if final_thinking != progress.thinking and on_thinking_update is not None:
            on_thinking_update(final_thinking)
        if final_text != progress.text and on_text_update is not None:
            on_text_update(final_text)

    def _build_runtime_response(
        self,
        *,
        final_result: AgentRunResultEvent[str],
        progress: _StreamProgress,
        attachments: list[Path],
        started_at: float,
        tool_state: _ToolState,
        agent_mode_used: bool,
        on_text_update: Callable[[str], None] | None,
        on_thinking_update: Callable[[str], None] | None,
    ) -> RuntimeResponse:
        final_text = final_result.result.output
        final_response = final_result.result.response
        final_thinking = final_response.thinking or progress.thinking
        self._flush_final_updates(
            final_text=final_text,
            final_thinking=final_thinking,
            progress=progress,
            on_text_update=on_text_update,
            on_thinking_update=on_thinking_update,
        )
        return RuntimeResponse(
            text=final_text,
            all_messages=list(final_result.result.all_messages()),
            usage=TurnUsage.from_run_usage(
                final_result.result.usage(),
                provider=self.model_entry.api_type,
                model_id=self.model_entry.id,
                image_count=len(attachments),
                duration_seconds=perf_counter() - started_at,
                tool_calls_started=tool_state.started,
                tool_calls_finished=tool_state.finished,
                tool_call_errors=tool_state.errors,
            ),
            agent_mode_used=agent_mode_used,
        )

    async def _rerun_without_tools(
        self,
        *,
        prompt_text: str,
        system_prompt: str,
        message_history: list[ModelMessage],
        attachments: list[Path],
        model_settings: dict[str, object],
        on_text_update: Callable[[str], None] | None,
        on_thinking_update: Callable[[str], None] | None,
        on_tool_event: Callable[[RuntimeToolEvent], None] | None,
    ) -> RuntimeResponse:
        return await self.run_stream(
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            message_history=message_history,
            attachments=attachments,
            tools=[],
            model_settings=model_settings,
            tool_call_limit=None,
            allow_tool_fallback=False,
            on_text_update=on_text_update,
            on_thinking_update=on_thinking_update,
            on_tool_event=on_tool_event,
        )

    async def _maybe_retry_without_tools(
        self,
        error: Exception,
        *,
        prompt_text: str,
        system_prompt: str,
        message_history: list[ModelMessage],
        attachments: list[Path],
        wrapped_tools: list[Tool[None]],
        model_settings: dict[str, object],
        allow_tool_fallback: bool,
        on_text_update: Callable[[str], None] | None,
        on_thinking_update: Callable[[str], None] | None,
        on_tool_event: Callable[[RuntimeToolEvent], None] | None,
    ) -> RuntimeResponse | None:
        if not wrapped_tools or not allow_tool_fallback:
            return None
        if not self._is_unsupported_tools_error(error):
            return None
        return await self._rerun_without_tools(
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            message_history=message_history,
            attachments=attachments,
            model_settings=model_settings,
            on_text_update=on_text_update,
            on_thinking_update=on_thinking_update,
            on_tool_event=on_tool_event,
        )

    async def _maybe_retry_text_only_after_tool_limit(
        self,
        error: Exception,
        *,
        partial_messages: list[ModelMessage] | None,
        system_prompt: str,
        wrapped_tools: list[Tool[None]],
        model_settings: dict[str, object],
        tool_call_limit: int | None,
        on_text_update: Callable[[str], None] | None,
        on_thinking_update: Callable[[str], None] | None,
        on_tool_event: Callable[[RuntimeToolEvent], None] | None,
    ) -> RuntimeResponse | None:
        if not self._should_retry_text_only_after_tool_limit(
            error,
            partial_messages=partial_messages,
            wrapped_tools=wrapped_tools,
            tool_call_limit=tool_call_limit,
        ):
            return None

        assert partial_messages is not None
        recovery_response = await self._rerun_without_tools(
            prompt_text=_TEXT_ONLY_TOOL_LIMIT_RECOVERY_PROMPT,
            system_prompt=system_prompt,
            message_history=partial_messages,
            attachments=[],
            model_settings=model_settings,
            on_text_update=on_text_update,
            on_thinking_update=on_thinking_update,
            on_tool_event=on_tool_event,
        )
        return RuntimeResponse(
            text=recovery_response.text,
            all_messages=recovery_response.all_messages,
            usage=recovery_response.usage,
            agent_mode_used=recovery_response.agent_mode_used,
            tool_limit_recovery_used=True,
        )

    async def run_stream(
        self,
        *,
        prompt_text: str,
        system_prompt: str,
        message_history: list[ModelMessage],
        attachments: list[Path],
        tools: list[Tool[None]],
        model_settings: dict[str, object],
        tool_call_limit: int | None = None,
        allow_tool_fallback: bool = True,
        on_text_update: Callable[[str], None] | None = None,
        on_thinking_update: Callable[[str], None] | None = None,
        on_tool_event: Callable[[RuntimeToolEvent], None] | None = None,
    ) -> RuntimeResponse:
        tool_state = _ToolState()
        wrapped_tools = [
            self._wrapped_tool(tool, on_tool_event=on_tool_event, tool_state=tool_state)
            for tool in tools
        ]
        user_prompt = self._build_user_prompt(prompt_text, attachments)
        agent = Agent(
            create_pydantic_model(self.model_entry),
            tools=wrapped_tools,
            instructions=system_prompt,
        )
        usage_limits = UsageLimits(tool_calls_limit=tool_call_limit) if wrapped_tools else None
        effective_model_settings = self._effective_model_settings(
            model_settings=model_settings,
            wrapped_tools=wrapped_tools,
        )
        started_at = perf_counter()
        captured_messages_ref = _CapturedRunMessages()

        try:
            final_result, progress = await self._collect_stream_result(
                agent=agent,
                user_prompt=user_prompt,
                message_history=message_history,
                effective_model_settings=effective_model_settings,
                usage_limits=usage_limits,
                captured_messages_ref=captured_messages_ref,
                on_text_update=on_text_update,
                on_thinking_update=on_thinking_update,
            )
            return self._build_runtime_response(
                final_result=final_result,
                progress=progress,
                attachments=attachments,
                started_at=started_at,
                tool_state=tool_state,
                agent_mode_used=bool(wrapped_tools),
                on_text_update=on_text_update,
                on_thinking_update=on_thinking_update,
            )
        except asyncio.CancelledError as exc:
            raise UserInterruptedError() from exc
        except Exception as exc:
            if isinstance(exc, UserInterruptedError):
                raise

            fallback_response = await self._maybe_retry_without_tools(
                exc,
                prompt_text=prompt_text,
                system_prompt=system_prompt,
                message_history=message_history,
                attachments=attachments,
                wrapped_tools=wrapped_tools,
                model_settings=model_settings,
                allow_tool_fallback=allow_tool_fallback,
                on_text_update=on_text_update,
                on_thinking_update=on_thinking_update,
                on_tool_event=on_tool_event,
            )
            if fallback_response is not None:
                return fallback_response

            partial_messages = self._preserve_partial_messages(
                captured_messages_ref.messages,
                exc,
            )
            recovery_response = await self._maybe_retry_text_only_after_tool_limit(
                exc,
                partial_messages=partial_messages,
                system_prompt=system_prompt,
                wrapped_tools=wrapped_tools,
                model_settings=model_settings,
                tool_call_limit=tool_call_limit,
                on_text_update=on_text_update,
                on_thinking_update=on_thinking_update,
                on_tool_event=on_tool_event,
            )
            if recovery_response is not None:
                return recovery_response

            if partial_messages is not None:
                raise RuntimePartialRunError(exc, partial_messages) from exc
            raise
