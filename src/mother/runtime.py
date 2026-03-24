"""pydantic-ai runtime adapter used by Mother's TUI."""

from __future__ import annotations

import asyncio
import mimetypes
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter, isawaitable, signature
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

from pydantic_ai import Agent, AgentRunResultEvent, Tool
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    UserContent,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from mother.interrupts import UserInterruptedError
from mother.models import ModelEntry, create_pydantic_model
from mother.stats import TurnUsage


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
    def _wrapped_tool(
        tool: Tool[None],
        *,
        on_tool_event: Callable[[RuntimeToolEvent], None] | None,
        tool_state: dict[str, int],
    ) -> Tool[None]:
        original = cast(Callable[..., object], tool.function)

        @wraps(original)
        async def wrapped(*args: object, **kwargs: object) -> object:
            arguments = ChatRuntime.tool_arguments(tool, args, kwargs)
            tool_state["sequence"] += 1
            call_id = f"{tool.name}-{tool_state['sequence']}"
            tool_state["started"] += 1
            if on_tool_event is not None:
                _ = on_tool_event(
                    RuntimeToolEvent(
                        phase="started",
                        tool_name=tool.name,
                        tool_call_id=call_id,
                        arguments=arguments,
                    )
                )
            try:
                result = original(*args, **kwargs)
                if isawaitable(result):
                    result = await cast(Awaitable[object], result)
            except Exception as exc:
                tool_state["errors"] += 1
                tool_state["finished"] += 1
                if on_tool_event is not None:
                    _ = on_tool_event(
                        RuntimeToolEvent(
                            phase="finished",
                            tool_name=tool.name,
                            tool_call_id=call_id,
                            arguments=arguments,
                            output=str(exc),
                            is_error=True,
                        )
                    )
                raise

            output = result if isinstance(result, str) else str(result)
            tool_state["finished"] += 1
            if on_tool_event is not None:
                _ = on_tool_event(
                    RuntimeToolEvent(
                        phase="finished",
                        tool_name=tool.name,
                        tool_call_id=call_id,
                        arguments=arguments,
                        output=output,
                    )
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
        tool_state = {"sequence": 0, "started": 0, "finished": 0, "errors": 0}
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
        started_at = perf_counter()

        try:
            full_text = ""
            full_thinking = ""
            final_result: AgentRunResultEvent[str] | None = None

            async for event in agent.run_stream_events(
                user_prompt,
                message_history=message_history,
                model_settings=cast(ModelSettings, cast(object, model_settings)),
                usage_limits=usage_limits,
            ):
                if isinstance(event, AgentRunResultEvent):
                    final_result = event
                    break

                if isinstance(event, PartStartEvent):
                    if isinstance(event.part, ThinkingPart) and event.part.content:
                        full_thinking += event.part.content
                        if on_thinking_update is not None:
                            on_thinking_update(full_thinking)
                    elif isinstance(event.part, TextPart) and event.part.content:
                        full_text += event.part.content
                        if on_text_update is not None:
                            on_text_update(full_text)
                    continue

                if isinstance(event, PartDeltaEvent):
                    if isinstance(event.delta, ThinkingPartDelta) and event.delta.content_delta:
                        full_thinking += event.delta.content_delta
                        if on_thinking_update is not None:
                            on_thinking_update(full_thinking)
                    elif isinstance(event.delta, TextPartDelta) and event.delta.content_delta:
                        full_text += event.delta.content_delta
                        if on_text_update is not None:
                            on_text_update(full_text)

            if final_result is None:
                raise RuntimeError("Agent stream ended without a final result")

            final_text = final_result.result.output
            final_response = final_result.result.response
            final_thinking = final_response.thinking or full_thinking
            if final_thinking != full_thinking and on_thinking_update is not None:
                on_thinking_update(final_thinking)
            if final_text != full_text and on_text_update is not None:
                on_text_update(final_text)

            return RuntimeResponse(
                text=final_text,
                all_messages=list(final_result.result.all_messages()),
                usage=TurnUsage.from_run_usage(
                    final_result.result.usage(),
                    provider=self.model_entry.api_type,
                    model_id=self.model_entry.id,
                    image_count=len(attachments),
                    duration_seconds=perf_counter() - started_at,
                    tool_calls_started=tool_state["started"],
                    tool_calls_finished=tool_state["finished"],
                    tool_call_errors=tool_state["errors"],
                ),
                agent_mode_used=bool(wrapped_tools),
            )
        except asyncio.CancelledError as exc:
            raise UserInterruptedError() from exc
        except Exception as exc:
            if isinstance(exc, UserInterruptedError):
                raise
            if (
                not wrapped_tools
                or not allow_tool_fallback
                or not self._is_unsupported_tools_error(exc)
            ):
                raise
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
