"""Tests for the pydantic-ai runtime adapter."""

import asyncio
from typing import cast, final
from unittest.mock import patch

from pydantic_ai import AgentRunResultEvent, Tool
from pydantic_ai._agent_graph import GraphAgentState
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
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
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import RunUsage

from mother.models import ModelEntry
from mother.runtime import ChatRuntime


@final
class _FakeAgent:
    events: tuple[object, ...] = ()
    event_batches: list[tuple[object, ...]] = []
    init_calls: list[dict[str, object]] = []
    run_calls: list[dict[str, object]] = []

    def __init__(self, model: object, tools: list[object], instructions: str) -> None:
        type(self).init_calls.append({"model": model, "tools": tools, "instructions": instructions})
        self._events = type(self).event_batches.pop(0) if type(self).event_batches else type(self).events

    async def run_stream_events(
        self,
        user_prompt: str | list[object],
        *,
        message_history: list[object],
        model_settings: dict[str, object],
        usage_limits: object,
    ):
        type(self).run_calls.append(
            {
                "user_prompt": user_prompt,
                "message_history": list(message_history),
                "model_settings": model_settings,
                "usage_limits": usage_limits,
            }
        )
        for event in self._events:
            if isinstance(event, Exception):
                raise event
            yield event


def test_chat_runtime_run_stream_events_streams_thinking_before_text() -> None:
    entry = ModelEntry(
        id="local_3",
        name="local_3",
        api_type="openai-chat",
        supports_reasoning=True,
    )
    response = ModelResponse(
        parts=[
            ThinkingPart(content="step 1\nstep 2"),
            TextPart(content="final answer"),
        ]
    )
    usage = RunUsage(input_tokens=123, output_tokens=45, cache_read_tokens=6)
    result = AgentRunResult(
        "final answer",
        _state=GraphAgentState(
            message_history=[cast(ModelMessage, response)],
            usage=usage,
        ),
    )
    _FakeAgent.events = (
        PartStartEvent(index=0, part=ThinkingPart(content="step 1\n")),
        PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="step 2")),
        PartStartEvent(index=1, part=TextPart(content="final ")),
        PartDeltaEvent(index=1, delta=TextPartDelta(content_delta="answer")),
        AgentRunResultEvent(result=result),
    )
    _FakeAgent.event_batches = []
    _FakeAgent.init_calls = []
    _FakeAgent.run_calls = []
    callback_events: list[tuple[str, str]] = []

    runtime = ChatRuntime(entry)

    with (
        patch("mother.runtime.Agent", _FakeAgent),
        patch("mother.runtime.create_pydantic_model", return_value=object()),
    ):
        runtime_response = asyncio.run(
            runtime.run_stream(
                prompt_text="hello",
                system_prompt="system",
                message_history=[],
                attachments=[],
                tools=[],
                model_settings={"openai_reasoning_effort": "high"},
                on_text_update=lambda text: callback_events.append(("text", text)),
                on_thinking_update=lambda text: callback_events.append(("thinking", text)),
            )
        )

    assert callback_events == [
        ("thinking", "step 1\n"),
        ("thinking", "step 1\nstep 2"),
        ("text", "final "),
        ("text", "final answer"),
    ]
    assert runtime_response.text == "final answer"
    assert runtime_response.all_messages == [cast(ModelMessage, response)]
    assert runtime_response.tool_limit_recovery_used is False
    assert runtime_response.usage.request_tokens == 123
    assert runtime_response.usage.response_tokens == 45
    assert runtime_response.usage.cache_read_tokens == 6
    assert _FakeAgent.init_calls == [
        {"model": _FakeAgent.init_calls[0]["model"], "tools": [], "instructions": "system"}
    ]
    assert _FakeAgent.run_calls == [
        {
            "user_prompt": _FakeAgent.run_calls[0]["user_prompt"],
            "message_history": [],
            "model_settings": {"openai_reasoning_effort": "high"},
            "usage_limits": None,
        }
    ]


def test_preserve_partial_messages_keeps_completed_tool_results() -> None:
    request = ModelRequest(parts=[UserPromptPart("read the readme")])
    first_response = ModelResponse(
        parts=[ToolCallPart(tool_name="bash", args={"command": "ls"}, tool_call_id="call-1")]
    )
    tool_result_request = ModelRequest(
        parts=[ToolReturnPart(tool_name="bash", content="README.md", tool_call_id="call-1")]
    )
    blocked_response = ModelResponse(
        parts=[ToolCallPart(tool_name="read", args={"path": "README.md"}, tool_call_id="call-2")]
    )

    preserved = ChatRuntime._preserve_partial_messages(  # pyright: ignore[reportPrivateUsage]
        [request, first_response, tool_result_request, blocked_response],
        UsageLimitExceeded("tool limit reached"),
    )

    assert preserved == [request, first_response, tool_result_request]


def test_preserve_partial_messages_skips_unresolved_tool_batches() -> None:
    request = ModelRequest(parts=[UserPromptPart("read the readme")])
    blocked_response = ModelResponse(
        parts=[
            ToolCallPart(tool_name="bash", args={"command": "ls"}, tool_call_id="call-1"),
            ToolCallPart(tool_name="read", args={"path": "README.md"}, tool_call_id="call-2"),
        ]
    )

    preserved = ChatRuntime._preserve_partial_messages(  # pyright: ignore[reportPrivateUsage]
        [request, blocked_response],
        UsageLimitExceeded("tool limit reached"),
    )

    assert preserved is None


def test_preserve_partial_messages_keeps_retry_prompt_requests() -> None:
    request = ModelRequest(parts=[UserPromptPart("open config")])
    retry_request = ModelRequest(
        parts=[RetryPromptPart(content="bad args", tool_name="read", tool_call_id="call-1")]
    )

    preserved = ChatRuntime._preserve_partial_messages(  # pyright: ignore[reportPrivateUsage]
        [request, retry_request],
        UsageLimitExceeded("tool limit reached"),
    )

    assert preserved == [request, retry_request]


def test_run_stream_recovers_with_text_only_retry_after_tool_limit() -> None:
    entry = ModelEntry(
        id="local_3",
        name="local_3",
        api_type="openai-chat",
        supports_reasoning=True,
    )
    preserved_messages = [
        ModelRequest(parts=[UserPromptPart("tell me about current project")]),
        ModelResponse(
            parts=[ToolCallPart(tool_name="bash", args={"command": "ls -la"}, tool_call_id="call-1")]
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="bash", content="README.md\nsrc\ntests", tool_call_id="call-1")]
        ),
    ]
    final_text = "I found README.md, src, and tests. Would you like me to inspect README next?"
    final_response = ModelResponse(parts=[TextPart(content=final_text)])
    usage = RunUsage(input_tokens=12, output_tokens=9)
    result = AgentRunResult(
        final_text,
        _state=GraphAgentState(
            message_history=[*preserved_messages, cast(ModelMessage, final_response)],
            usage=usage,
        ),
    )

    async def sample_tool() -> str:
        return "ok"

    _FakeAgent.events = ()
    _FakeAgent.event_batches = [
        (UsageLimitExceeded("tool limit reached"),),
        (AgentRunResultEvent(result=result),),
    ]
    _FakeAgent.init_calls = []
    _FakeAgent.run_calls = []

    runtime = ChatRuntime(entry)

    with (
        patch("mother.runtime.Agent", _FakeAgent),
        patch("mother.runtime.create_pydantic_model", return_value=object()),
        patch.object(ChatRuntime, "_preserve_partial_messages", return_value=preserved_messages),
    ):
        runtime_response = asyncio.run(
            runtime.run_stream(
                prompt_text="hi, tell me about current project",
                system_prompt="system",
                message_history=[],
                attachments=[],
                tools=[Tool(sample_tool, name="bash")],
                model_settings={},
                tool_call_limit=1,
                allow_tool_fallback=False,
            )
        )

    assert runtime_response.text == final_text
    assert runtime_response.tool_limit_recovery_used is True
    assert len(_FakeAgent.init_calls) == 2
    assert len(cast(list[object], _FakeAgent.init_calls[0]["tools"])) == 1
    assert len(cast(list[object], _FakeAgent.init_calls[1]["tools"])) == 0
    assert _FakeAgent.run_calls[1]["message_history"] == preserved_messages
    assert _FakeAgent.run_calls[1]["user_prompt"] == (
        "Reply to the user in plain text only using the completed tool result. Do not call tools."
    )


def test_tool_arguments_omit_function_defaults() -> None:
    def sample_tool(query: str, timeout: float = 30.0, mode: str = "auto") -> str:
        _ = timeout, mode
        return query

    tool = Tool(sample_tool, name="sample")

    only_required = ChatRuntime.tool_arguments(tool, ("docs",), {})
    assert only_required == {"query": "docs"}

    custom_timeout = ChatRuntime.tool_arguments(tool, ("docs",), {"timeout": 12.0})
    assert custom_timeout == {"query": "docs", "timeout": 12.0}

    explicit_default = ChatRuntime.tool_arguments(tool, ("docs",), {"timeout": 30.0})
    assert explicit_default == {"query": "docs"}
