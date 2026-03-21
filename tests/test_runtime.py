"""Tests for the pydantic-ai runtime adapter."""

import asyncio
from typing import cast, final
from unittest.mock import patch

from pydantic_ai import AgentRunResultEvent
from pydantic_ai._agent_graph import GraphAgentState
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import RunUsage

from mother.models import ModelEntry
from mother.runtime import ChatRuntime


@final
class _FakeAgent:
    events: tuple[object, ...] = ()
    init_calls: list[dict[str, object]] = []
    run_calls: list[dict[str, object]] = []

    def __init__(self, model: object, tools: list[object], instructions: str) -> None:
        type(self).init_calls.append({"model": model, "tools": tools, "instructions": instructions})

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
        for event in type(self).events:
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
