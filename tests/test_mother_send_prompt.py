"""Tests for the pydantic-ai request flow in MotherApp."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import cast, final
from unittest.mock import patch

from pydantic_ai import Tool
from pydantic_ai.messages import ModelMessage

from mother import MotherApp, MotherConfig
from mother.models import ModelEntry
from mother.runtime import RuntimeResponse, RuntimeToolEvent
from mother.stats import TurnUsage
from mother.widgets import Response, ThinkingOutput


@final
class _FakeResponse:
    def __init__(self) -> None:
        self.updated_texts: list[str] = []
        self.reset_texts: list[str] = []

    def update(self, text: str) -> None:
        self.updated_texts.append(text)

    def reset_state(self, text: str) -> None:
        self.reset_texts.append(text)


@final
class _FakeThinkingOutput:
    def __init__(self) -> None:
        self.display = False
        self.updated_texts: list[str] = []
        self.started = 0
        self.finished = 0

    def start_streaming(self) -> None:
        self.started += 1

    def finish_streaming(self) -> None:
        self.finished += 1

    def set_text(self, text: str) -> None:
        self.updated_texts.append(text)
        self.display = bool(text.strip())


@final
class _FakeRuntime:
    def __init__(self, response: RuntimeResponse, updates: list[tuple[str, str]]) -> None:
        self.response = response
        self.updates = updates
        self.calls: list[dict[str, object]] = []

    async def run_stream(
        self,
        *,
        prompt_text: str,
        system_prompt: str,
        message_history: list[object],
        attachments: list[Path],
        tools: list[object],
        model_settings: dict[str, object],
        allow_tool_fallback: bool = True,
        on_text_update: Callable[[str], None] | None = None,
        on_thinking_update: Callable[[str], None] | None = None,
        on_tool_event: Callable[[RuntimeToolEvent], None] | None = None,
    ) -> RuntimeResponse:
        self.calls.append(
            {
                "prompt_text": prompt_text,
                "system_prompt": system_prompt,
                "message_history": list(message_history),
                "attachments": attachments,
                "tools": tools,
                "model_settings": model_settings,
                "allow_tool_fallback": allow_tool_fallback,
                "has_tool_callback": on_tool_event is not None,
            }
        )
        latest_text = ""
        latest_thinking = ""
        for text, thinking in self.updates:
            if on_thinking_update is not None and thinking != latest_thinking:
                latest_thinking = thinking
                on_thinking_update(thinking)
            if on_text_update is not None and text != latest_text:
                latest_text = text
                on_text_update(text)
        return self.response


@final
class _FailingRuntime:
    async def run_stream(self, **_: object) -> RuntimeResponse:
        raise RuntimeError("boom")


def _call_from_thread(callback: object, *args: object) -> object:
    return cast(Callable[..., object], callback)(*args)


def _make_app(
    *,
    reasoning: bool = False,
    agent_mode: bool = False,
    openai_reasoning_summary: str = "auto",
) -> MotherApp:
    app = MotherApp(
        config=MotherConfig(
            model="test-model",
            reasoning_effort="high",
            openai_reasoning_summary=openai_reasoning_summary,
        )
    )
    app.current_model_entry = ModelEntry(
        id="test-model",
        name="test-model",
        api_type="openai-responses",
        supports_reasoning=reasoning,
        supports_tools=True,
        supports_images=True,
    )
    app.agent_mode = agent_mode
    return app


def test_run_runtime_request_streams_thinking_and_updates_usage() -> None:
    app = _make_app(reasoning=True)
    app.conversation_state.message_history = [cast(ModelMessage, object())]
    response = _FakeResponse()
    thinking_output = _FakeThinkingOutput()
    fake_runtime = _FakeRuntime(
        RuntimeResponse(
            text="final answer",
            all_messages=[cast(ModelMessage, object())],
            usage=TurnUsage(
                request_tokens=123,
                response_tokens=45,
                cache_read_tokens=6,
                duration_seconds=1.5,
                model_id="test-model",
                provider="openai-responses",
            ),
            agent_mode_used=False,
        ),
        [
            ("", "step 1\n"),
            ("", "step 1\nstep 2"),
            ("final ", "step 1\nstep 2"),
            ("final answer", "step 1\nstep 2"),
        ],
    )

    with (
        patch("mother.mother.ChatRuntime", return_value=fake_runtime),
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
        patch.object(app, "notify"),
        patch.object(app, "_scroll_chat_to_end"),
    ):
        full_text = asyncio.run(
            app._run_runtime_request(  # pyright: ignore[reportPrivateUsage]
                "hello",
                cast(Response, cast(object, response)),
                "system",
                tools=[],
                attachments=[Path("/tmp/pasted.png")],
                thinking_output=cast(ThinkingOutput, cast(object, thinking_output)),
            )
        )

    assert full_text == "final answer"
    assert thinking_output.started == 1
    assert thinking_output.finished == 1
    assert thinking_output.updated_texts[-1] == "step 1\nstep 2"
    assert response.updated_texts == ["final ", "final answer"]
    assert response.reset_texts == ["final answer"]
    assert app._last_context_tokens == 123  # pyright: ignore[reportPrivateUsage]
    assert app._session_input_tokens == 123  # pyright: ignore[reportPrivateUsage]
    assert app._session_output_tokens == 45  # pyright: ignore[reportPrivateUsage]
    assert app._session_cached_tokens == 6  # pyright: ignore[reportPrivateUsage]
    assert app._last_response_time_seconds == 1.5  # pyright: ignore[reportPrivateUsage]
    assert len(app.conversation_state.message_history) == 1
    assert fake_runtime.calls == [
        {
            "prompt_text": "hello",
            "system_prompt": "system",
            "message_history": fake_runtime.calls[0]["message_history"],
            "attachments": [Path("/tmp/pasted.png")],
            "tools": [],
            "model_settings": {"openai_reasoning_effort": "high"},
            "allow_tool_fallback": True,
            "has_tool_callback": True,
        }
    ]
    message_history = fake_runtime.calls[0]["message_history"]
    assert isinstance(message_history, list)
    history_list = cast(list[object], message_history)
    assert len(history_list) == 1


def test_reasoning_options_include_openai_reasoning_summary() -> None:
    app = _make_app(reasoning=True, openai_reasoning_summary="detailed")

    assert app._reasoning_options() == {  # pyright: ignore[reportPrivateUsage]
        "openai_reasoning_effort": "high",
        "openai_reasoning_summary": "detailed",
    }


def test_run_runtime_request_disables_agent_mode_after_tool_fallback() -> None:
    app = _make_app(agent_mode=True)
    response = _FakeResponse()
    fake_runtime = _FakeRuntime(
        RuntimeResponse(
            text="fallback response",
            all_messages=[cast(ModelMessage, object())],
            usage=TurnUsage(model_id="test-model", provider="openai-responses"),
            agent_mode_used=False,
        ),
        [("fallback response", "")],
    )

    with (
        patch("mother.mother.ChatRuntime", return_value=fake_runtime),
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
        patch.object(app, "notify") as notify,
    ):
        full_text = asyncio.run(
            app._run_runtime_request(  # pyright: ignore[reportPrivateUsage]
                "hello",
                cast(Response, cast(object, response)),
                "system",
                tools=[Tool(lambda: "ok")],
                attachments=[],
                thinking_output=None,
            )
        )

    assert full_text == "fallback response"
    assert app.agent_mode is False
    notify.assert_called_with(
        "test-model does not support tools — agent mode disabled",
        title="Agent mode",
        severity="warning",
    )


def test_run_runtime_request_shows_error_when_runtime_fails() -> None:
    app = _make_app()
    response = _FakeResponse()

    with (
        patch("mother.mother.ChatRuntime", return_value=_FailingRuntime()),
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
    ):
        full_text = asyncio.run(
            app._run_runtime_request(  # pyright: ignore[reportPrivateUsage]
                "hello",
                cast(Response, cast(object, response)),
                "system",
                tools=[],
                attachments=[],
                thinking_output=None,
            )
        )

    assert full_text is None
    assert response.updated_texts == ["**Error:** boom"]
    assert response.reset_texts == ["**Error:** boom"]
