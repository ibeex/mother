"""Tests for the refactored LLM request flow in MotherApp."""

from collections.abc import Callable, Iterable, Iterator
from typing import cast, final
from unittest.mock import patch

from llm.models import Conversation, ToolDef

from mother import MotherApp, MotherConfig
from mother.widgets import Response


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
class _UnsupportedToolsConversation:
    def __init__(self) -> None:
        self.chain_calls: int = 0
        self.prompt_calls: int = 0
        self.chain_limit: int | None = None
        self.prompt_systems: list[str | None] = []

    def chain(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        chain_limit: int | None = None,
        before_call: object | None = None,
        after_call: object | None = None,
    ) -> Iterable[str]:
        _ = prompt
        _ = system
        _ = tools
        _ = before_call
        _ = after_call
        self.chain_calls += 1
        self.chain_limit = chain_limit
        raise RuntimeError("test-model does not support tools")

    def prompt(self, prompt: str, *, system: str | None = None) -> Iterable[str]:
        _ = prompt
        self.prompt_systems.append(system)
        self.prompt_calls += 1
        return ["fallback", " response"]


@final
class _FailingConversation:
    def chain(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        chain_limit: int | None = None,
        before_call: object | None = None,
        after_call: object | None = None,
    ) -> Iterable[str]:
        _ = prompt
        _ = system
        _ = tools
        _ = chain_limit
        _ = before_call
        _ = after_call
        raise RuntimeError("boom")

    def prompt(self, prompt: str, *, system: str | None = None) -> Iterable[str]:
        _ = prompt
        _ = system
        raise AssertionError("prompt should not be called")


def _call_from_thread(callback: object, *args: object) -> object:
    return cast(Callable[..., object], callback)(*args)


def _broken_stream() -> Iterator[str]:
    yield "partial"
    raise RuntimeError("stream failed")


def test_request_llm_response_falls_back_when_model_rejects_tools():
    app = MotherApp(config=MotherConfig(model="test-model", tools_enabled=True))
    app.agent_mode = True
    response = _FakeResponse()
    conversation = _UnsupportedToolsConversation()
    tools = cast(list[ToolDef], [object()])

    with (
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
        patch.object(app, "notify"),
    ):
        llm_response = app._request_llm_response(  # pyright: ignore[reportPrivateUsage]
            conversation=cast(Conversation, cast(object, conversation)),
            prompt="hello",
            system="system",
            tools=tools,
            response=cast(Response, cast(object, response)),
        )

    assert llm_response is not None
    assert list(llm_response) == ["fallback", " response"]
    assert app.agent_mode is False
    assert conversation.chain_calls == 1
    assert conversation.chain_limit == 3
    assert conversation.prompt_calls == 1
    assert conversation.prompt_systems == [app._build_system_prompt(None, agent_mode=False)]  # pyright: ignore[reportPrivateUsage]
    assert response.updated_texts == []
    assert response.reset_texts == []


def test_request_llm_response_shows_error_for_non_tool_failures():
    app = MotherApp(config=MotherConfig(model="test-model", tools_enabled=True))
    app.agent_mode = True
    response = _FakeResponse()
    tools = cast(list[ToolDef], [object()])

    with patch.object(app, "call_from_thread", side_effect=_call_from_thread):
        llm_response = app._request_llm_response(  # pyright: ignore[reportPrivateUsage]
            conversation=cast(Conversation, cast(object, _FailingConversation())),
            prompt="hello",
            system="system",
            tools=tools,
            response=cast(Response, cast(object, response)),
        )

    assert llm_response is None
    assert app.agent_mode is True
    assert response.updated_texts == ["**Error:** boom"]
    assert response.reset_texts == ["**Error:** boom"]


def test_stream_llm_response_refreshes_context_size_on_success():
    app = MotherApp()
    response = _FakeResponse()

    with (
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
        patch.object(app, "_refresh_context_size") as refresh_context_size,
    ):
        success = app._stream_llm_response(  # pyright: ignore[reportPrivateUsage]
            ["hello", " world"],
            cast(Response, cast(object, response)),
        )

    assert success is True
    assert response.updated_texts == ["hello", "hello world"]
    assert response.reset_texts == ["hello world"]
    refresh_context_size.assert_called_once_with()


def test_stream_llm_response_shows_error_when_streaming_fails():
    app = MotherApp()
    response = _FakeResponse()

    with patch.object(app, "call_from_thread", side_effect=_call_from_thread):
        success = app._stream_llm_response(  # pyright: ignore[reportPrivateUsage]
            _broken_stream(),
            cast(Response, cast(object, response)),
        )

    assert success is False
    assert response.updated_texts == ["partial", "**Error:** stream failed"]
    assert response.reset_texts == ["**Error:** stream failed"]
