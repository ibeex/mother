"""Tests for the refactored LLM request flow in MotherApp."""

from collections.abc import Callable, Iterable, Iterator
from enum import StrEnum
from typing import ClassVar, cast, final
from unittest.mock import patch

from llm.models import Attachment, Conversation, Model, ToolDef

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
class _FakeChatView:
    def __init__(self, *, scroll_y: float, max_scroll_y: float) -> None:
        self.scroll_y = scroll_y
        self.max_scroll_y = max_scroll_y
        self.scroll_end_calls: int = 0

    def scroll_end(self, *, animate: bool = False) -> None:
        _ = animate
        self.scroll_end_calls += 1


@final
class _UnsupportedToolsConversation:
    def __init__(self) -> None:
        self.chain_calls: int = 0
        self.prompt_calls: int = 0
        self.chain_limit: int | None = None
        self.prompt_systems: list[str | None] = []
        self.chain_attachments: list[list[Attachment] | None] = []
        self.prompt_attachments: list[list[Attachment] | None] = []

    def chain(
        self,
        prompt: str,
        *,
        system: str | None = None,
        attachments: list[Attachment] | None = None,
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
        self.chain_attachments.append(attachments)
        raise RuntimeError("test-model does not support tools")

    def prompt(
        self,
        prompt: str,
        *,
        system: str | None = None,
        attachments: list[Attachment] | None = None,
    ) -> Iterable[str]:
        _ = prompt
        self.prompt_systems.append(system)
        self.prompt_calls += 1
        self.prompt_attachments.append(attachments)
        return ["fallback", " response"]


@final
class _FailingConversation:
    def chain(
        self,
        prompt: str,
        *,
        system: str | None = None,
        attachments: list[Attachment] | None = None,
        tools: list[ToolDef] | None = None,
        chain_limit: int | None = None,
        before_call: object | None = None,
        after_call: object | None = None,
    ) -> Iterable[str]:
        _ = prompt
        _ = system
        _ = attachments
        _ = tools
        _ = chain_limit
        _ = before_call
        _ = after_call
        raise RuntimeError("boom")

    def prompt(
        self,
        prompt: str,
        *,
        system: str | None = None,
        attachments: list[Attachment] | None = None,
    ) -> Iterable[str]:
        _ = prompt
        _ = system
        _ = attachments
        raise AssertionError("prompt should not be called")


@final
class _PromptOnlyConversation:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, list[Attachment] | None]] = []

    def prompt(
        self,
        prompt: str,
        *,
        system: str | None = None,
        attachments: list[Attachment] | None = None,
    ) -> Iterable[str]:
        self.calls.append((prompt, system, attachments))
        return ["ok"]


@final
class _ReasoningPromptConversation:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def prompt(
        self,
        prompt: str,
        *,
        system: str | None = None,
        attachments: list[Attachment] | None = None,
        reasoning_effort: str | None = None,
    ) -> Iterable[str]:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "attachments": attachments,
                "reasoning_effort": reasoning_effort,
            }
        )
        return ["ok"]


@final
class _ReasoningChainConversation:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chain(
        self,
        prompt: str,
        *,
        system: str | None = None,
        attachments: list[Attachment] | None = None,
        tools: list[ToolDef] | None = None,
        chain_limit: int | None = None,
        before_call: object | None = None,
        after_call: object | None = None,
        options: dict[str, object] | None = None,
    ) -> Iterable[str]:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "attachments": attachments,
                "tools": tools,
                "chain_limit": chain_limit,
                "before_call": before_call,
                "after_call": after_call,
                "options": options,
            }
        )
        return ["ok"]


class _ReasoningEnum(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


@final
class _Field:
    def __init__(self, annotation: object) -> None:
        self.annotation = annotation


@final
class _ModelWithReasoningOptions:
    @final
    class Options:
        model_fields: ClassVar[dict[str, object]] = {
            "reasoning_effort": _Field(_ReasoningEnum | None)
        }


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

    attachment = Attachment(path="/tmp/pasted.png")

    with (
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
        patch.object(app, "notify"),
    ):
        llm_response = app._request_llm_response(  # pyright: ignore[reportPrivateUsage]
            conversation=cast(Conversation, cast(object, conversation)),
            prompt="hello",
            system="system",
            tools=tools,
            attachments=[attachment],
            response=cast(Response, cast(object, response)),
        )

    assert llm_response is not None
    assert list(llm_response) == ["fallback", " response"]
    assert app.agent_mode is False
    assert conversation.chain_calls == 1
    assert conversation.chain_limit == 3
    assert conversation.chain_attachments == [[attachment]]
    assert conversation.prompt_calls == 1
    assert conversation.prompt_systems == [app._build_system_prompt(None, agent_mode=False)]  # pyright: ignore[reportPrivateUsage]
    assert conversation.prompt_attachments == [[attachment]]
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
            attachments=[Attachment(path="/tmp/pasted.png")],
            response=cast(Response, cast(object, response)),
        )

    assert llm_response is None
    assert app.agent_mode is True
    assert response.updated_texts == ["**Error:** boom"]
    assert response.reset_texts == ["**Error:** boom"]


def test_request_llm_response_passes_attachments_without_tools():
    app = MotherApp(config=MotherConfig(model="test-model"))
    response = _FakeResponse()
    attachment = Attachment(path="/tmp/pasted.png")
    conversation = _PromptOnlyConversation()

    llm_response = app._request_llm_response(  # pyright: ignore[reportPrivateUsage]
        conversation=cast(Conversation, cast(object, conversation)),
        prompt="hello",
        system="system",
        tools=None,
        attachments=[attachment],
        response=cast(Response, cast(object, response)),
    )

    assert llm_response is not None
    assert list(llm_response) == ["ok"]
    assert conversation.calls == [("hello", "system", [attachment])]


def test_request_llm_response_passes_reasoning_effort_without_tools() -> None:
    app = MotherApp(config=MotherConfig(model="test-model", reasoning_effort="high"))
    app.model = cast(Model, cast(object, _ModelWithReasoningOptions()))
    response = _FakeResponse()
    conversation = _ReasoningPromptConversation()

    llm_response = app._request_llm_response(  # pyright: ignore[reportPrivateUsage]
        conversation=cast(Conversation, cast(object, conversation)),
        prompt="hello",
        system="system",
        tools=None,
        attachments=None,
        response=cast(Response, cast(object, response)),
    )

    assert llm_response is not None
    assert list(llm_response) == ["ok"]
    assert conversation.calls == [
        {
            "prompt": "hello",
            "system": "system",
            "attachments": None,
            "reasoning_effort": "high",
        }
    ]


def test_request_llm_response_passes_reasoning_effort_with_tools() -> None:
    app = MotherApp(config=MotherConfig(model="test-model", reasoning_effort="medium"))
    app.model = cast(Model, cast(object, _ModelWithReasoningOptions()))
    response = _FakeResponse()
    conversation = _ReasoningChainConversation()
    tools = cast(list[ToolDef], [object()])

    llm_response = app._request_llm_response(  # pyright: ignore[reportPrivateUsage]
        conversation=cast(Conversation, cast(object, conversation)),
        prompt="hello",
        system="system",
        tools=tools,
        attachments=None,
        response=cast(Response, cast(object, response)),
    )

    assert llm_response is not None
    assert list(llm_response) == ["ok"]
    assert len(conversation.calls) == 1
    call = conversation.calls[0]
    assert call["prompt"] == "hello"
    assert call["system"] == "system"
    assert call["attachments"] is None
    assert call["tools"] == tools
    assert call["chain_limit"] == 3
    assert call["before_call"] is not None
    assert call["after_call"] is not None
    assert call["options"] == {"reasoning_effort": "medium"}


def test_stream_llm_response_refreshes_context_size_on_success():
    app = MotherApp()
    response = _FakeResponse()

    with (
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
        patch.object(app, "_refresh_context_size") as refresh_context_size,
    ):
        full_text = app._stream_llm_response(  # pyright: ignore[reportPrivateUsage]
            ["hello", " world"],
            cast(Response, cast(object, response)),
        )

    assert full_text == "hello world"
    assert response.updated_texts == ["hello", "hello world"]
    assert response.reset_texts == ["hello world"]
    refresh_context_size.assert_called_once_with()


def test_stream_llm_response_shows_error_when_streaming_fails():
    app = MotherApp()
    response = _FakeResponse()

    with patch.object(app, "call_from_thread", side_effect=_call_from_thread):
        full_text = app._stream_llm_response(  # pyright: ignore[reportPrivateUsage]
            _broken_stream(),
            cast(Response, cast(object, response)),
        )

    assert full_text is None
    assert response.updated_texts == ["partial", "**Error:** stream failed"]
    assert response.reset_texts == ["**Error:** stream failed"]


def test_update_response_output_scrolls_when_near_end():
    app = MotherApp()
    response = _FakeResponse()
    chat_view = _FakeChatView(scroll_y=9, max_scroll_y=10)

    with patch.object(app, "query_one", return_value=chat_view):
        app._update_response_output(  # pyright: ignore[reportPrivateUsage]
            cast(Response, cast(object, response)),
            "hello",
        )

    assert response.updated_texts == ["hello"]
    assert chat_view.scroll_end_calls == 1


def test_update_response_output_does_not_scroll_when_user_is_reading_history():
    app = MotherApp()
    response = _FakeResponse()
    chat_view = _FakeChatView(scroll_y=4, max_scroll_y=10)

    with patch.object(app, "query_one", return_value=chat_view):
        app._update_response_output(  # pyright: ignore[reportPrivateUsage]
            cast(Response, cast(object, response)),
            "hello",
        )

    assert response.updated_texts == ["hello"]
    assert chat_view.scroll_end_calls == 0


def test_update_response_output_respects_manual_scroll_mode():
    app = MotherApp()
    app.auto_scroll_enabled = False
    response = _FakeResponse()
    chat_view = _FakeChatView(scroll_y=10, max_scroll_y=10)

    with patch.object(app, "query_one", return_value=chat_view):
        app._update_response_output(  # pyright: ignore[reportPrivateUsage]
            cast(Response, cast(object, response)),
            "hello",
        )

    assert response.updated_texts == ["hello"]
    assert chat_view.scroll_end_calls == 0
