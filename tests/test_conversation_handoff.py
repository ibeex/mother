"""Tests for portable cross-model conversation handoff."""

from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)

from mother.conversation_handoff import portable_history


def _request_text(message: ModelRequest) -> str:
    return "\n".join(
        part.content
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )


def _response_text(message: ModelResponse) -> str:
    return "\n".join(part.content for part in message.parts if isinstance(part, TextPart))


def test_portable_history_preserves_user_and_assistant_text() -> None:
    history = [
        ModelRequest(parts=[UserPromptPart("hello")]),
        ModelResponse(parts=[TextPart(content="hi")]),
    ]

    converted = portable_history(history)

    assert len(converted) == 2
    assert isinstance(converted[0], ModelRequest)
    assert isinstance(converted[1], ModelResponse)
    assert _request_text(converted[0]) == "hello"
    assert _response_text(converted[1]) == "hi"


def test_portable_history_drops_empty_provider_specific_response() -> None:
    history = [ModelResponse(parts=[ThinkingPart(content="hidden reasoning")])]

    assert portable_history(history) == []


def test_portable_history_handles_sequence_user_content_with_placeholder() -> None:
    history = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    [
                        TextContent(content="look at this"),
                        BinaryContent(data=b"fake", media_type="image/png"),
                    ]
                )
            ]
        )
    ]

    converted = portable_history(history)

    assert len(converted) == 1
    assert isinstance(converted[0], ModelRequest)
    text = _request_text(converted[0])
    assert "look at this" in text
    assert "[image omitted from model handoff]" in text


def test_portable_history_does_not_mutate_input() -> None:
    request = ModelRequest(parts=[UserPromptPart("hello")])
    response = ModelResponse(
        parts=[TextPart(content="hi", provider_details={"signature": "unsafe"})],
        provider_response_id="response-id",
        provider_name="test-provider",
    )
    history = [request, response]
    original_parts = (request.parts, response.parts)

    converted = portable_history(history)

    assert history == [request, response]
    assert (request.parts, response.parts) == original_parts
    assert converted is not history
    assert converted[0] is not request
    assert converted[1] is not response
