"""Portable conversation history handoff between models/providers."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    UserContent,
    UserPromptPart,
)


def portable_history(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """Return a safe text-only copy of model history for cross-model replay."""
    portable: list[ModelMessage] = []
    for message in messages:
        converted = _message_to_text(message)
        if converted is None:
            continue
        role, text = converted
        if role == "user":
            portable.append(ModelRequest(parts=[UserPromptPart(text)]))
        else:
            portable.append(ModelResponse(parts=[TextPart(content=text)]))
    return portable


def _message_to_text(message: ModelMessage) -> tuple[str, str] | None:
    """Convert a pydantic-ai message to a simple role/text pair when possible."""
    if isinstance(message, ModelRequest):
        text = _request_text(message)
        if text:
            return ("user", text)
        return None
    text = _response_text(message)
    if text:
        return ("assistant", text)
    return None


def _request_text(request: ModelRequest) -> str:
    """Extract user-visible prompt text from a model request."""
    parts: list[str] = []
    for part in request.parts:
        if isinstance(part, UserPromptPart):
            text = _part_text(part)
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _response_text(response: ModelResponse) -> str:
    """Extract only user-visible assistant text from a model response."""
    parts: list[str] = []
    for part in response.parts:
        if isinstance(part, TextPart) and part.content:
            parts.append(part.content)
    return "\n\n".join(parts)


def _part_text(part: object) -> str:
    """Return safe portable text for the supported pydantic-ai message parts."""
    if isinstance(part, UserPromptPart):
        return _user_content_text(part.content)
    if isinstance(part, TextPart):
        return part.content
    return ""


def _user_content_text(content: str | Sequence[UserContent]) -> str:
    """Render user prompt content, replacing non-text attachments with placeholders."""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for item in content:
        text = _user_content_item_text(item)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _user_content_item_text(item: UserContent) -> str:
    """Render a single user-content item safely for model handoff."""
    if isinstance(item, str):
        return item
    if isinstance(item, TextContent):
        return item.content
    if isinstance(item, BinaryContent):
        return _binary_placeholder(item.media_type)

    kind = getattr(item, "kind", "attachment")
    if kind == "cache-point":
        return ""
    return f"[{kind} omitted from model handoff]"


def _binary_placeholder(media_type: str) -> str:
    """Return a compact placeholder for omitted binary content."""
    if media_type.startswith("image/"):
        return "[image omitted from model handoff]"
    if media_type.startswith("audio/"):
        return "[audio omitted from model handoff]"
    if media_type.startswith("video/"):
        return "[video omitted from model handoff]"
    return "[file omitted from model handoff]"
