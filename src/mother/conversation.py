"""Conversation state management for Mother."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

ConversationRole = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class TranscriptMessage:
    """A plain-text conversation message used for context snapshots."""

    role: ConversationRole
    content: str


@dataclass(slots=True)
class ConversationState:
    """Track model-visible history plus a plain-text transcript for snapshots."""

    message_history: list[ModelMessage] = field(default_factory=list)
    transcript_messages: list[TranscriptMessage] = field(default_factory=list)

    def clear(self) -> None:
        self.message_history.clear()
        self.transcript_messages.clear()

    @property
    def has_history(self) -> bool:
        return len(self.message_history) > 0 or len(self.transcript_messages) > 0

    def append_transcript_message(self, role: ConversationRole, content: str) -> None:
        """Append a plain-text transcript message."""
        self.transcript_messages.append(TranscriptMessage(role=role, content=content))

    def append_transcript_turn(self, user_text: str, assistant_text: str) -> None:
        """Append a completed user/assistant pair to the plain-text transcript."""
        self.append_transcript_message("user", user_text)
        self.append_transcript_message("assistant", assistant_text)

    def append_synthetic_turn(self, user_text: str, assistant_text: str) -> None:
        """Append a completed turn directly into model history and transcript state."""
        self.message_history.extend(
            [
                ModelRequest(parts=[UserPromptPart(user_text)]),
                ModelResponse(parts=[TextPart(content=assistant_text)]),
            ]
        )
        self.append_transcript_turn(user_text, assistant_text)

    def formatted_recent_transcript(self, *, max_turns: int, max_chars: int) -> str:
        """Render a bounded plain-text transcript for council-style context snapshots."""
        if max_turns <= 0 or max_chars <= 0 or not self.transcript_messages:
            return ""

        message_limit = max_turns * 2
        selected = list(self.transcript_messages[-message_limit:])
        while selected:
            rendered = self._render_transcript(selected)
            if len(rendered) <= max_chars:
                return rendered
            _ = selected.pop(0)
        return ""

    @staticmethod
    def _render_transcript(messages: list[TranscriptMessage]) -> str:
        return "\n\n".join(
            f"{message.role.capitalize()}: {message.content}" for message in messages
        )
