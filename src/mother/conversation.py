"""Conversation state management for Mother."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai.messages import ModelMessage


@dataclass(slots=True)
class ConversationState:
    """Track model-visible message history for the active chat."""

    message_history: list[ModelMessage] = field(default_factory=list)

    def clear(self) -> None:
        self.message_history.clear()

    @property
    def has_history(self) -> bool:
        return len(self.message_history) > 0
