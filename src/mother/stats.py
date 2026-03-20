"""Usage normalization and session statistics for Mother."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pydantic_ai.usage import RunUsage


def _add_optional(total: int | None, value: int | None) -> int | None:
    if value is None:
        return total
    if total is None:
        return value
    return total + value


@dataclass(frozen=True, slots=True)
class TurnUsage:
    request_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    tool_calls_started: int = 0
    tool_calls_finished: int = 0
    tool_call_errors: int = 0
    image_count: int = 0
    duration_seconds: float | None = None
    provider: str = ""
    model_id: str = ""

    def to_event_details(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_run_usage(
        cls,
        usage: RunUsage | None,
        *,
        provider: str,
        model_id: str,
        image_count: int,
        duration_seconds: float | None,
        tool_calls_started: int,
        tool_calls_finished: int,
        tool_call_errors: int,
    ) -> TurnUsage:
        if usage is None or not usage.has_values():
            return cls(
                tool_calls_started=tool_calls_started,
                tool_calls_finished=tool_calls_finished,
                tool_call_errors=tool_call_errors,
                image_count=image_count,
                duration_seconds=duration_seconds,
                provider=provider,
                model_id=model_id,
            )

        total_tokens = usage.total_tokens
        return cls(
            request_tokens=usage.input_tokens,
            response_tokens=usage.output_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            tool_calls_started=tool_calls_started,
            tool_calls_finished=tool_calls_finished,
            tool_call_errors=tool_call_errors,
            image_count=image_count,
            duration_seconds=duration_seconds,
            provider=provider,
            model_id=model_id,
        )


@dataclass(slots=True)
class SessionUsage:
    request_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    tool_calls_started: int = 0
    tool_calls_finished: int = 0
    tool_call_errors: int = 0
    image_count: int = 0
    last_context_tokens: int | None = None
    last_response_time_seconds: float | None = None

    def add_turn(self, usage: TurnUsage) -> None:
        self.request_tokens = _add_optional(self.request_tokens, usage.request_tokens)
        self.response_tokens = _add_optional(self.response_tokens, usage.response_tokens)
        self.total_tokens = _add_optional(self.total_tokens, usage.total_tokens)
        self.cache_read_tokens = _add_optional(self.cache_read_tokens, usage.cache_read_tokens)
        self.cache_write_tokens = _add_optional(self.cache_write_tokens, usage.cache_write_tokens)
        self.tool_calls_started += usage.tool_calls_started
        self.tool_calls_finished += usage.tool_calls_finished
        self.tool_call_errors += usage.tool_call_errors
        self.image_count += usage.image_count
        self.last_context_tokens = usage.request_tokens
        self.last_response_time_seconds = usage.duration_seconds
