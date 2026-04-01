"""Subtitle and status-line helpers for MotherApp."""

from __future__ import annotations

from dataclasses import dataclass

from mother.agent_modes import AgentProfile
from mother.app_session import AppSession
from mother.widgets import StatusLine


@dataclass(frozen=True, slots=True)
class StatusLineState:
    """Normalized status-line values derived from app session state."""

    model_name: str
    agent_mode: bool
    context_tokens: int | None
    auto_scroll_enabled: bool
    reasoning_effort: str | None
    last_response_time_seconds: float | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    agent_label: str | None


def subtitle_text(
    *,
    model_name: str,
    agent_mode: bool,
    agent_profile: AgentProfile,
) -> str:
    """Build the app subtitle for the current model and runtime mode."""
    if not agent_mode:
        return model_name
    if agent_profile == "deep_research":
        return f"{model_name} [RESEARCH]"
    return f"{model_name} [AGENT]"


def build_status_line_state(
    session: AppSession,
    *,
    auto_scroll_enabled: bool,
) -> StatusLineState:
    """Collect the visible status-line state for the current app session."""
    return StatusLineState(
        model_name=session.config.model,
        agent_mode=session.agent_mode,
        context_tokens=session.last_context_tokens,
        auto_scroll_enabled=auto_scroll_enabled,
        reasoning_effort=session.status_reasoning_effort(),
        last_response_time_seconds=session.last_response_time_seconds,
        input_tokens=session.session_input_tokens,
        output_tokens=session.session_output_tokens,
        cached_tokens=session.session_cached_tokens,
        agent_label=session.status_agent_label(),
    )


def update_status_line(status_line: StatusLine, state: StatusLineState) -> None:
    """Apply a normalized state snapshot to the mounted status-line widget."""
    status_line.set_status(
        model_name=state.model_name,
        agent_mode=state.agent_mode,
        context_tokens=state.context_tokens,
        auto_scroll_enabled=state.auto_scroll_enabled,
        reasoning_effort=state.reasoning_effort,
        last_response_time_seconds=state.last_response_time_seconds,
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        cached_tokens=state.cached_tokens,
        agent_label=state.agent_label,
    )
