"""Runtime tool-event bookkeeping helpers for MotherApp."""

from __future__ import annotations

from collections.abc import Callable

from mother.runtime import RuntimeToolEvent
from mother.session import SessionManager


def _record_tool_started(
    session_manager: SessionManager | None,
    event: RuntimeToolEvent,
) -> None:
    """Persist a started tool call when session recording is enabled."""
    if session_manager is None:
        return
    session_manager.record_tool_call(
        tool_name=event.tool_name,
        tool_call_id=event.tool_call_id,
        arguments=event.arguments,
    )


def _record_tool_finished(
    session_manager: SessionManager | None,
    event: RuntimeToolEvent,
) -> None:
    """Persist a finished tool result when session recording is enabled."""
    if session_manager is None:
        return
    session_manager.record_tool_result(
        tool_name=event.tool_name,
        tool_call_id=event.tool_call_id,
        arguments=event.arguments,
        output=event.output or "",
        is_error=event.is_error,
    )


def handle_runtime_tool_event(
    *,
    event: RuntimeToolEvent,
    session_manager: SessionManager | None,
    call_from_thread: Callable[..., object],
    show_tool_started: Callable[[str, str | None, dict[str, object]], None],
    show_tool_finished: Callable[[str, str | None, dict[str, object], str], None],
) -> None:
    """Mirror one runtime tool event into session persistence and chat presentation."""
    if event.phase == "started":
        _record_tool_started(session_manager, event)
        _ = call_from_thread(
            show_tool_started,
            event.tool_name,
            event.tool_call_id,
            event.arguments,
        )
        return

    _record_tool_finished(session_manager, event)
    _ = call_from_thread(
        show_tool_finished,
        event.tool_name,
        event.tool_call_id,
        event.arguments,
        event.output or "",
    )
