"""Session-save workflow helpers for MotherApp."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mother.session import MarkdownFormatNotice, SessionManager, format_markdown_export


@dataclass(frozen=True, slots=True)
class SessionSaveNotification:
    """Describe one user-visible result from attempting a session save."""

    message: str
    severity: Literal["warning", "error"] | None = None


@dataclass(frozen=True, slots=True)
class SessionSaveResult:
    """Describe the complete outcome of a session-save attempt."""

    notifications: tuple[SessionSaveNotification, ...]
    output_path: Path | None = None


def save_session_markdown(
    session_manager: SessionManager | None,
    *,
    format_export: Callable[[Path], MarkdownFormatNotice | None] = format_markdown_export,
) -> SessionSaveResult:
    """Save the current session to markdown and collect user-visible notices."""
    if session_manager is None:
        return SessionSaveResult(
            notifications=(
                SessionSaveNotification(
                    "Session saving is unavailable.",
                    severity="warning",
                ),
            )
        )

    try:
        output_path = session_manager.save_as_markdown()
    except RuntimeError as exc:
        return SessionSaveResult(
            notifications=(SessionSaveNotification(str(exc), severity="warning"),)
        )
    except Exception as exc:
        return SessionSaveResult(
            notifications=(
                SessionSaveNotification(
                    f"Failed to save session: {exc}",
                    severity="error",
                ),
            )
        )

    notifications: list[SessionSaveNotification] = [
        SessionSaveNotification(f"Saved to {output_path}")
    ]
    format_notice = format_export(output_path)
    if format_notice is not None:
        notifications.append(
            SessionSaveNotification(
                format_notice.message,
                severity=format_notice.severity,
            )
        )

    return SessionSaveResult(
        notifications=tuple(notifications),
        output_path=output_path,
    )
