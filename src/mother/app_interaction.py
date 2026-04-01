"""Prompt-interaction and interruption helpers for MotherApp."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InterruptEscapeDecision:
    """Describe how Mother should react to one Escape key press."""

    handled: bool
    should_notify: bool = False
    should_interrupt: bool = False
    next_escape_at: float | None = None


def decide_interrupt_escape(
    *,
    has_interruptible_work: bool,
    now: float,
    previous_escape_at: float | None,
    double_escape_window_seconds: float,
) -> InterruptEscapeDecision:
    """Return the double-Escape interruption decision for the current key press."""
    if not has_interruptible_work:
        return InterruptEscapeDecision(handled=False)

    if previous_escape_at is None or (now - previous_escape_at) > double_escape_window_seconds:
        return InterruptEscapeDecision(
            handled=True,
            should_notify=True,
            next_escape_at=now,
        )

    return InterruptEscapeDecision(
        handled=True,
        should_interrupt=True,
        next_escape_at=None,
    )
