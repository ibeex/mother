"""Helpers for user-requested interruption of active work."""

from __future__ import annotations

from typing import final


@final
class UserInterruptedError(Exception):
    """Raised when the user interrupts the active request."""

    def __init__(self, message: str = "Interrupted by user.", *, partial_output: str = "") -> None:
        self.message: str = message
        self.partial_output: str = partial_output
        rendered = message
        if partial_output.strip():
            rendered = f"{message}\n\n{partial_output}"
        super().__init__(rendered)
