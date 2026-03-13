"""Output capture, truncation, and sanitization for bash tool results."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from typing import IO, ClassVar

_ANSI_ESCAPE = re.compile(
    r"\x1b\[[0-9;]*[mGKHFJA-Z]|\x1b\[[0-9;]*[a-z]|\x1b\].*?\x07|\x1b[@-Z\\-_]"
)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_output(raw: bytes) -> str:
    """Decode bytes, strip ANSI, normalize newlines, remove control chars."""
    text = raw.decode("utf-8", errors="replace")
    text = _ANSI_ESCAPE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    return text


@dataclass
class TruncationResult:
    content: str
    truncated: bool
    original_lines: int
    kept_from: int
    kept_to: int


def truncate_tail(text: str, max_lines: int = 2000, max_bytes: int = 50 * 1024) -> TruncationResult:
    """Tail-truncate text to max_lines or max_bytes, whichever is hit first."""
    lines = text.splitlines(keepends=True)
    original_lines = len(lines)

    # Apply line limit first
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        truncated_by_lines = True
    else:
        truncated_by_lines = False

    # Apply byte limit
    joined = "".join(lines)
    encoded = joined.encode("utf-8")

    if len(encoded) > max_bytes:
        # Tail-truncate to max_bytes, respecting UTF-8 boundaries
        tail_bytes = encoded[-max_bytes:]
        # Find the first valid UTF-8 boundary
        for offset in range(4):
            try:
                content = tail_bytes[offset:].decode("utf-8")
                break
            except UnicodeDecodeError:
                continue
        else:
            content = tail_bytes.decode("utf-8", errors="replace")

        kept_lines = content.splitlines(keepends=True)
        kept_from = original_lines - len(kept_lines) + 1
        kept_to = original_lines
        return TruncationResult(
            content=content,
            truncated=True,
            original_lines=original_lines,
            kept_from=kept_from,
            kept_to=kept_to,
        )

    if truncated_by_lines:
        kept_from = original_lines - len(lines) + 1
        kept_to = original_lines
        return TruncationResult(
            content=joined,
            truncated=True,
            original_lines=original_lines,
            kept_from=kept_from,
            kept_to=kept_to,
        )

    return TruncationResult(
        content=joined,
        truncated=False,
        original_lines=original_lines,
        kept_from=1,
        kept_to=original_lines,
    )


def format_truncation_notice(trunc: TruncationResult, full_output_path: str | None) -> str:
    """Format a user-facing notice about truncated output."""
    path_info = f" Full output: {full_output_path}" if full_output_path else ""
    return (
        f"\n[Showing lines {trunc.kept_from}-{trunc.kept_to} of {trunc.original_lines}.{path_info}]"
    )


@dataclass
class BashResult:
    output: str = ""
    exit_code: int | None = None
    cancelled: bool = False
    truncated: bool = False
    full_output_path: str | None = None


class OutputCapture:
    """Rolling buffer with temp file for large output."""

    MAX_LINES: ClassVar[int] = 2000
    MAX_BYTES: ClassVar[int] = 50 * 1024
    ROLLING_BYTES: ClassVar[int] = 100 * 1024

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._chunks_bytes: int = 0
        self._total_bytes: int = 0
        self._temp_file: IO[str] | None = None
        self._temp_file_path: str | None = None

    def add_bytes(self, raw: bytes) -> None:
        self._total_bytes += len(raw)
        text = sanitize_output(raw)

        # Open temp file once threshold is crossed, write all buffered content + new
        if self._total_bytes > self.MAX_BYTES and self._temp_file is None:
            tf = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="mother_bash_", delete=False, encoding="utf-8"
            )
            self._temp_file = tf
            self._temp_file_path = tf.name
            for chunk in self._chunks:
                _ = self._temp_file.write(chunk)
            self._temp_file.flush()

        if self._temp_file is not None:
            _ = self._temp_file.write(text)
            self._temp_file.flush()

        self._chunks.append(text)
        self._chunks_bytes += len(text.encode("utf-8"))

        # Evict from the front to keep rolling buffer within limit
        while self._chunks_bytes > self.ROLLING_BYTES and len(self._chunks) > 1:
            removed = self._chunks.pop(0)
            self._chunks_bytes -= len(removed.encode("utf-8"))

    def current_tail_preview(self) -> TruncationResult:
        return truncate_tail("".join(self._chunks), self.MAX_LINES, self.MAX_BYTES)

    def finalize(self) -> tuple[TruncationResult, str | None]:
        text = "".join(self._chunks)
        trunc = truncate_tail(text, self.MAX_LINES, self.MAX_BYTES)
        return trunc, self._temp_file_path
