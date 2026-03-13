"""Helpers for separating ``<think>`` blocks from streamed model output."""

from typing import final


@final
class ThinkTagStreamParser:
    """Incrementally split streamed text into thinking and response sections."""

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False
        self.has_thinking = False

    def feed(self, chunk: str) -> tuple[str, str]:
        """Consume one streamed chunk and return thinking/response deltas."""
        self._buffer += chunk
        return self._drain(final=False)

    def flush(self) -> tuple[str, str]:
        """Flush any remaining buffered text at end-of-stream."""
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> tuple[str, str]:
        thinking_parts: list[str] = []
        response_parts: list[str] = []

        while self._buffer:
            marker = self.CLOSE_TAG if self._in_think else self.OPEN_TAG
            marker_index = self._buffer.find(marker)

            if marker_index != -1:
                prefix = self._buffer[:marker_index]
                if self._in_think:
                    thinking_parts.append(prefix)
                else:
                    response_parts.append(prefix)
                self._buffer = self._buffer[marker_index + len(marker) :]
                if not self._in_think:
                    self.has_thinking = True
                self._in_think = not self._in_think
                continue

            if final:
                prefix = self._buffer
                self._buffer = ""
            else:
                safe_length = len(self._buffer) - (len(marker) - 1)
                if safe_length <= 0:
                    break
                prefix = self._buffer[:safe_length]
                self._buffer = self._buffer[safe_length:]

            if self._in_think:
                thinking_parts.append(prefix)
            else:
                response_parts.append(prefix)

        return "".join(thinking_parts), "".join(response_parts)
