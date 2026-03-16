"""Tests for ``<think>`` parsing and the dedicated thinking widget."""

from collections.abc import Callable
from typing import cast, final
from unittest.mock import patch

from mother import MotherApp
from mother.thinking import ThinkTagStreamParser
from mother.widgets import Response, ThinkingOutput


@final
class _FakeResponse:
    def __init__(self) -> None:
        self.updated_texts: list[str] = []
        self.reset_texts: list[str] = []

    def update(self, text: str) -> None:
        self.updated_texts.append(text)

    def reset_state(self, text: str) -> None:
        self.reset_texts.append(text)


@final
class _FakeThinkingOutput:
    def __init__(self) -> None:
        self.display = False
        self.updated_texts: list[str] = []
        self.started = 0
        self.finished = 0

    def start_streaming(self) -> None:
        self.started += 1

    def finish_streaming(self) -> None:
        self.finished += 1

    def set_text(self, text: str) -> None:
        self.updated_texts.append(text)
        self.display = bool(text.strip())


def _call_from_thread(callback: object, *args: object) -> object:
    return cast(Callable[..., object], callback)(*args)


def test_think_tag_stream_parser_handles_split_tags():
    parser = ThinkTagStreamParser()
    thinking = ""
    response = ""

    for chunk in ["preface ", "<th", "ink>line 1\n", "line 2", "</th", "ink>answer"]:
        thinking_delta, response_delta = parser.feed(chunk)
        thinking += thinking_delta
        response += response_delta

    thinking_delta, response_delta = parser.flush()
    thinking += thinking_delta
    response += response_delta

    assert parser.has_thinking is True
    assert thinking == "line 1\nline 2"
    assert response == "preface answer"


def test_thinking_output_wraps_and_shows_full_text_while_streaming_then_collapses():
    lines = [f"line {index}" for index in range(12)]
    widget = ThinkingOutput()

    assert widget.soft_wrap is True

    widget.start_streaming()
    widget.set_text("\n".join(lines))

    assert widget.display is True
    assert widget.text == "\n".join(lines)

    widget.finish_streaming()

    assert "line 0" in widget.text
    assert "line 9" in widget.text
    assert "line 10" not in widget.text
    assert "Press Ctrl+O for rest." in widget.text

    widget.action_toggle_expanded()

    assert widget.text == "\n".join(lines)


def test_stream_llm_response_routes_thinking_to_dedicated_widget():
    app = MotherApp()
    response = _FakeResponse()
    thinking_output = _FakeThinkingOutput()

    with (
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
        patch.object(app, "_refresh_context_size") as refresh_context_size,
        patch.object(app, "_scroll_chat_to_end"),
    ):
        full_text = app._stream_llm_response(  # pyright: ignore[reportPrivateUsage]
            ["<th", "ink>step 1\n", "step 2</think>", "final", " answer"],
            cast(Response, cast(object, response)),
            cast(ThinkingOutput, cast(object, thinking_output)),
        )

    assert full_text == "final answer"
    assert thinking_output.display is True
    assert thinking_output.started == 1
    assert thinking_output.finished == 1
    assert thinking_output.updated_texts[-1] == "step 1\nstep 2"
    assert response.updated_texts == ["final ", "final answer"]
    assert response.reset_texts == ["final answer"]
    refresh_context_size.assert_called_once_with()
