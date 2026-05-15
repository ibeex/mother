"""Tests for runtime presentation follow/scroll behavior."""

from __future__ import annotations

from typing import cast, final

from mother.runtime_presentation import RuntimePresentationController
from mother.widgets import Response


@final
class _FakeChatView:
    def __init__(self, host: _FakeHost) -> None:
        self.host = host
        self.mounted: list[object] = []

    def mount(self, widget: object) -> None:
        self.mounted.append(widget)
        self.host.near_end = False

    def scroll_end(self, *, animate: bool = False) -> None:
        self.host.scroll_end_calls.append(animate)


@final
class _FakeHost:
    def __init__(self) -> None:
        self.auto_scroll_enabled = True
        self.near_end = True
        self.scroll_calls: list[bool] = []
        self.scroll_end_calls: list[bool] = []
        self.chat_view = _FakeChatView(self)

    def query_one(self, selector: object, expect_type: object = None) -> object:
        _ = selector, expect_type
        return self.chat_view

    def should_follow_chat_updates(self) -> bool:
        return self.auto_scroll_enabled and self.near_end

    def scroll_chat_to_end(self, *, force: bool = False) -> None:
        self.scroll_calls.append(force)


class _FakeResponse:
    def __init__(self) -> None:
        self.updated_texts: list[str] = []
        self.active_classes: set[str] = set()

    def add_class(self, class_name: str) -> None:
        self.active_classes.add(class_name)

    def remove_class(self, class_name: str) -> None:
        self.active_classes.discard(class_name)

    def update(self, text: str) -> None:
        self.updated_texts.append(text)


def test_show_tool_started_keeps_following_when_mount_pushes_view_off_end() -> None:
    host = _FakeHost()
    presentation = RuntimePresentationController(host, waiting_messages=("WAITING",))

    presentation.show_tool_started("bash", "bash-1", {"command": "pwd"})

    assert len(host.chat_view.mounted) == 1
    assert host.scroll_calls == [True]


def test_start_response_waiting_animation_uses_follow_state_before_render() -> None:
    host = _FakeHost()
    presentation = RuntimePresentationController(host, waiting_messages=("WAITING",))
    response = _FakeResponse()

    presentation.start_response_waiting_animation(cast(Response, cast(object, response)))

    assert response.updated_texts == ["`W`AITING"]
    assert "response-awaiting" in response.active_classes
    assert host.scroll_calls == [True]
