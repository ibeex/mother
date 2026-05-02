"""UI-oriented runtime presentation helpers for MotherApp."""

from __future__ import annotations

from dataclasses import dataclass
from random import choice
from typing import Protocol, cast

from textual.app import ScreenStackError
from textual.containers import VerticalScroll
from textual.css.query import NoMatches

from mother.council import CouncilResult
from mother.tool_trace import format_tool_event, format_tool_limit_recovery
from mother.widgets import (
    ConversationTurn,
    CouncilOutput,
    OutputSection,
    Response,
    ThinkingOutput,
    ToolOutput,
)


class RuntimePresentationHost(Protocol):
    """Minimal MotherApp surface used by the runtime presentation controller."""

    auto_scroll_enabled: bool

    def query_one(self, selector: object, expect_type: object = None) -> object: ...

    def should_follow_chat_updates(self) -> bool: ...

    def scroll_chat_to_end(self, *, force: bool = False) -> None: ...


@dataclass
class ResponseWaitingAnimation:
    """Track the lightweight response placeholder animation for a single turn."""

    response: Response
    message: str
    frame_index: int = 0


class RuntimePresentationController:
    """Encapsulate chat-view presentation for streamed runtime activity."""

    def __init__(
        self,
        host: RuntimePresentationHost,
        *,
        waiting_messages: tuple[str, ...],
    ) -> None:
        self.host: RuntimePresentationHost = host
        self.waiting_messages: tuple[str, ...] = waiting_messages
        self.active_turn: ConversationTurn | None = None
        self._tool_outputs: dict[str, ToolOutput] = {}
        self._response_waiting_animations: dict[int, ResponseWaitingAnimation] = {}

    def _chat_view(self) -> VerticalScroll:
        return cast(VerticalScroll, self.host.query_one("#chat-view", VerticalScroll))

    def chat_is_near_end(self, margin: int = 1) -> bool:
        """Return whether the chat view is already at or very near the bottom."""
        try:
            chat_view = self._chat_view()
        except (NoMatches, ScreenStackError):
            return True
        return (chat_view.max_scroll_y - chat_view.scroll_y) <= margin

    def should_follow_chat_updates(self) -> bool:
        """Return whether the next chat update should keep following new output."""
        return self.host.auto_scroll_enabled and self.chat_is_near_end()

    def scroll_chat_to_end(self, *, force: bool = False) -> None:
        """Keep the chat view pinned to the latest streamed content."""
        if not force and not self.host.auto_scroll_enabled:
            return
        try:
            chat_view = self._chat_view()
        except (NoMatches, ScreenStackError):
            return
        _ = chat_view.scroll_end(animate=False)

    def _tool_output_key(
        self,
        tool_name: str,
        tool_call_id: str | None,
        arguments: dict[str, object],
    ) -> str:
        if tool_call_id:
            return tool_call_id
        command = arguments.get("command")
        if isinstance(command, str) and command:
            return f"{tool_name}:{command}"
        return tool_name

    def _mount_trace_section(self, section: OutputSection) -> None:
        active_turn = self.active_turn
        if active_turn is not None:
            active_turn.tool_trace_stack.display = True
            _ = active_turn.tool_trace_stack.mount(section)
            return
        _ = self._chat_view().mount(section)

    def show_tool_started(
        self,
        tool_name: str,
        tool_call_id: str | None,
        arguments: dict[str, object],
    ) -> None:
        """Mount a widget showing that a tool call has started."""
        key = self._tool_output_key(tool_name, tool_call_id, arguments)
        widget = ToolOutput(format_tool_event(tool_name, arguments, status="started"))
        self._tool_outputs[key] = widget
        self._mount_trace_section(OutputSection("Tool", "tool-title", widget))
        if self.host.should_follow_chat_updates():
            self.host.scroll_chat_to_end(force=True)

    def show_tool_finished(
        self,
        tool_name: str,
        tool_call_id: str | None,
        arguments: dict[str, object],
        output: str,
    ) -> None:
        """Update a tool trace widget when a tool call finishes."""
        key = self._tool_output_key(tool_name, tool_call_id, arguments)
        widget = self._tool_outputs.pop(key, None)
        if widget is None:
            widget = ToolOutput()
            self._mount_trace_section(OutputSection("Tool", "tool-title", widget))
        widget.set_text(format_tool_event(tool_name, arguments, status="finished", output=output))
        if self.host.should_follow_chat_updates():
            self.host.scroll_chat_to_end(force=True)

    def show_tool_limit_recovery(
        self,
        tool_call_limit: int | None,
        mode: str,
        profile: str,
    ) -> None:
        """Mount a visible trace section when a turn falls back to text-only recovery."""
        widget = ToolOutput(
            format_tool_limit_recovery(
                tool_call_limit=tool_call_limit,
                mode=mode,
                profile=profile,
            )
        )
        self._mount_trace_section(OutputSection("Recovery", "recovery-title", widget))
        if self.host.should_follow_chat_updates():
            self.host.scroll_chat_to_end(force=True)

    def show_council_trace(self, result: CouncilResult) -> None:
        """Mount inspectable council stage traces within the active conversation turn."""
        sections = result.trace_sections()
        if not sections:
            return

        try:
            for trace in sections:
                self._mount_trace_section(
                    OutputSection(trace.title, "council-title", CouncilOutput(trace.text))
                )
        except (NoMatches, ScreenStackError):
            return

        if self.host.should_follow_chat_updates():
            self.host.scroll_chat_to_end(force=True)

    def waiting_response_positions(self, message: str | None = None) -> tuple[int, ...]:
        """Return the character positions used by the animated waiting wave."""
        active_message = message or self.waiting_messages[0]
        return tuple(index for index, character in enumerate(active_message) if character.isalnum())

    def waiting_response_highlight_position(
        self,
        frame_index: int,
        message: str | None = None,
    ) -> int:
        """Return the current highlighted character index for the waiting wave."""
        positions = self.waiting_response_positions(message)
        if len(positions) <= 1:
            return positions[0] if positions else 0
        cycle_length = (len(positions) * 2) - 2
        step = frame_index % cycle_length
        if step < len(positions):
            return positions[step]
        return positions[cycle_length - step]

    def waiting_response_text(self, frame_index: int, message: str | None = None) -> str:
        """Render the waiting message with a highlighted character sweep."""
        active_message = message or self.waiting_messages[0]
        highlight_index = self.waiting_response_highlight_position(frame_index, active_message)
        return (
            f"{active_message[:highlight_index]}`{active_message[highlight_index]}`"
            f"{active_message[highlight_index + 1 :]}"
        )

    def _render_response_waiting_frame(self, animation: ResponseWaitingAnimation) -> None:
        _ = animation.response.add_class("response-awaiting")
        _ = animation.response.update(
            self.waiting_response_text(animation.frame_index, animation.message)
        )

    def start_response_waiting_animation(
        self,
        response: Response,
        message: str | None = None,
    ) -> None:
        """Begin animating the temporary MU-TH-UR-style waiting line."""
        animation = ResponseWaitingAnimation(
            response=response,
            message=message or choice(self.waiting_messages),
        )
        self._response_waiting_animations[id(response)] = animation
        self._render_response_waiting_frame(animation)
        if self.host.should_follow_chat_updates():
            self.host.scroll_chat_to_end(force=True)

    def set_response_waiting_message(self, response: Response, message: str) -> None:
        """Update the animated waiting line for an in-flight response."""
        animation = self._response_waiting_animations.get(id(response))
        if animation is None:
            self.start_response_waiting_animation(response, message)
            return
        animation.message = message
        animation.frame_index = 0
        self._render_response_waiting_frame(animation)
        if self.host.should_follow_chat_updates():
            self.host.scroll_chat_to_end(force=True)

    def clear_response_waiting_animation(self, response: Response) -> None:
        """Remove waiting animation classes and stop updating the response widget."""
        _ = self._response_waiting_animations.pop(id(response), None)
        _ = response.remove_class("response-awaiting")

    def tick_response_waiting_animations(self) -> None:
        """Advance the lightweight waiting animation for any pending responses."""
        if not self._response_waiting_animations:
            return
        should_follow = self.host.should_follow_chat_updates()
        for animation in self._response_waiting_animations.values():
            animation.frame_index += 1
            self._render_response_waiting_frame(animation)
        if should_follow:
            self.host.scroll_chat_to_end(force=True)

    def has_waiting_animation(self, response: Response) -> bool:
        """Return whether the response currently has an active waiting animation."""
        return id(response) in self._response_waiting_animations

    async def update_response_output(self, response: Response, text: str) -> None:
        """Render response text incrementally while following when appropriate."""
        had_waiting_animation = self.has_waiting_animation(response)
        self.clear_response_waiting_animation(response)
        should_follow = self.host.should_follow_chat_updates()
        current_text = response.raw_markdown
        if had_waiting_animation:
            await response.replace_markdown(text)
        elif text.startswith(current_text):
            await response.append_fragment(text[len(current_text) :])
        else:
            await response.replace_markdown(text)
        if should_follow:
            self.host.scroll_chat_to_end(force=True)

    def start_thinking_output(self, thinking_output: ThinkingOutput) -> None:
        """Switch the structured thinking widget into live streaming mode."""
        should_follow = self.host.should_follow_chat_updates()
        thinking_output.start_streaming()
        if should_follow:
            self.host.scroll_chat_to_end(force=True)

    def update_thinking_output(self, thinking_output: ThinkingOutput, text: str) -> None:
        """Render streamed thinking text in the dedicated widget."""
        should_follow = self.host.should_follow_chat_updates()
        thinking_output.set_text(text)
        if should_follow:
            self.host.scroll_chat_to_end(force=True)

    def finish_thinking_output(self, thinking_output: ThinkingOutput) -> None:
        """Collapse the structured thinking widget back to its preview."""
        should_follow = self.host.should_follow_chat_updates()
        thinking_output.finish_streaming()
        if should_follow:
            self.host.scroll_chat_to_end(force=True)
