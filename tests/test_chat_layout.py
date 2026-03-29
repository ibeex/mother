"""Tests for chat layout spacing."""

import asyncio

from textual.containers import Horizontal, Vertical, VerticalScroll

from mother import MotherApp, MotherConfig
from mother.widgets import (
    ConversationTurn,
    OutputSection,
    PromptTextArea,
    ToolOutput,
    TurnLabel,
    WelcomeBanner,
)


def test_chat_layout_renders_welcome_state_and_prompt_gutter() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))

        async with app.run_test() as pilot:
            await pilot.pause()
            _ = app.query_one("#main-pane", Vertical)
            chat = app.query_one("#chat-view", VerticalScroll)
            _ = app.query_one("#prompt-area", Vertical)
            prompt_row = app.query_one("#prompt-row", Horizontal)
            prompt = app.query_one("#prompt-input", PromptTextArea)
            input_gutter = app.query_one(".input-gutter", TurnLabel)
            welcome = app.query_one(WelcomeBanner)

            assert len(chat.children) == 1
            assert chat.children[0] is welcome
            assert len(prompt_row.children) == 2
            assert prompt_row.children[-1] is prompt
            assert str(input_gutter.render()) == ">"

    asyncio.run(run())


def test_tool_output_is_nested_above_response_within_turn() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))

        async with app.run_test() as pilot:
            chat = app.query_one("#chat-view", VerticalScroll)
            turn = ConversationTurn(prompt_text="prompt", response_text="reply")
            await chat.mount(turn)
            await pilot.pause()

            turn.tool_trace_stack.display = True
            await turn.tool_trace_stack.mount(
                OutputSection("Tool", "tool-title", ToolOutput("pwd"))
            )
            await pilot.pause()

            assert len(chat.children) == 2
            assert chat.children[-1] is turn
            assert turn.tool_trace_stack.display is True
            assert len(turn.tool_trace_stack.children) == 1
            assert "response-section" in turn.children[-1].classes

    asyncio.run(run())
