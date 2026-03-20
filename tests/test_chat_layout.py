"""Tests for chat layout spacing."""

import asyncio

from textual.containers import Horizontal, Vertical, VerticalScroll

from mother import MotherApp, MotherConfig
from mother.widgets import ConversationTurn, PromptTextArea, TurnLabel, WelcomeBanner


def test_last_markdown_block_has_no_trailing_margin_in_conversation_turn() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))

        async with app.run_test() as pilot:
            chat = app.query_one("#chat-view", VerticalScroll)
            turn = ConversationTurn(
                prompt_text="first\n\nsecond",
                response_text="alpha\n\nbeta",
            )
            await chat.mount(turn)
            await pilot.pause()

            assert turn.prompt_widget is not None
            prompt_paragraphs = list(turn.prompt_widget.query("MarkdownParagraph"))
            response_paragraphs = list(turn.response_widget.query("MarkdownParagraph"))
            gutters = list(turn.query(TurnLabel))

            assert len(prompt_paragraphs) == 2
            assert prompt_paragraphs[0].styles.margin.bottom == 1
            assert prompt_paragraphs[-1].styles.margin.bottom == 0

            assert len(response_paragraphs) == 2
            assert response_paragraphs[0].styles.margin.bottom == 1
            assert response_paragraphs[-1].styles.margin.bottom == 0

            assert len(gutters) == 1
            assert str(gutters[0].render()) == ">"

    asyncio.run(run())


def test_chat_layout_uses_minimal_chrome_and_prompt_gutter() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))

        async with app.run_test() as pilot:
            await pilot.pause()
            pane = app.query_one("#main-pane", Vertical)
            chat = app.query_one("#chat-view", VerticalScroll)
            prompt_area = app.query_one("#prompt-area", Vertical)
            prompt_row = app.query_one("#prompt-row", Horizontal)
            prompt = app.query_one("#prompt-input", PromptTextArea)
            input_gutter = app.query_one(".input-gutter", TurnLabel)
            welcome = app.query_one(WelcomeBanner)

            assert len(chat.children) == 1
            assert chat.children[0] is welcome
            assert "INTERFACE 2037 READY" in str(welcome.render())
            assert "MU-TH-UR 6000 SYSTEM" in str(welcome.render())
            assert pane.styles.margin.left == 0
            assert pane.styles.margin.right == 0
            assert pane.styles.border_left[0] == ""
            assert pane.styles.border_right[0] == ""
            assert prompt_area.styles.border_top[0] == "solid"
            assert prompt.styles.border_top[0] == ""
            assert prompt.styles.border_left[0] == ""
            assert prompt.styles.border_right[0] == ""
            assert len(prompt_row.children) == 2
            assert str(input_gutter.render()) == ">"

    asyncio.run(run())
