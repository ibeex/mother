"""Tests for chat layout spacing."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from llm.models import Conversation
from textual.containers import VerticalScroll

from mother import MotherApp, MotherConfig
from mother.widgets import ConversationTurn


def test_last_markdown_block_has_no_trailing_margin_in_conversation_turn() -> None:
    async def run() -> None:
        model = MagicMock()
        conversation = cast(Conversation, cast(object, SimpleNamespace(responses=[])))
        model.conversation.return_value = conversation  # pyright: ignore[reportAny]
        app = MotherApp(config=MotherConfig(model="test-model"))

        with patch("mother.mother.llm.get_model", return_value=model):
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

                assert len(prompt_paragraphs) == 2
                assert prompt_paragraphs[0].styles.margin.bottom == 1
                assert prompt_paragraphs[-1].styles.margin.bottom == 0

                assert len(response_paragraphs) == 2
                assert response_paragraphs[0].styles.margin.bottom == 1
                assert response_paragraphs[-1].styles.margin.bottom == 0

    asyncio.run(run())
