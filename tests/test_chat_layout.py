"""Tests for chat layout spacing."""

import asyncio

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Collapsible

from mother import MotherApp, MotherConfig
from mother.council import (
    CouncilAggregateRanking,
    CouncilAssessment,
    CouncilCandidateResponse,
    CouncilPeerReview,
    CouncilResult,
)
from mother.widgets import (
    ConversationTurn,
    OutputSection,
    PromptTextArea,
    Response,
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


def test_ctrl_e_cycles_prompt_area_size() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))

        async with app.run_test() as pilot:
            prompt_area = app.query_one("#prompt-area")

            await pilot.press("ctrl+e")
            await pilot.pause()
            assert app._prompt_area_expansion_mode == 1  # pyright: ignore[reportPrivateUsage]
            assert str(prompt_area.styles.height) == "50h"

            await pilot.press("ctrl+e")
            await pilot.pause()
            assert app._prompt_area_expansion_mode == 2  # pyright: ignore[reportPrivateUsage]
            assert str(prompt_area.styles.height) == "100h"

            await pilot.press("ctrl+e")
            await pilot.pause()
            assert app._prompt_area_expansion_mode == 0  # pyright: ignore[reportPrivateUsage]
            assert str(prompt_area.styles.height) == "auto"

    asyncio.run(run())


def test_turn_sections_are_visually_distinct() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))

        async with app.run_test() as pilot:
            chat = app.query_one("#chat-view", VerticalScroll)
            turn = ConversationTurn(
                prompt_text="prompt",
                response_text="reply",
                include_thinking=True,
            )
            await chat.mount(turn)
            turn.tool_trace_stack.display = True
            await turn.tool_trace_stack.mount(
                OutputSection("Tool", "tool-title", ToolOutput("pwd"))
            )
            await pilot.pause()

            sections = {
                "prompt": turn.query_one(".prompt-section"),
                "thinking": turn.query_one(".thinking-section"),
                "response": turn.query_one(".response-section"),
                "tool": turn.query_one(".tool-section"),
            }
            border_colors = {
                name: section.styles.border_left[1].hex for name, section in sections.items()
            }
            assert len(set(border_colors.values())) == len(sections)
            assert all(section.styles.border_left[0] == "solid" for section in sections.values())
            assert sections["tool"].styles.background.a > sections["thinking"].styles.background.a

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


def test_council_trace_is_nested_within_turn() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))

        async with app.run_test() as pilot:
            chat = app.query_one("#chat-view", VerticalScroll)
            turn = ConversationTurn(prompt_text="/council question", response_text="final reply")
            await chat.mount(turn)
            await pilot.pause()

            app._active_turn = turn  # pyright: ignore[reportPrivateUsage]
            app._show_council_trace(  # pyright: ignore[reportPrivateUsage]
                CouncilResult(
                    final_text="final reply",
                    judge_model_id="opus",
                    duration_seconds=28.84,
                    stage1=(
                        CouncilCandidateResponse(
                            label="Response A",
                            model_id="gpt-5",
                            text="**Candidate A**",
                        ),
                    ),
                    stage2=(
                        CouncilPeerReview(
                            reviewer_model_id="g3",
                            text="Response A is best.\n\nFINAL RANKING:\n1. Response A",
                            parsed_ranking=("Response A",),
                        ),
                    ),
                    aggregate_rankings=(
                        CouncilAggregateRanking(
                            label="Response A",
                            average_rank=1.0,
                            rankings_count=1,
                        ),
                    ),
                    label_to_model={"Response A": "gpt-5"},
                    assessments=(CouncilAssessment("Response A", "Clear explanation.", (1,)),),
                )
            )
            await pilot.pause()

            report = turn.query_one(".council-report")
            assert turn.children[-1] is report
            assert "response-section" in turn.children[-2].classes
            answers = report.query(Collapsible).first()
            assert answers.collapsed
            reviews = next(
                group
                for group in report.query(Collapsible)
                if group.title.startswith("Peer reviews")
            )
            table = reviews.query(Response).first().raw_markdown
            assert reviews.collapsed
            assert reviews.title == "Peer reviews & ballots (1)"
            assert "The judge cites Review 1 for the comparison note." in table
            footer = report.query_one(".council-footer")
            assert report.children[-1] is footer
            assert str(footer.render()) == "Judge: opus · 28.84s"
            assert "| g3 | gpt-5 |" in table
            assert "Response A" not in table
            _ = answers.query_one("CollapsibleTitle").focus()
            await pilot.press("enter")
            assert not answers.collapsed
            candidate = answers.query_one(Collapsible)
            assert candidate.title == "gpt-5"
            _ = candidate.query_one("CollapsibleTitle").focus()
            await pilot.press("enter")
            assert not candidate.collapsed
            assert candidate.query_one(Response).raw_markdown == "**Candidate A**"

    asyncio.run(run())


def test_tool_limit_recovery_trace_is_nested_within_turn() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))

        async with app.run_test() as pilot:
            chat = app.query_one("#chat-view", VerticalScroll)
            turn = ConversationTurn(prompt_text="prompt", response_text="reply")
            await chat.mount(turn)
            await pilot.pause()

            app._active_turn = turn  # pyright: ignore[reportPrivateUsage]
            app.runtime_presentation.show_tool_limit_recovery(1, "agent", "standard")
            await pilot.pause()

            assert turn.tool_trace_stack.display is True
            assert len(turn.tool_trace_stack.children) == 1
            assert "response-section" in turn.children[-1].classes

    asyncio.run(run())
