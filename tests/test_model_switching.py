"""Tests for model switching and model-picker actions."""

import asyncio
from collections.abc import Callable, Sequence
from unittest.mock import patch

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from textual.containers import VerticalScroll

from mother import MotherApp
from mother.config import MotherConfig
from mother.widgets import ConversationTurn, OutputSection, ShellOutput


def _history_text(messages: Sequence[object]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    parts.append(part.content)
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, TextPart):
                    parts.append(part.content)
    return "\n".join(parts)


def test_model_switch_preserves_agent_mode() -> None:
    app = MotherApp()
    app.agent_mode = True
    app.action_switch_model("gpt-4o-mini")
    assert app.agent_mode is True


def test_model_switch_syncs_tools_enabled_to_runtime_state() -> None:
    """config.tools_enabled should reflect the live agent_mode after a switch."""
    app = MotherApp()
    app.agent_mode = True
    app.action_switch_model("gpt-4o-mini")
    assert app.config.tools_enabled is True

    app.agent_mode = False
    app.action_switch_model("gpt-5")
    assert app.config.tools_enabled is False


def test_show_models_selection_calls_switch_model() -> None:
    app = MotherApp()

    def fake_push_screen(_screen: object, callback: Callable[[str | None], None]) -> None:
        callback("gpt-4o-mini")

    with (
        patch.object(app, "push_screen", side_effect=fake_push_screen),
        patch.object(app, "action_switch_model") as switch_model,
    ):
        app.action_show_models()

    switch_model.assert_called_once_with("gpt-4o-mini")


def test_switch_model_preserves_message_history() -> None:
    app = MotherApp()
    app.conversation_state.message_history = [
        ModelRequest(parts=[UserPromptPart("hello")]),
        ModelResponse(parts=[TextPart(content="hi")]),
    ]

    app.action_switch_model("gpt-4o-mini")

    assert app.config.model == "gpt-4o-mini"
    assert app.conversation_state.has_history is True
    assert "hello" in _history_text(app.conversation_state.message_history)
    assert "hi" in _history_text(app.conversation_state.message_history)
    assert app.current_model_entry.id == "gpt-4o-mini"


def test_switch_model_does_not_show_context_loss_confirmation() -> None:
    app = MotherApp()
    app.conversation_state.message_history = [
        ModelRequest(parts=[UserPromptPart("hello")]),
        ModelResponse(parts=[TextPart(content="hi")]),
    ]

    with patch.object(app, "push_screen") as push_screen:
        app.action_switch_model("gpt-4o-mini")

    push_screen.assert_not_called()
    assert app.config.model == "gpt-4o-mini"


def test_switch_model_waits_when_runtime_is_busy() -> None:
    app = MotherApp(config=MotherConfig(model="test-model"))
    app.set_active_turn(ConversationTurn(prompt_text="hello"))

    with patch.object(app, "notify") as notify:
        app.action_switch_model("gpt-4o-mini")

    assert app.config.model == "test-model"
    notify.assert_called_once()


def test_switch_model_switches_when_visible_turn_exists() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))

        async with app.run_test() as pilot:
            chat_view = app.query_one("#chat-view", VerticalScroll)
            _ = await chat_view.mount(ConversationTurn(prompt_text="hello"))
            await pilot.pause()

            assert app.conversation_state.has_history is False

            with patch.object(app, "push_screen") as push_screen:
                app.action_switch_model("gpt-4o-mini")
                await pilot.pause()

            push_screen.assert_not_called()
            assert app.config.model == "gpt-4o-mini"

    asyncio.run(run())


def test_switch_model_does_not_warn_for_shell_output_only() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))

        async with app.run_test() as pilot:
            chat_view = app.query_one("#chat-view", VerticalScroll)
            _ = await chat_view.mount(
                OutputSection("Shell", "shell-title", ShellOutput("pwd\n/tmp"))
            )
            await pilot.pause()

            with patch.object(app, "notify"), patch.object(app, "push_screen") as push_screen:
                app.action_switch_model("gpt-4o-mini")
                await pilot.pause()

            assert app.config.model == "gpt-4o-mini"
            push_screen.assert_not_called()

    asyncio.run(run())
