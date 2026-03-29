"""Tests for model switching and model-picker actions."""

from collections.abc import Callable
from typing import cast
from unittest.mock import patch

from pydantic_ai.messages import ModelMessage

from mother import MotherApp


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


def test_switch_model_asks_for_confirmation_when_conversation_has_history() -> None:
    app = MotherApp()
    app.conversation_state.message_history = [cast(ModelMessage, object())]

    with patch.object(app, "push_screen") as push_screen:
        app.action_switch_model("gpt-4o-mini")

    push_screen.assert_called_once()
    assert app.config.model != "gpt-4o-mini"


def test_switch_model_cancel_keeps_current_model() -> None:
    app = MotherApp()
    app.conversation_state.message_history = [cast(ModelMessage, object())]

    def fake_push_screen(_screen: object, callback: Callable[[bool | None], None]) -> None:
        callback(False)

    with patch.object(app, "push_screen", side_effect=fake_push_screen):
        app.action_switch_model("gpt-4o-mini")

    assert app.config.model != "gpt-4o-mini"


def test_switch_model_confirm_applies_switch() -> None:
    app = MotherApp()
    app.conversation_state.message_history = [cast(ModelMessage, object())]

    def fake_push_screen(_screen: object, callback: Callable[[bool | None], None]) -> None:
        callback(True)

    with (
        patch.object(app, "push_screen", side_effect=fake_push_screen),
        patch.object(app, "notify"),
    ):
        app.action_switch_model("gpt-4o-mini")

    assert app.config.model == "gpt-4o-mini"
    assert app.conversation_state.has_history is False
    assert app.current_model_entry.id == "gpt-4o-mini"
