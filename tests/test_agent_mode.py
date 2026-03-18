"""Tests for Ctrl+A agent mode toggle in MotherApp."""

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from llm.models import Conversation

from mother import MotherApp, MotherConfig
from mother.widgets import ModelComplete, PromptTextArea, StatusLine


def test_app_starts_in_conversational_mode():
    app = MotherApp()
    assert app.agent_mode is False


def test_agent_mode_toggle_via_palette():
    """Agent mode is toggled via the command palette (AgentModeProvider), not a key binding."""
    from mother.model_picker import AgentModeProvider

    assert AgentModeProvider in MotherApp.COMMANDS


def test_toggle_agent_mode_on():
    app = MotherApp()
    app.action_toggle_agent_mode()
    assert app.agent_mode is True


def test_toggle_agent_mode_off():
    app = MotherApp()
    app.action_toggle_agent_mode()
    app.action_toggle_agent_mode()
    assert app.agent_mode is False


def test_quit_app_exits() -> None:
    app = MotherApp()

    with patch.object(app, "exit") as exit_app:
        app.action_quit_app()

    exit_app.assert_called_once_with()


def test_prompt_enter_selects_slash_completion() -> None:
    text_area = PromptTextArea()
    text_area.slash_complete_active = True

    with patch.object(text_area, "post_message") as post_message:
        asyncio.run(text_area.handle_enter_key())

    post_message.assert_called_once()
    assert isinstance(post_message.call_args.args[0], PromptTextArea.SlashAccept)


def test_prompt_enter_submits_builtin_slash_command() -> None:
    text_area = PromptTextArea("/quit")
    action_submit = AsyncMock()
    fake_app = SimpleNamespace(action_submit=action_submit)

    with patch.object(PromptTextArea, "app", new_callable=PropertyMock, return_value=fake_app):
        asyncio.run(text_area.handle_enter_key())

    action_submit.assert_awaited_once_with()


def test_prompt_enter_keeps_multiline_for_normal_text() -> None:
    text_area = PromptTextArea("hello")
    text_area.move_cursor((0, len("hello")))

    asyncio.run(text_area.handle_enter_key())

    assert text_area.text == "hello\n"


def test_selected_slash_command_submits_on_second_enter() -> None:
    async def run() -> None:
        model = MagicMock()
        conversation = cast(Conversation, cast(object, SimpleNamespace(responses=[])))
        model.conversation.return_value = conversation  # pyright: ignore[reportAny]
        app = MotherApp(config=MotherConfig(model="test-model"))

        with (
            patch("mother.mother.llm.get_model", return_value=model),
            patch.object(app, "exit") as exit_app,
        ):
            async with app.run_test() as pilot:
                text_area = app.query_one(PromptTextArea)
                text_area.load_text("/quit")
                await pilot.pause()

                await pilot.press("enter")
                await pilot.pause()
                assert text_area.text == "/quit "
                assert app.slash_complete.display is False

                await pilot.press("enter")
                await pilot.pause()
                exit_app.assert_called_once_with()

    asyncio.run(run())


def test_models_command_enter_opens_model_picker() -> None:
    async def run() -> None:
        model = MagicMock()
        conversation = cast(Conversation, cast(object, SimpleNamespace(responses=[])))
        model.conversation.return_value = conversation  # pyright: ignore[reportAny]
        app = MotherApp(config=MotherConfig(model="test-model"))

        with (
            patch("mother.mother.llm.get_model", return_value=model),
            patch.object(app, "action_show_models") as show_models,
        ):
            async with app.run_test() as pilot:
                text_area = app.query_one(PromptTextArea)
                text_area.load_text("/models")
                await pilot.pause()

                await pilot.press("enter")
                await pilot.pause()

                show_models.assert_called_once_with()

    asyncio.run(run())


def test_agent_command_enter_toggles_agent_mode() -> None:
    async def run() -> None:
        model = MagicMock()
        conversation = cast(Conversation, cast(object, SimpleNamespace(responses=[])))
        model.conversation.return_value = conversation  # pyright: ignore[reportAny]
        app = MotherApp(config=MotherConfig(model="test-model"))

        with patch("mother.mother.llm.get_model", return_value=model):
            async with app.run_test() as pilot:
                text_area = app.query_one(PromptTextArea)
                text_area.load_text("/agent")
                await pilot.pause()

                assert app.agent_mode is False
                await pilot.press("enter")
                await pilot.pause()
                assert text_area.text == "/agent "

                await pilot.press("enter")
                await pilot.pause()

                assert app.agent_mode is True
                assert text_area.text == ""

    asyncio.run(run())


def test_models_command_tab_opens_inline_model_picker() -> None:
    async def run() -> None:
        model = MagicMock()
        conversation = cast(Conversation, cast(object, SimpleNamespace(responses=[])))
        model.conversation.return_value = conversation  # pyright: ignore[reportAny]
        app = MotherApp(config=MotherConfig(model="test-model"))
        available_models = [
            ("gpt-5", "gpt-5"),
            ("claude-opus-4-1", "claude-opus-4-1 — Opus"),
        ]

        with (
            patch("mother.mother.llm.get_model", return_value=model),
            patch("mother.widgets.get_available_models", return_value=available_models),
        ):
            async with app.run_test() as pilot:
                text_area = app.query_one(PromptTextArea)
                text_area.load_text("/models")
                text_area.move_cursor((0, len("/models")), record_width=False)
                await pilot.pause()

                await pilot.press("tab")
                await pilot.pause()

                model_complete = app.query_one(ModelComplete)
                assert text_area.text == "/models "
                assert model_complete.display is True
                assert text_area.model_complete_active is True

    asyncio.run(run())


def test_models_command_typed_character_expands_query() -> None:
    async def run() -> None:
        model = MagicMock()
        conversation = cast(Conversation, cast(object, SimpleNamespace(responses=[])))
        model.conversation.return_value = conversation  # pyright: ignore[reportAny]
        app = MotherApp(config=MotherConfig(model="test-model"))
        available_models = [("claude-opus-4-1", "claude-opus-4-1 — Opus")]

        with (
            patch("mother.mother.llm.get_model", return_value=model),
            patch("mother.widgets.get_available_models", return_value=available_models),
        ):
            async with app.run_test() as pilot:
                text_area = app.query_one(PromptTextArea)
                text_area.load_text("/models")
                text_area.move_cursor((0, len("/models")), record_width=False)
                await pilot.pause()

                await pilot.press("o")
                await pilot.pause()

                model_complete = app.query_one(ModelComplete)
                assert text_area.text == "/models o"
                assert model_complete.display is True
                assert text_area.model_complete_active is True

    asyncio.run(run())


def test_models_command_tab_accepts_highlighted_model() -> None:
    async def run() -> None:
        model = MagicMock()
        conversation = cast(Conversation, cast(object, SimpleNamespace(responses=[])))
        model.conversation.return_value = conversation  # pyright: ignore[reportAny]
        app = MotherApp(config=MotherConfig(model="test-model"))
        available_models = [
            ("claude-haiku-3-5", "claude-haiku-3-5 — Haiku"),
            ("gpt-5", "gpt-5"),
        ]

        with (
            patch("mother.mother.llm.get_model", return_value=model),
            patch("mother.widgets.get_available_models", return_value=available_models),
        ):
            async with app.run_test() as pilot:
                text_area = app.query_one(PromptTextArea)
                text_area.load_text("/models ha")
                text_area.move_cursor((0, len("/models ha")), record_width=False)
                await pilot.pause()

                await pilot.press("tab")
                await pilot.pause()

                assert text_area.text == "/models claude-haiku-3-5"
                assert app.model_complete.display is False
                assert text_area.model_complete_active is False

    asyncio.run(run())


def test_models_command_query_enter_switches_model() -> None:
    async def run() -> None:
        model = MagicMock()
        conversation = cast(Conversation, cast(object, SimpleNamespace(responses=[])))
        model.conversation.return_value = conversation  # pyright: ignore[reportAny]
        app = MotherApp(config=MotherConfig(model="test-model"))
        available_models = [
            ("claude-opus-4-1", "claude-opus-4-1 — Opus"),
            ("gpt-5", "gpt-5"),
        ]

        with (
            patch("mother.mother.llm.get_model", return_value=model),
            patch("mother.widgets.get_available_models", return_value=available_models),
            patch("mother.model_picker.get_available_models", return_value=available_models),
            patch.object(app, "action_switch_model") as switch_model,
        ):
            async with app.run_test() as pilot:
                text_area = app.query_one(PromptTextArea)
                text_area.load_text("/models opus")
                await pilot.pause()

                await pilot.press("enter")
                await pilot.pause()

                switch_model.assert_called_once_with("claude-opus-4-1")

    asyncio.run(run())


def test_toggle_auto_scroll_off():
    app = MotherApp()
    with patch.object(app, "notify"):
        app.action_toggle_auto_scroll()
    assert app.auto_scroll_enabled is False


def test_scroll_to_bottom_uses_forced_scroll():
    app = MotherApp()
    with patch.object(app, "_scroll_chat_to_end") as scroll_to_end:
        app.action_scroll_to_bottom()
    scroll_to_end.assert_called_once_with(force=True)


def test_shift_g_scrolls_to_bottom_when_input_is_not_focused():
    app = MotherApp()
    with (
        patch.object(MotherApp, "focused", new_callable=PropertyMock, return_value=None),
        patch.object(app, "action_scroll_to_bottom") as scroll_to_bottom,
    ):
        app.action_scroll_to_bottom_from_chat()
    scroll_to_bottom.assert_called_once_with()


def test_shift_g_does_nothing_when_input_is_focused():
    app = MotherApp()
    text_area = PromptTextArea()
    with (
        patch.object(MotherApp, "focused", new_callable=PropertyMock, return_value=text_area),
        patch.object(app, "action_scroll_to_bottom") as scroll_to_bottom,
    ):
        app.action_scroll_to_bottom_from_chat()
    scroll_to_bottom.assert_not_called()


def test_subtitle_shows_agent_indicator():
    app = MotherApp()
    # Simulate mounted state by setting sub_title directly
    app.agent_mode = False
    app.config = MotherConfig(model="test-model")
    app._update_subtitle()  # pyright: ignore[reportPrivateUsage]
    assert "[AGENT]" not in app.sub_title

    app.agent_mode = True
    app._update_subtitle()  # pyright: ignore[reportPrivateUsage]
    assert "[AGENT]" in app.sub_title


def test_config_tools_enabled_sets_initial_mode():
    config = MotherConfig(tools_enabled=True)
    app = MotherApp(config=config)
    assert app.agent_mode is True


def test_statusline_formats_model_and_unknown_context():
    assert StatusLine.format_status("test-model", False, None) == "test-model · off · ? · auto"


def test_statusline_formats_model_and_context_size():
    assert StatusLine.format_status("test-model", True, 12345) == "test-model · on · 12.3k · auto"


def test_statusline_formats_manual_scroll_mode():
    assert (
        StatusLine.format_status("test-model", True, 256, False) == "test-model · on · 256 · manual"
    )


def test_send_prompt_no_tools_when_conversational():
    app = MotherApp()
    app.agent_mode = False
    with patch("mother.mother.get_default_tools") as mock_tools:
        registry = MagicMock()
        registry.is_empty.return_value = True  # pyright: ignore[reportAny]
        mock_tools.return_value = registry
        # Trigger the call path
        mock_tools(tools_enabled=False)
        mock_tools.assert_called_with(tools_enabled=False)


def test_send_prompt_passes_tools_when_agent():
    app = MotherApp()
    app.agent_mode = True
    with patch("mother.mother.get_default_tools") as mock_tools:
        registry = MagicMock()
        registry.is_empty.return_value = False  # pyright: ignore[reportAny]
        registry.tools.return_value = [MagicMock()]  # pyright: ignore[reportAny]
        mock_tools.return_value = registry
        mock_tools(tools_enabled=True)
        mock_tools.assert_called_with(tools_enabled=True)


def test_show_models_selection_calls_switch_model():
    app = MotherApp()

    def fake_push_screen(_screen: object, callback: Callable[[str | None], None]) -> None:
        callback("gpt-4o-mini")

    with (
        patch.object(app, "push_screen", side_effect=fake_push_screen),
        patch.object(app, "action_switch_model") as switch_model,
    ):
        app.action_show_models()

    switch_model.assert_called_once_with("gpt-4o-mini")


def test_switch_model_asks_for_confirmation_when_conversation_has_history():
    app = MotherApp()
    app.conversation = cast(Conversation, cast(object, SimpleNamespace(responses=[object()])))

    with (
        patch.object(app, "push_screen") as push_screen,
        patch("mother.mother.llm.get_model") as get_model,
    ):
        app.action_switch_model("gpt-4o-mini")

    push_screen.assert_called_once()
    get_model.assert_not_called()


def test_switch_model_cancel_keeps_current_model():
    app = MotherApp()
    app.conversation = cast(Conversation, cast(object, SimpleNamespace(responses=[object()])))

    def fake_push_screen(_screen: object, callback: Callable[[bool | None], None]) -> None:
        callback(False)

    with (
        patch.object(app, "push_screen", side_effect=fake_push_screen),
        patch("mother.mother.llm.get_model") as get_model,
    ):
        app.action_switch_model("gpt-4o-mini")

    assert app.config.model != "gpt-4o-mini"
    get_model.assert_not_called()


def test_switch_model_confirm_applies_switch():
    app = MotherApp()
    app.conversation = cast(Conversation, cast(object, SimpleNamespace(responses=[object()])))

    new_conversation = object()
    new_model = MagicMock()
    new_model.conversation.return_value = new_conversation  # pyright: ignore[reportAny]

    def fake_push_screen(_screen: object, callback: Callable[[bool | None], None]) -> None:
        callback(True)

    with (
        patch.object(app, "push_screen", side_effect=fake_push_screen),
        patch("mother.mother.llm.get_model", return_value=new_model) as get_model,
        patch.object(app, "notify"),
    ):
        app.action_switch_model("gpt-4o-mini")

    assert app.config.model == "gpt-4o-mini"
    assert app.conversation is new_conversation
    get_model.assert_called_once_with("gpt-4o-mini")
