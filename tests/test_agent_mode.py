"""Tests for agent-mode toggles, slash commands, and status-line behavior."""

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from pydantic_ai.messages import ModelMessage

from mother import MotherApp, MotherConfig
from mother.models import ModelEntry
from mother.widgets import ModelComplete, PromptTextArea, StatusLine


def _reasoning_entry() -> ModelEntry:
    return ModelEntry(
        id="test-model",
        name="test-model",
        api_type="openai-responses",
        supports_reasoning=True,
        supports_tools=True,
        supports_images=True,
    )


def test_app_starts_in_conversational_mode() -> None:
    app = MotherApp()
    assert app.agent_mode is False


def test_agent_mode_toggle_via_palette() -> None:
    from mother.model_picker import AgentModeProvider

    assert AgentModeProvider in MotherApp.COMMANDS


def test_toggle_agent_mode_on() -> None:
    app = MotherApp()
    app.action_toggle_agent_mode()
    assert app.agent_mode is True


def test_toggle_agent_mode_off() -> None:
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
        app = MotherApp(config=MotherConfig(model="test-model"))

        with patch.object(app, "exit") as exit_app:
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
        app = MotherApp(config=MotherConfig(model="test-model"))

        with patch.object(app, "action_show_models") as show_models:
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
        app = MotherApp(config=MotherConfig(model="test-model"))

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


def test_reasoning_command_updates_runtime_setting() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model", models=[_reasoning_entry()]))

        with patch.object(app, "notify") as notify:
            async with app.run_test() as pilot:
                text_area = app.query_one(PromptTextArea)
                text_area.load_text("/reasoning high")
                await pilot.pause()

                await pilot.press("enter")
                await pilot.pause()

                assert app.config.reasoning_effort == "high"
                assert text_area.text == ""
                notify.assert_called_with(
                    "Reasoning set to high for test-model",
                    title="Reasoning",
                )

    asyncio.run(run())


def test_reasoning_command_tab_opens_inline_picker() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))

        async with app.run_test() as pilot:
            text_area = app.query_one(PromptTextArea)
            text_area.load_text("/reasoning")
            text_area.move_cursor((0, len("/reasoning")), record_width=False)
            await pilot.pause()

            await pilot.press("tab")
            await pilot.pause()

            model_complete = app.query_one(ModelComplete)
            assert text_area.text == "/reasoning "
            assert model_complete.display is True
            assert text_area.model_complete_active is True

    asyncio.run(run())


def test_reasoning_command_partial_query_enter_uses_highlighted_value() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model", models=[_reasoning_entry()]))

        with patch.object(app, "notify") as notify:
            async with app.run_test() as pilot:
                text_area = app.query_one(PromptTextArea)
                text_area.load_text("/reasoning h")
                await pilot.pause()

                await pilot.press("enter")
                await pilot.pause()

                assert app.config.reasoning_effort == "high"
                assert text_area.text == ""
                notify.assert_called_with(
                    "Reasoning set to high for test-model",
                    title="Reasoning",
                )

    asyncio.run(run())


def test_models_command_tab_opens_inline_model_picker() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))
        available_models = [
            ("gpt-5", "gpt-5"),
            ("claude-opus-4-1", "claude-opus-4-1 — Opus"),
        ]

        with patch("mother.slash_commands.get_available_models", return_value=available_models):
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


def test_models_command_query_enter_switches_model() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))
        available_models = [
            ("claude-opus-4-1", "claude-opus-4-1 — Opus"),
            ("gpt-5", "gpt-5"),
        ]

        with (
            patch("mother.slash_commands.get_available_models", return_value=available_models),
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


def test_toggle_auto_scroll_off() -> None:
    app = MotherApp()
    with patch.object(app, "notify"):
        app.action_toggle_auto_scroll()
    assert app.auto_scroll_enabled is False


def test_scroll_to_bottom_uses_forced_scroll() -> None:
    app = MotherApp()
    with patch.object(app, "_scroll_chat_to_end") as scroll_to_end:
        app.action_scroll_to_bottom()
    scroll_to_end.assert_called_once_with(force=True)


def test_shift_g_scrolls_to_bottom_when_input_is_not_focused() -> None:
    app = MotherApp()
    with (
        patch.object(MotherApp, "focused", new_callable=PropertyMock, return_value=None),
        patch.object(app, "action_scroll_to_bottom") as scroll_to_bottom,
    ):
        app.action_scroll_to_bottom_from_chat()
    scroll_to_bottom.assert_called_once_with()


def test_shift_g_does_nothing_when_input_is_focused() -> None:
    app = MotherApp()
    text_area = PromptTextArea()
    with (
        patch.object(MotherApp, "focused", new_callable=PropertyMock, return_value=text_area),
        patch.object(app, "action_scroll_to_bottom") as scroll_to_bottom,
    ):
        app.action_scroll_to_bottom_from_chat()
    scroll_to_bottom.assert_not_called()


def test_subtitle_shows_agent_indicator() -> None:
    app = MotherApp()
    app.agent_mode = False
    app.config = MotherConfig(model="test-model")
    app._update_subtitle()  # pyright: ignore[reportPrivateUsage]
    assert "[AGENT]" not in app.sub_title

    app.agent_mode = True
    app._update_subtitle()  # pyright: ignore[reportPrivateUsage]
    assert "[AGENT]" in app.sub_title


def test_config_tools_enabled_sets_initial_mode() -> None:
    config = MotherConfig(tools_enabled=True)
    app = MotherApp(config=config)
    assert app.agent_mode is True


def test_statusline_formats_model_and_unknown_context() -> None:
    assert StatusLine.format_status("test-model", False, None) == "test-model · off · ? · auto"


def test_statusline_formats_model_and_context_size() -> None:
    assert StatusLine.format_status("test-model", True, 12345) == "test-model · on · 12.3k · auto"


def test_statusline_formats_token_usage_when_known() -> None:
    assert (
        StatusLine.format_status(
            "test-model",
            True,
            12345,
            False,
            "high",
            1.25,
            12345,
            678,
            9000,
        )
        == "test-model · on · 12.3k · in 12.3k · out 678 · cache 9.0k · manual · high · last 1.2s"
    )


def test_statusline_formats_manual_scroll_mode() -> None:
    assert (
        StatusLine.format_status("test-model", True, 256, False) == "test-model · on · 256 · manual"
    )


def test_statusline_formats_reasoning_when_supported() -> None:
    assert (
        StatusLine.format_status("test-model", True, 256, False, "high")
        == "test-model · on · 256 · manual · high"
    )


def test_statusline_formats_last_response_time() -> None:
    assert (
        StatusLine.format_status(
            "test-model",
            True,
            256,
            False,
            "high",
            1.25,
        )
        == "test-model · on · 256 · manual · high · last 1.2s"
    )


def test_statusline_formats_subsecond_last_response_time() -> None:
    assert StatusLine.format_response_time(0.806) == "0.8s"


def test_statusline_formats_minute_last_response_time() -> None:
    assert StatusLine.format_response_time(61.0) == "1m 1s"


def test_statusline_formats_hour_last_response_time() -> None:
    assert StatusLine.format_response_time(3661.0) == "1h 1m 1s"


def test_status_reasoning_effort_visible_for_reasoning_models() -> None:
    app = MotherApp(config=MotherConfig(reasoning_effort="none"))
    app.current_model_entry = _reasoning_entry()
    assert app._status_reasoning_effort() == "off"  # pyright: ignore[reportPrivateUsage]


def test_send_prompt_no_tools_when_conversational() -> None:
    app = MotherApp()
    app.agent_mode = False
    with patch("mother.mother.get_default_tools") as mock_tools:
        registry = MagicMock()
        registry.is_empty.return_value = True  # pyright: ignore[reportAny]
        mock_tools.return_value = registry
        mock_tools(tools_enabled=False)
        mock_tools.assert_called_with(tools_enabled=False)


def test_send_prompt_passes_tools_when_agent() -> None:
    app = MotherApp()
    app.agent_mode = True
    with patch("mother.mother.get_default_tools") as mock_tools:
        registry = MagicMock()
        registry.is_empty.return_value = False  # pyright: ignore[reportAny]
        registry.tools.return_value = [MagicMock()]  # pyright: ignore[reportAny]
        mock_tools.return_value = registry
        mock_tools(tools_enabled=True)
        mock_tools.assert_called_with(tools_enabled=True)


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
