"""Tests for agent-mode toggles, slash commands, and status-line behavior."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from mother import MotherApp, MotherConfig
from mother.models import ModelEntry
from mother.widgets import ModelComplete, PromptTextArea


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


def test_toggle_agent_mode_on() -> None:
    app = MotherApp()
    app.action_toggle_agent_mode()
    assert app.agent_mode is True


def test_toggle_agent_mode_off() -> None:
    app = MotherApp()
    app.action_toggle_agent_mode()
    app.action_toggle_agent_mode()
    assert app.agent_mode is False


def test_set_deep_research_mode_on() -> None:
    app = MotherApp()
    with patch.object(app, "notify"):
        app.action_set_agent_profile("deep_research")
    assert app.agent_mode is True
    assert app.agent_profile == "deep_research"


def test_prompt_enter_selects_slash_completion() -> None:
    text_area = PromptTextArea()
    text_area.slash_complete_active = True

    with patch.object(text_area, "post_message") as post_message:
        asyncio.run(text_area.handle_enter_key())

    post_message.assert_called_once()
    assert isinstance(post_message.call_args.args[0], PromptTextArea.SlashAccept)


def test_prompt_enter_selects_slash_argument_completion() -> None:
    text_area = PromptTextArea()
    text_area.slash_argument_complete_active = True

    with patch.object(text_area, "post_message") as post_message:
        asyncio.run(text_area.handle_enter_key())

    post_message.assert_called_once()
    assert isinstance(post_message.call_args.args[0], PromptTextArea.SlashArgumentAccept)


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

            assert app.agent_mode is True
            assert text_area.text == ""

    asyncio.run(run())


def test_agent_deep_research_command_enter_enables_research_mode() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))

        async with app.run_test() as pilot:
            text_area = app.query_one(PromptTextArea)
            text_area.load_text("/agent deep research")
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert text_area.text == "/agent deep research"
            assert app.agent_mode is False

            await pilot.press("enter")
            await pilot.pause()

            assert app.agent_mode is True
            assert app.agent_profile == "deep_research"
            assert text_area.text == ""

    asyncio.run(run())


def test_agent_command_tab_opens_inline_profile_picker() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))

        async with app.run_test() as pilot:
            text_area = app.query_one(PromptTextArea)
            text_area.load_text("/agent")
            text_area.move_cursor((0, len("/agent")), record_width=False)
            await pilot.pause()

            await pilot.press("tab")
            await pilot.pause()

            model_complete = app.query_one(ModelComplete)
            assert text_area.text == "/agent "
            assert model_complete.display is True
            assert text_area.model_complete_active is True

    asyncio.run(run())


def test_agent_command_partial_query_enter_accepts_inline_selection_before_submit() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))

        async with app.run_test() as pilot:
            text_area = app.query_one(PromptTextArea)
            model_complete = app.query_one(ModelComplete)
            text_area.load_text("/agent d")
            await pilot.pause()

            assert model_complete.display is True

            await pilot.press("enter")
            await pilot.pause()

            assert text_area.text == "/agent deep research"
            assert model_complete.display is False
            assert app.agent_mode is False

            await pilot.press("enter")
            await pilot.pause()

            assert app.agent_mode is True
            assert app.agent_profile == "deep_research"
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

                assert text_area.text == "/reasoning high"
                assert app.config.reasoning_effort == "medium"

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

                assert text_area.text == "/reasoning high"
                assert app.config.reasoning_effort == "medium"

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

                assert text_area.text == "/models claude-opus-4-1"
                switch_model.assert_not_called()

                await pilot.press("enter")
                await pilot.pause()

                switch_model.assert_called_once_with("claude-opus-4-1")

    asyncio.run(run())


def test_models_command_fuzzy_query_enter_switches_model() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))
        available_models = [
            ("local_1", "local_1 — local 1"),
            ("local_2", "local_2 — local 2"),
            ("local_3", "local_3 — local 3"),
        ]

        with (
            patch("mother.slash_commands.get_available_models", return_value=available_models),
            patch.object(app, "action_switch_model") as switch_model,
        ):
            async with app.run_test() as pilot:
                text_area = app.query_one(PromptTextArea)
                text_area.load_text("/models lo3")
                await pilot.pause()

                await pilot.press("enter")
                await pilot.pause()

                assert text_area.text == "/models local_3"
                switch_model.assert_not_called()

                await pilot.press("enter")
                await pilot.pause()

                switch_model.assert_called_once_with("local_3")

    asyncio.run(run())


def test_models_command_enter_accepts_highlighted_inline_selection_before_submit() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))
        available_models = [
            ("local_1", "local_1 — local 1"),
            ("local_2", "local_2 — local 2"),
            ("local_3", "local_3 — local 3"),
        ]

        with (
            patch("mother.slash_commands.get_available_models", return_value=available_models),
            patch.object(app, "action_switch_model") as switch_model,
        ):
            async with app.run_test() as pilot:
                text_area = app.query_one(PromptTextArea)
                model_complete = app.query_one(ModelComplete)
                text_area.load_text("/models l")
                await pilot.pause()

                assert model_complete.display is True
                await pilot.press("down", "down")
                await pilot.pause()

                await pilot.press("enter")
                await pilot.pause()

                assert text_area.text == "/models local_3"
                assert model_complete.display is False
                switch_model.assert_not_called()

                await pilot.press("enter")
                await pilot.pause()

                switch_model.assert_called_once_with("local_3")

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


def test_send_prompt_no_tools_when_conversational() -> None:
    app = MotherApp()
    app.agent_mode = False
    with patch("mother.mother.get_default_tools") as mock_tools:
        registry = MagicMock()
        registry.is_empty.return_value = True  # pyright: ignore[reportAny]
        mock_tools.return_value = registry

        _ = app._get_enabled_tools()  # pyright: ignore[reportPrivateUsage]

        mock_tools.assert_called_with(
            tools_enabled=False,
            ca_bundle_path=app.config.ca_bundle_path,
            agent_profile="standard",
        )


def test_send_prompt_passes_tools_when_agent() -> None:
    app = MotherApp()
    app.agent_mode = True
    with patch("mother.mother.get_default_tools") as mock_tools:
        registry = MagicMock()
        registry.is_empty.return_value = False  # pyright: ignore[reportAny]
        registry.tools.return_value = [MagicMock()]  # pyright: ignore[reportAny]
        mock_tools.return_value = registry

        _ = app._get_enabled_tools()  # pyright: ignore[reportPrivateUsage]

        mock_tools.assert_called_with(
            tools_enabled=True,
            ca_bundle_path=app.config.ca_bundle_path,
            agent_profile="standard",
        )
