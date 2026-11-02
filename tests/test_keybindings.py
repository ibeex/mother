"""Tests for config-driven key binding overrides."""

import asyncio
from unittest.mock import AsyncMock, patch

from mother import MotherApp
from mother.config import MotherConfig
from mother.widgets import PromptTextArea


def test_app_keybinding_override_is_applied() -> None:
    async def run() -> None:
        app = MotherApp(
            config=MotherConfig(
                model="test-model",
                keybindings={"toggle_auto_scroll": "ctrl+t"},
            )
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.auto_scroll_enabled is True

            await pilot.press("ctrl+t")
            await pilot.pause()

            assert app.auto_scroll_enabled is False

    asyncio.run(run())


def test_quit_keybinding_override_is_applied() -> None:
    async def run() -> None:
        app = MotherApp(
            config=MotherConfig(
                model="test-model",
                keybindings={"quit": "ctrl+x"},
            )
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            quit_bindings = {
                key: value.binding
                for key, value in app.active_bindings.items()
                if value.binding.action == "quit_app"
            }

            assert set(quit_bindings) == {"ctrl+x"}
            assert quit_bindings["ctrl+x"].id == "quit"

    asyncio.run(run())


def test_prompt_submit_keybinding_override_is_applied() -> None:
    async def run() -> None:
        app = MotherApp(
            config=MotherConfig(
                model="test-model",
                keybindings={"submit": "ctrl+enter", "newline": "enter"},
            )
        )

        async with app.run_test() as pilot:
            text_area = app.query_one(PromptTextArea)
            text_area.load_text("hello")
            text_area.move_cursor((0, len("hello")))
            await pilot.pause()

            with patch.object(app, "action_submit", new_callable=AsyncMock) as action_submit:
                await pilot.press("enter")
                await pilot.pause()
                assert text_area.text == "hello\n"
                action_submit.assert_not_awaited()

                await pilot.press("ctrl+enter")
                await pilot.pause()

            action_submit.assert_awaited_once_with()

    asyncio.run(run())
