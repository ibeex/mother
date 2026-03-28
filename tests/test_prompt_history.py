"""Tests for prompt-history persistence and prompt input navigation."""

import asyncio
from pathlib import Path
from unittest.mock import patch

from textual.binding import Binding, BindingType

from mother import MotherApp
from mother.history import PromptHistory
from mother.widgets import PromptHistoryComplete, PromptTextArea


def _binding_signature(binding: BindingType) -> tuple[str, str]:
    if isinstance(binding, Binding):
        return binding.key, binding.action
    return binding[0], binding[1]


def test_prompt_history_appends_persists_and_searches(tmp_path: Path) -> None:
    history_path = tmp_path / "prompt_history.jsonl"
    history = PromptHistory(history_path)

    history.append("first prompt")
    history.append("alpha test")
    history.append("second alpha test")

    reloaded = PromptHistory(history_path)

    assert reloaded.size == 3
    assert reloaded.entry(1) == "second alpha test"
    assert reloaded.entry(2) == "alpha test"
    assert reloaded.entry(3) == "first prompt"
    assert reloaded.find_previous("alpha") == (1, "second alpha test")
    assert reloaded.find_previous("alpha", before_index=1) == (2, "alpha test")
    assert reloaded.find_previous("missing") is None
    assert [match.text for match in reloaded.search("alpha")] == [
        "alpha test",
        "second alpha test",
    ]


def test_prompt_text_area_exposes_history_search_binding() -> None:
    bindings = {_binding_signature(binding) for binding in PromptTextArea.BINDINGS}

    assert ("ctrl+r", "history_search") in bindings


def test_up_and_down_browse_prompt_history_and_restore_draft(tmp_path: Path) -> None:
    async def run() -> None:
        history = PromptHistory(tmp_path / "prompt_history.jsonl")
        history.append("first")
        history.append("second")
        app = MotherApp(prompt_history=history)

        async with app.run_test() as pilot:
            await pilot.pause()
            text_area = app.query_one(PromptTextArea)
            text_area.load_text("draft")
            await pilot.pause()

            await pilot.press("up")
            await pilot.pause()
            assert text_area.text == "second"

            await pilot.press("up")
            await pilot.pause()
            assert text_area.text == "first"

            await pilot.press("down")
            await pilot.pause()
            assert text_area.text == "second"

            await pilot.press("down")
            await pilot.pause()
            assert text_area.text == "draft"

    asyncio.run(run())


def test_ctrl_r_on_empty_opens_fuzzy_history_search(tmp_path: Path) -> None:
    async def run() -> None:
        history = PromptHistory(tmp_path / "prompt_history.jsonl")
        history.append("older alpha")
        history.append("beta")
        history.append("newer alpha")
        app = MotherApp(prompt_history=history)

        async with app.run_test() as pilot:
            await pilot.pause()
            text_area = app.query_one(PromptTextArea)
            history_complete = app.query_one(PromptHistoryComplete)

            await pilot.press("ctrl+r")
            await pilot.pause()

            assert text_area.history_search_active is True
            assert history_complete.display is True
            assert [match.text for match in history_complete.matches] == [
                "newer alpha",
                "beta",
                "older alpha",
            ]

            await pilot.press("a", "l", "p")
            await pilot.pause()
            assert [match.text for match in history_complete.matches] == [
                "newer alpha",
                "older alpha",
            ]

            await pilot.press("enter")
            await pilot.pause()
            assert text_area.text == "newer alpha"
            assert text_area.history_search_active is False
            assert history_complete.display is False

    asyncio.run(run())


def test_ctrl_r_on_current_input_shows_multiple_fuzzy_matches(tmp_path: Path) -> None:
    async def run() -> None:
        history = PromptHistory(tmp_path / "prompt_history.jsonl")
        history.append("older alpha")
        history.append("beta")
        history.append("newer alpha")
        app = MotherApp(prompt_history=history)

        async with app.run_test() as pilot:
            await pilot.pause()
            text_area = app.query_one(PromptTextArea)
            history_complete = app.query_one(PromptHistoryComplete)
            text_area.load_text("alpha")
            await pilot.pause()

            await pilot.press("ctrl+r")
            await pilot.pause()

            assert text_area.text == "alpha"
            assert text_area.history_search_active is True
            assert history_complete.display is True
            assert [match.text for match in history_complete.matches] == [
                "newer alpha",
                "older alpha",
            ]

            await pilot.press("down")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert text_area.text == "older alpha"
            assert history_complete.display is False

    asyncio.run(run())


def test_history_search_escape_restores_original_text(tmp_path: Path) -> None:
    async def run() -> None:
        history = PromptHistory(tmp_path / "prompt_history.jsonl")
        history.append("older alpha")
        history.append("newer alpha")
        app = MotherApp(prompt_history=history)

        async with app.run_test() as pilot:
            await pilot.pause()
            text_area = app.query_one(PromptTextArea)
            history_complete = app.query_one(PromptHistoryComplete)
            text_area.load_text("alpha")
            await pilot.pause()

            await pilot.press("ctrl+r")
            await pilot.pause()
            await pilot.press("backspace")
            await pilot.pause()
            assert history_complete.display is True

            await pilot.press("escape")
            await pilot.pause()

            assert text_area.text == "alpha"
            assert text_area.history_search_active is False
            assert history_complete.display is False

    asyncio.run(run())


def test_submitted_normal_prompt_is_written_to_prompt_history(tmp_path: Path) -> None:
    async def run() -> None:
        history_path = tmp_path / "prompt_history.jsonl"
        app = MotherApp(prompt_history=PromptHistory(history_path))

        async with app.run_test() as pilot:
            await pilot.pause()
            text_area = app.query_one(PromptTextArea)
            text_area.load_text("hello there")
            await pilot.pause()

            with patch.object(app, "send_prompt", return_value=object()) as send_prompt:
                await app.action_submit()
                await pilot.pause()
                send_prompt.assert_called_once()

        reloaded = PromptHistory(history_path)
        assert reloaded.entry(1) == "hello there"

    asyncio.run(run())


def test_slash_command_is_not_written_to_prompt_history(tmp_path: Path) -> None:
    async def run() -> None:
        history_path = tmp_path / "prompt_history.jsonl"
        app = MotherApp(prompt_history=PromptHistory(history_path))

        async with app.run_test() as pilot:
            await pilot.pause()
            text_area = app.query_one(PromptTextArea)
            text_area.load_text("/agent")
            await pilot.pause()

            with patch.object(app, "action_toggle_agent_mode") as toggle_agent_mode:
                await app.action_submit()
                await pilot.pause()
                toggle_agent_mode.assert_called_once_with()

        reloaded = PromptHistory(history_path)
        assert reloaded.size == 0

    asyncio.run(run())


def test_bang_command_is_written_to_prompt_history(tmp_path: Path) -> None:
    async def run() -> None:
        history_path = tmp_path / "prompt_history.jsonl"
        app = MotherApp(prompt_history=PromptHistory(history_path))

        async with app.run_test() as pilot:
            await pilot.pause()
            text_area = app.query_one(PromptTextArea)
            text_area.load_text("!ls -la")
            await pilot.pause()

            def fake_run_worker(coro: object, **_kwargs: object) -> object:
                close = getattr(coro, "close", None)
                if callable(close):
                    _ = close()
                return object()

            with patch.object(app, "run_worker", side_effect=fake_run_worker) as run_worker:
                await app.action_submit()
                await pilot.pause()
                run_worker.assert_called_once()

        reloaded = PromptHistory(history_path)
        assert reloaded.entry(1) == "!ls -la"

    asyncio.run(run())


def test_double_bang_command_is_not_written_to_prompt_history(tmp_path: Path) -> None:
    async def run() -> None:
        history_path = tmp_path / "prompt_history.jsonl"
        app = MotherApp(prompt_history=PromptHistory(history_path))

        async with app.run_test() as pilot:
            await pilot.pause()
            text_area = app.query_one(PromptTextArea)
            text_area.load_text("!!ls -la")
            await pilot.pause()

            def fake_run_worker(coro: object, **_kwargs: object) -> object:
                close = getattr(coro, "close", None)
                if callable(close):
                    _ = close()
                return object()

            with patch.object(app, "run_worker", side_effect=fake_run_worker) as run_worker:
                await app.action_submit()
                await pilot.pause()
                run_worker.assert_called_once()

        reloaded = PromptHistory(history_path)
        assert reloaded.size == 0

    asyncio.run(run())
