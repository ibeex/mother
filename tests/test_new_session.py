"""Tests for starting a fresh chat session with /new."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from textual.containers import VerticalScroll

from mother import MotherApp, MotherConfig
from mother.bash_execution import BashExecution
from mother.session import SessionManager
from mother.stats import TurnUsage
from mother.widgets import ConversationTurn, PromptTextArea, WelcomeBanner


def test_new_command_enter_starts_new_session() -> None:
    async def run() -> None:
        app = MotherApp(config=MotherConfig(model="test-model"))

        with patch.object(app, "action_new_session") as action_new_session:
            async with app.run_test() as pilot:
                text_area = app.query_one(PromptTextArea)
                text_area.load_text("/new")
                await pilot.pause()

                await app.action_submit()
                await pilot.pause()

                action_new_session.assert_called_once_with()
                assert text_area.text == ""

    asyncio.run(run())


def test_action_new_session_clears_context_outputs_and_usage(tmp_path: Path) -> None:
    async def run() -> None:
        cwd = tmp_path / "project"
        cwd.mkdir()
        sessions_root = tmp_path / "sessions"
        markdown_dir = tmp_path / "markdown"
        session_manager = SessionManager.create(
            sessions_dir=sessions_root,
            markdown_dir=markdown_dir,
            cwd=cwd,
            model_name="test-model",
        )
        app = MotherApp(
            config=MotherConfig(model="test-model"),
            session_manager=session_manager,
        )

        app.conversation_state.append_synthetic_turn("hello", "hi")
        app.app_session.pending_executions.append(
            BashExecution(
                command="pwd",
                output=str(cwd),
                exit_code=0,
                timestamp=datetime.now(),
            )
        )
        image_path = tmp_path / "clipboard.png"
        app.app_session.pending_image_attachments[str(image_path)] = image_path
        app.app_session.apply_turn_usage(
            TurnUsage(
                request_tokens=123,
                response_tokens=45,
                cache_read_tokens=6,
                duration_seconds=1.5,
                provider="test",
                model_id="test-model",
            )
        )
        old_session_manager = app.session_manager

        async with app.run_test() as pilot:
            chat_view = app.query_one("#chat-view", VerticalScroll)
            prompt_input = app.query_one(PromptTextArea)
            prompt_input.load_text("draft")
            turn = ConversationTurn(prompt_text="hello", response_text="hi")
            app.set_active_turn(turn)
            await chat_view.mount(turn)
            await pilot.pause()

            app.action_new_session()
            await pilot.pause()

            assert len(chat_view.children) == 1
            assert isinstance(chat_view.children[0], WelcomeBanner)
            assert prompt_input.text == ""
            assert prompt_input.read_only is False
            assert app.conversation_state.has_history is False
            assert app.app_session.pending_executions == []
            assert app.app_session.pending_image_attachments == {}
            assert app.app_session.last_turn_usage is None
            assert app.app_session.last_context_tokens is None
            assert app.app_session.session_input_tokens is None
            assert app.app_session.session_output_tokens is None
            assert app.app_session.session_cached_tokens is None
            assert app.app_session.last_response_time_seconds is None
            assert app.runtime_presentation.active_turn is None
            assert app.session_manager is not None
            assert old_session_manager is not None
            assert app.session_manager is not old_session_manager
            assert app.session_manager.path != old_session_manager.path
            assert app.session_manager.sessions_dir == old_session_manager.sessions_dir
            assert app.session_manager.markdown_dir == old_session_manager.markdown_dir

    asyncio.run(run())
