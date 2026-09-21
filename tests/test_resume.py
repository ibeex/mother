"""Tests for resuming persisted sessions."""

import asyncio
from pathlib import Path
from time import sleep

from textual.containers import VerticalScroll

from mother import MotherApp
from mother.app_session import AppSession
from mother.config import MotherConfig
from mother.models import ModelEntry
from mother.session import SessionManager
from mother.widgets import ConversationTurn, WelcomeBanner


def _config() -> MotherConfig:
    return MotherConfig(
        model="test-model",
        models=[ModelEntry(id="test-model", name="test-model", api_type="openai-responses")],
    )


def test_load_last_restores_completed_turns_and_appends_to_same_file(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    cwd = tmp_path / "project"
    manager = SessionManager.create(sessions_dir=sessions_dir, cwd=cwd, model_name="test-model")
    manager.append("user", "What did I ask?")
    manager.record_prompt(
        user_text="What did I ask?",
        prompt_text="What did I ask?",
        system_prompt="system",
        agent_mode=True,
        tool_names=[],
        attachment_paths=[],
    )
    manager.append("assistant", "You asked a question.")

    resumed = SessionManager.load_last(sessions_dir=sessions_dir, cwd=cwd)

    assert resumed is not None
    assert resumed.path == manager.path
    app_session = AppSession(_config(), session_manager=resumed, loaded_session=True)
    assert app_session.loaded_session is True
    assert app_session.agent_mode is True
    assert len(app_session.conversation_state.message_history) == 2
    assert [message.content for message in app_session.conversation_state.transcript_messages] == [
        "What did I ask?",
        "You asked a question.",
    ]

    resumed.append("user", "Continue")
    assert manager.path.read_text(encoding="utf-8").count('"type": "session"') == 1
    assert '"content": "Continue"' in manager.path.read_text(encoding="utf-8")


def test_resumed_app_renders_history_instead_of_the_welcome_banner(tmp_path: Path) -> None:
    async def run() -> None:
        manager = SessionManager.create(
            sessions_dir=tmp_path / "sessions", cwd=tmp_path / "project"
        )
        manager.append("user", "hello")
        manager.append("assistant", "welcome back")
        resumed = SessionManager.load_last(
            sessions_dir=tmp_path / "sessions", cwd=tmp_path / "project"
        )
        assert resumed is not None
        app = MotherApp(config=_config(), session_manager=resumed, loaded_session=True)

        async with app.run_test() as pilot:
            await pilot.pause()
            chat_view = app.query_one("#chat-view", VerticalScroll)
            assert not any(isinstance(child, WelcomeBanner) for child in chat_view.children)
            turns = [child for child in chat_view.children if isinstance(child, ConversationTurn)]
            assert len(turns) == 1

    asyncio.run(run())


def test_list_sessions_is_newest_first_for_the_requested_cwd(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    cwd = tmp_path / "project"
    first = SessionManager.create(sessions_dir=sessions_dir, cwd=cwd)
    first.append("user", "first")
    sleep(0.01)
    second = SessionManager.create(sessions_dir=sessions_dir, cwd=cwd)
    second.append("user", "second")
    other = SessionManager.create(sessions_dir=sessions_dir, cwd=tmp_path / "other")
    other.append("user", "other")

    sessions = SessionManager.list_sessions(sessions_dir=sessions_dir, cwd=cwd)

    assert [session.path for session in sessions] == [second.path, first.path]


def test_load_last_rejects_an_unknown_session_version(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    cwd = tmp_path / "project"
    manager = SessionManager.create(sessions_dir=sessions_dir, cwd=cwd)
    manager.append("user", "hello")
    contents = manager.path.read_text(encoding="utf-8")
    _ = manager.path.write_text(
        contents.replace('"version": 3', '"version": 999'), encoding="utf-8"
    )

    assert SessionManager.load_last(sessions_dir=sessions_dir, cwd=cwd) is None
