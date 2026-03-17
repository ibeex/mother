"""Tests for session persistence and markdown export."""

from pathlib import Path
from typing import cast, final
from unittest.mock import patch

from mother import MotherApp, MotherConfig
from mother.session import SessionManager


@final
class _FakeSessionManager:
    output_path: Path
    save_calls: int

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.save_calls = 0

    def save_as_markdown(self) -> Path:
        self.save_calls += 1
        return self.output_path


def test_session_save_writes_markdown_and_keeps_transient_jsonl_for_future_saves(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    markdown_dir = tmp_path / "markdown"

    manager = SessionManager.create(
        sessions_dir=sessions_dir,
        markdown_dir=markdown_dir,
        cwd=tmp_path / "project",
        model_name="gpt-test",
    )
    manager.append("user", "Hello")
    manager.record_prompt(
        user_text="Hello",
        prompt_text="Hello",
        system_prompt="You are Mother.",
        agent_mode=False,
        tool_names=[],
    )
    manager.record_event("model_change", {"from": "gpt-test", "model": "gpt-4o-mini"})
    manager.record_tool_call(
        tool_name="bash",
        tool_call_id="call-1",
        arguments={"command": "pwd"},
    )
    manager.record_tool_result(
        tool_name="bash",
        tool_call_id="call-1",
        arguments={"command": "pwd"},
        output="/tmp/project\n",
    )
    manager.append("assistant", "Hi there")

    output_path = manager.save_as_markdown()

    assert output_path.exists()
    markdown = output_path.read_text(encoding="utf-8")
    assert "# Mother Session" in markdown
    assert "## Session Summary" in markdown
    assert "## System Prompt" in markdown
    assert "prompt contexts" in markdown
    assert "tool calls" in markdown
    assert "tool results" in markdown
    assert "Models seen: `gpt-test`, `gpt-4o-mini`" in markdown
    assert "### Prompt Context" in markdown
    assert markdown.count("## System Prompt") == 1
    assert "### Tool Call · `bash`" in markdown
    assert "### Tool Result · `bash`" not in markdown
    assert "Tool output" in markdown
    assert "### Event · `model_change`" in markdown
    assert "## User" in markdown
    assert "Hello" in markdown
    assert "## Assistant" in markdown
    assert "Hi there" in markdown
    assert "**Model:** `gpt-test`" in markdown
    assert manager.path.exists() is True
    assert manager.last_file.exists() is True


def test_prompt_context_does_not_repeat_same_system_prompt(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    markdown_dir = tmp_path / "markdown"

    manager = SessionManager.create(sessions_dir=sessions_dir, markdown_dir=markdown_dir)
    manager.record_prompt(
        user_text="one",
        prompt_text="one",
        system_prompt="same prompt",
        agent_mode=False,
        tool_names=[],
    )
    manager.record_prompt(
        user_text="two",
        prompt_text="two",
        system_prompt="same prompt",
        agent_mode=False,
        tool_names=[],
    )

    output_path = manager.save_as_markdown()
    markdown = output_path.read_text(encoding="utf-8")

    assert "## System Prompt" in markdown
    assert markdown.count("same prompt") == 1
    assert "System prompt updated" not in markdown


def test_repeated_saves_overwrite_same_markdown_file_for_one_session(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    markdown_dir = tmp_path / "markdown"

    manager = SessionManager.create(sessions_dir=sessions_dir, markdown_dir=markdown_dir)
    manager.append("user", "first")
    first_output_path = manager.save_as_markdown()
    first_markdown = first_output_path.read_text(encoding="utf-8")

    manager.append("assistant", "second")
    second_output_path = manager.save_as_markdown()
    second_markdown = second_output_path.read_text(encoding="utf-8")

    assert second_output_path == first_output_path
    assert "first" in first_markdown
    assert "second" not in first_markdown
    assert "first" in second_markdown
    assert "second" in second_markdown


def test_new_session_deletes_previous_unsaved_jsonl(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    markdown_dir = tmp_path / "markdown"

    first = SessionManager.create(sessions_dir=sessions_dir, markdown_dir=markdown_dir)
    first.append("user", "unsaved")
    assert first.path.exists()

    second = SessionManager.create(sessions_dir=sessions_dir, markdown_dir=markdown_dir)

    assert first.path.exists() is False
    assert second.last_file.read_text(encoding="utf-8").strip() == str(second.path)


def test_save_last_exports_existing_unsaved_session(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    markdown_dir = tmp_path / "markdown"

    manager = SessionManager.create(sessions_dir=sessions_dir, markdown_dir=markdown_dir)
    manager.append("user", "recover me")
    manager.append("assistant", "saved")

    output_path = SessionManager.save_last(sessions_dir=sessions_dir, markdown_dir=markdown_dir)

    assert output_path is not None
    assert output_path.exists()
    markdown = output_path.read_text(encoding="utf-8")
    assert "recover me" in markdown
    assert "saved" in markdown
    assert manager.path.exists() is False
    assert (sessions_dir / "last").exists() is False


def test_action_save_session_keeps_current_session_manager(tmp_path: Path):
    config = MotherConfig(session_markdown_dir=str(tmp_path / "markdown"))
    saved_path = tmp_path / "markdown" / "mother.md"
    fake_session_manager = _FakeSessionManager(saved_path)
    app = MotherApp(config=config)
    app.session_manager = cast(SessionManager, cast(object, fake_session_manager))

    with patch.object(app, "notify") as notify:
        app.action_save_session()

    assert fake_session_manager.save_calls == 1
    assert app.session_manager is not None
    notify.assert_called_once_with(f"Saved to {saved_path}", title="Session")
