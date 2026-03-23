"""Tests for session persistence and markdown export."""

import json
from pathlib import Path
from typing import cast, final
from unittest.mock import call, patch

from mother import MotherApp, MotherConfig
from mother.session import MarkdownFormatNotice, SessionManager, format_markdown_export


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
        attachment_paths=["/tmp/screenshot.png"],
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
    assert "Attachments: `/tmp/screenshot.png`" in markdown
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
        attachment_paths=[],
    )
    manager.record_prompt(
        user_text="two",
        prompt_text="two",
        system_prompt="same prompt",
        agent_mode=False,
        tool_names=[],
        attachment_paths=[],
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


def test_new_session_keeps_previous_live_session_jsonl(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    markdown_dir = tmp_path / "markdown"

    first = SessionManager.create(sessions_dir=sessions_dir, markdown_dir=markdown_dir)
    first.append("user", "still running")
    lines = first.path.read_text(encoding="utf-8").splitlines()
    loaded_header = cast(object, json.loads(lines[0]))
    if not isinstance(loaded_header, dict):
        raise AssertionError("Expected session header object")
    header = cast(dict[str, object], loaded_header)
    header["pid"] = 12345
    _ = first.path.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n", encoding="utf-8")

    with patch("mother.session._process_is_alive", return_value=True):
        second = SessionManager.create(sessions_dir=sessions_dir, markdown_dir=markdown_dir)

    assert first.path.exists() is True
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


def test_save_last_rejects_active_session(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    markdown_dir = tmp_path / "markdown"

    manager = SessionManager.create(sessions_dir=sessions_dir, markdown_dir=markdown_dir)
    manager.append("user", "still open")
    lines = manager.path.read_text(encoding="utf-8").splitlines()
    loaded_header = cast(object, json.loads(lines[0]))
    if not isinstance(loaded_header, dict):
        raise AssertionError("Expected session header object")
    header = cast(dict[str, object], loaded_header)
    header["pid"] = 12345
    _ = manager.path.write_text(
        "\n".join([json.dumps(header), *lines[1:]]) + "\n", encoding="utf-8"
    )

    with patch("mother.session._process_is_alive", return_value=True):
        try:
            _ = SessionManager.save_last(sessions_dir=sessions_dir, markdown_dir=markdown_dir)
        except RuntimeError as exc:
            assert (
                str(exc)
                == "Last session is still active in another Mother instance. Use /save there instead."
            )
        else:
            raise AssertionError("Expected save_last to reject active sessions")


def test_save_as_markdown_reports_missing_active_session_log(tmp_path: Path):
    manager = SessionManager.create(
        sessions_dir=tmp_path / "sessions",
        markdown_dir=tmp_path / "markdown",
    )
    manager.append("user", "hello")
    manager.path.unlink()

    try:
        _ = manager.save_as_markdown()
    except RuntimeError as exc:
        assert (
            str(exc)
            == "Current session log is missing on disk. It may have been removed by another Mother instance or `mother --save`."
        )
    else:
        raise AssertionError("Expected a missing active session log error")


def test_append_recreates_session_header_when_log_was_deleted(tmp_path: Path):
    manager = SessionManager.create(
        sessions_dir=tmp_path / "sessions",
        markdown_dir=tmp_path / "markdown",
    )
    manager.append("user", "hello")
    manager.path.unlink()

    manager.append("assistant", "back again")

    lines = manager.path.read_text(encoding="utf-8").splitlines()
    loaded_header = cast(object, json.loads(lines[0]))
    if not isinstance(loaded_header, dict):
        raise AssertionError("Expected session header object")
    header = cast(dict[str, object], loaded_header)
    assert header["type"] == "session"
    assert any("back again" in line for line in lines[1:])


def test_format_markdown_export_uses_rumdl_when_uv_is_available(tmp_path: Path):
    markdown_path = tmp_path / "mother.md"
    _ = markdown_path.write_text("# Mother Session\n", encoding="utf-8")

    with (
        patch("mother.session.shutil.which", return_value="/usr/bin/uv"),
        patch("mother.session.subprocess.run") as run,
    ):
        notice = format_markdown_export(markdown_path)

    assert notice is None
    run.assert_called_once_with(
        ["uv", "run", "rumdl", "fmt", "--disable", "MD013", str(markdown_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_format_markdown_export_reports_missing_uv(tmp_path: Path):
    markdown_path = tmp_path / "mother.md"

    with patch("mother.session.shutil.which", return_value=None):
        notice = format_markdown_export(markdown_path)

    assert notice == MarkdownFormatNotice(
        "Install uv to enable better markdown formatting on save."
    )


def test_action_save_session_keeps_current_session_manager(tmp_path: Path):
    config = MotherConfig(session_markdown_dir=str(tmp_path / "markdown"))
    saved_path = tmp_path / "markdown" / "mother.md"
    fake_session_manager = _FakeSessionManager(saved_path)
    app = MotherApp(config=config)
    app.session_manager = cast(SessionManager, cast(object, fake_session_manager))

    with (
        patch.object(app, "notify") as notify,
        patch("mother.mother.format_markdown_export", return_value=None),
    ):
        app.action_save_session()

    assert fake_session_manager.save_calls == 1
    assert app.session_manager is not None
    notify.assert_called_once_with(f"Saved to {saved_path}", title="Session")


def test_action_save_session_notifies_when_uv_is_missing(tmp_path: Path):
    config = MotherConfig(session_markdown_dir=str(tmp_path / "markdown"))
    saved_path = tmp_path / "markdown" / "mother.md"
    fake_session_manager = _FakeSessionManager(saved_path)
    app = MotherApp(config=config)
    app.session_manager = cast(SessionManager, cast(object, fake_session_manager))

    with (
        patch.object(app, "notify") as notify,
        patch(
            "mother.mother.format_markdown_export",
            return_value=MarkdownFormatNotice(
                "Install uv to enable better markdown formatting on save."
            ),
        ),
    ):
        app.action_save_session()

    assert fake_session_manager.save_calls == 1
    assert notify.call_args_list == [
        call(f"Saved to {saved_path}", title="Session"),
        call("Install uv to enable better markdown formatting on save.", title="Session"),
    ]
