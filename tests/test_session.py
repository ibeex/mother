"""Tests for session persistence and markdown export."""

import json
from pathlib import Path
from typing import cast, final
from unittest.mock import call, patch

from mother import MotherApp, MotherConfig
from mother.council import (
    CouncilAggregateRanking,
    CouncilCandidateResponse,
    CouncilPeerReview,
    CouncilResult,
)
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


def test_tool_output_with_nested_fences_uses_longer_outer_fence(tmp_path: Path) -> None:
    manager = SessionManager.create(
        sessions_dir=tmp_path / "sessions",
        markdown_dir=tmp_path / "markdown",
    )
    manager.record_tool_call(
        tool_name="bash",
        tool_call_id="call-1",
        arguments={"command": "cat README.md"},
    )
    manager.record_tool_result(
        tool_name="bash",
        tool_call_id="call-1",
        arguments={"command": "cat README.md"},
        output=(
            "# Example\n\n"
            "```text\n[[fetch https://example.com/page]]\n```\n\n"
            "## Development\n"
            "Keep going.\n"
        ),
    )
    manager.record_event("model_change", {"from": "gpt-test", "model": "gpt-4o-mini"})

    output_path = manager.save_as_markdown()
    markdown = output_path.read_text(encoding="utf-8")

    assert "````\n# Example" in markdown
    assert "````text\n# Example" not in markdown
    assert "```text\n[[fetch https://example.com/page]]\n```" in markdown
    assert "\n````\n\n</details>" in markdown
    assert "</details>\n\n---\n\n### Event · `model_change`" in markdown


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


def test_turn_usage_event_renders_compact_summary_without_json(tmp_path: Path):
    manager = SessionManager.create(
        sessions_dir=tmp_path / "sessions",
        markdown_dir=tmp_path / "markdown",
    )
    manager.record_event(
        "turn_usage",
        {
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "duration_seconds": 1.7219462500070222,
            "image_count": 0,
            "model_id": "gpt-5-mini",
            "provider": "openai-responses",
            "request_tokens": 125,
            "response_tokens": 11,
            "tool_call_errors": 0,
            "tool_calls_finished": 0,
            "tool_calls_started": 0,
            "total_tokens": 136,
        },
    )

    output_path = manager.save_as_markdown()
    markdown = output_path.read_text(encoding="utf-8")

    assert "### Event · `turn_usage`" in markdown
    assert (
        "- Usage: duration `1.72s`, request tokens `125`, response tokens `11`, total tokens `136`"
        in markdown
    )
    assert "```json" not in markdown
    assert "cache_read_tokens" not in markdown
    assert "provider" not in markdown
    assert "model_id" not in markdown


def test_tool_limit_recovery_event_renders_compact_summary_without_json(tmp_path: Path) -> None:
    manager = SessionManager.create(
        sessions_dir=tmp_path / "sessions",
        markdown_dir=tmp_path / "markdown",
    )
    manager.record_event(
        "tool_limit_recovery",
        {
            "strategy": "text_only",
            "model": "gpt-5-mini",
            "mode": "agent",
            "profile": "standard",
            "tool_call_limit": 1,
            "tool_calls_started": 1,
            "tool_calls_finished": 1,
        },
    )

    output_path = manager.save_as_markdown()
    markdown = output_path.read_text(encoding="utf-8")

    assert "### Event · `tool_limit_recovery`" in markdown
    assert "- Recovery: `text only` after reaching tool-call limit `1`" in markdown
    assert "- Model: `gpt-5-mini`" in markdown
    assert "- Mode: `agent`" in markdown
    assert "- Profile: `standard`" in markdown
    assert "- Tool calls: started `1`, finished `1`" in markdown
    assert "```json" not in markdown


def test_council_completed_event_renders_full_trace_sections(tmp_path: Path) -> None:
    manager = SessionManager.create(
        sessions_dir=tmp_path / "sessions",
        markdown_dir=tmp_path / "markdown",
    )
    result = CouncilResult(
        final_text="Flagged rollout recommended.",
        judge_model_id="opus",
        stage1=(
            CouncilCandidateResponse(
                label="Response A",
                model_id="gpt-5",
                text="Roll out in two phases.",
            ),
            CouncilCandidateResponse(
                label="Response B",
                model_id="g3",
                text="Use a feature flag first.",
            ),
        ),
        stage2=(
            CouncilPeerReview(
                reviewer_model_id="opus",
                text="Response B is safer.\n\nFINAL RANKING:\n1. Response B\n2. Response A",
                parsed_ranking=("Response B", "Response A"),
            ),
        ),
        aggregate_rankings=(
            CouncilAggregateRanking(label="Response B", average_rank=1.0, rankings_count=1),
        ),
        label_to_model={"Response A": "gpt-5", "Response B": "g3"},
        duration_seconds=9.5,
    )

    manager.record_event(
        "council_invoked",
        {
            "question": "How should we launch this?",
            "members": ["gpt-5", "g3", "opus"],
            "judge": "opus",
        },
    )
    manager.append("assistant", result.final_text)
    manager.record_event("council_completed", result.to_event_details())

    output_path = manager.save_as_markdown()
    markdown = output_path.read_text(encoding="utf-8")

    assert "### Event · `council_invoked`" in markdown
    assert "- Members: `gpt-5`, `g3`, `opus`" in markdown
    assert "### Event · `council_completed`" in markdown
    assert "- Judge: `opus`" in markdown
    assert "- Stage 1 responses: `2`" in markdown
    assert "- Stage 2 reviews: `1`" in markdown
    assert "- Duration: `9.50s`" in markdown
    assert "<summary>Council · Stage 1 · Response A · gpt-5</summary>" in markdown
    assert "Roll out in two phases." in markdown
    assert "<summary>Council · Stage 2 · Review 1</summary>" in markdown
    assert "Parsed ranking: Response B · g3 > Response A · gpt-5" in markdown
    assert "<summary>Council · Stage 2 · Aggregate rankings</summary>" in markdown
    assert "<summary>Council · Stage 3 · Judge metadata</summary>" in markdown
    assert (
        "Models seen: `opus`, `gpt-5`, `g3`" in markdown
        or "Models seen: `gpt-5`, `g3`, `opus`" in markdown
    )
    assert '"trace_sections"' not in markdown


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
