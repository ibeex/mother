"""Tests for conversation replay into normal terminal scrollback."""

from io import StringIO
from pathlib import Path

from rich.console import Console

from mother.conversation import ConversationState
from mother.session import SessionManager
from mother.terminal_transcript import print_terminal_transcript


def test_terminal_transcript_renders_turns_and_resume_command(tmp_path: Path) -> None:
    state = ConversationState()
    state.append_transcript_turn("hi [bold]literally[/bold]", "Hello **there**!")
    state.append_transcript_turn("What next?", "Try this:\n\n```python\nprint(42)\n```")
    manager = SessionManager.create(sessions_dir=tmp_path, model_name="test-model")
    stream = StringIO()
    print_terminal_transcript(
        state,
        model_name="test-model",
        session_manager=manager,
        console=Console(file=stream, width=100, color_system=None),
    )
    output = stream.getvalue()
    assert "Mother · test-model" in output
    assert "> hi [bold]literally[/bold]" in output
    assert "Hello there!" in output
    assert output.index("Hello there!") < output.index("> What next?")
    assert "print(42)" in output
    assert f"To resume this session: mother --session {manager.header.get('id')}" in output


def test_terminal_transcript_empty_session_still_leaves_resume_hint(tmp_path: Path) -> None:
    manager = SessionManager.create(sessions_dir=tmp_path, model_name="test-model")
    stream = StringIO()
    print_terminal_transcript(
        ConversationState(),
        model_name="test-model",
        session_manager=manager,
        console=Console(file=stream, color_system=None),
    )
    assert "Mother · test-model" in stream.getvalue()
    assert "To resume this session: mother --session" in stream.getvalue()


def test_terminal_transcript_without_session_manager() -> None:
    state = ConversationState()
    state.append_transcript_turn("Question", "Answer")
    stream = StringIO()
    print_terminal_transcript(
        state,
        model_name="test-model",
        session_manager=None,
        console=Console(file=stream, color_system=None),
    )
    assert "Question" in stream.getvalue()
    assert "Answer" in stream.getvalue()
    assert "To resume" not in stream.getvalue()
