"""Replay completed chat turns after Textual restores the normal terminal."""

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from mother.conversation import ConversationState
from mother.session import SessionManager


def print_terminal_transcript(
    conversation: ConversationState,
    *,
    model_name: str,
    session_manager: SessionManager | None,
    console: Console | None = None,
) -> None:
    """Leave a readable conversation and resume hint in terminal scrollback."""
    output = console if console is not None else Console()
    output.print()
    output.print(Text(f"Mother · {model_name}", style="bold"))
    for message in conversation.transcript_messages:
        output.print()
        if message.role == "user":
            output.print(Text(f"> {message.content}"))
        else:
            output.print(Markdown(message.content))
    output.print()
    if session_manager is not None:
        session_id = session_manager.header.get("id")
        if session_id:
            output.print(Text(f"To resume this session: mother --session {session_id}"))
