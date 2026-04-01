"""UI shell layout builders for MotherApp."""

from __future__ import annotations

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static

from mother.app_chrome import StatusLineState
from mother.slash_commands import SLASH_COMMANDS
from mother.widgets import (
    PromptHistoryComplete,
    PromptTextArea,
    SlashArgumentComplete,
    SlashComplete,
    StatusLine,
    TurnLabel,
    WelcomeBanner,
)

PROMPT_HISTORY_HELP_TEXT = "History search · ↑↓ navigate · Enter accept · Esc cancel"
COUNCIL_MULTILINE_HELP_TEXT = "Council multiline mode · Enter newline · Ctrl+Enter submit"


def build_prompt_area() -> Vertical:
    """Build the prompt input area and its inline helper popups."""
    slash_complete = SlashComplete(SLASH_COMMANDS)
    slash_complete.display = False

    slash_argument_complete = SlashArgumentComplete()
    slash_argument_complete.display = False

    prompt_history_help = Static(
        PROMPT_HISTORY_HELP_TEXT,
        id="prompt-history-help",
    )
    prompt_history_help.display = False

    prompt_history_complete = PromptHistoryComplete()
    prompt_history_complete.display = False

    prompt_council_help = Static(
        COUNCIL_MULTILINE_HELP_TEXT,
        id="prompt-council-help",
    )
    prompt_council_help.display = False

    prompt_row = Horizontal(
        TurnLabel(">", classes="turn-gutter input-gutter"),
        PromptTextArea(id="prompt-input"),
        id="prompt-row",
    )

    return Vertical(
        slash_complete,
        slash_argument_complete,
        prompt_history_help,
        prompt_history_complete,
        prompt_council_help,
        prompt_row,
        id="prompt-area",
    )


def build_main_pane() -> Vertical:
    """Build the main chat shell with chat history and prompt area."""
    return Vertical(
        VerticalScroll(WelcomeBanner(), id="chat-view"),
        build_prompt_area(),
        id="main-pane",
    )


def build_status_line(state: StatusLineState) -> StatusLine:
    """Build the footer-adjacent status line for the current app session."""
    return StatusLine(
        state.model_name,
        state.agent_mode,
        state.context_tokens,
        state.auto_scroll_enabled,
        state.reasoning_effort,
        state.last_response_time_seconds,
        state.input_tokens,
        state.output_tokens,
        state.cached_tokens,
        agent_label=state.agent_label,
    )
