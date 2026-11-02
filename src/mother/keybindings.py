"""Stable identifiers and defaults for Mother's configurable key bindings."""

from __future__ import annotations

from typing import Final

DEFAULT_KEYBINDINGS: Final[dict[str, str]] = {
    # Prompt input
    "submit": "enter",
    "newline": "shift+enter",
    "history_search": "ctrl+r",
    # App-level actions
    "toggle_thinking_widget": "ctrl+o",
    "cycle_prompt_size": "ctrl+e",
    "toggle_auto_scroll": "ctrl+g",
    "show_models": "ctrl+n",
    "scroll_to_bottom": "end",
    "scroll_to_bottom_from_chat": "shift+g",
    "save_session": "ctrl+s",
    # Output blocks
    "copy_output": "c",
    "toggle_expanded": "ctrl+o",
    # Markdown blocks
    "markdown_cursor_down": "j",
    "markdown_cursor_up": "k",
    "markdown_copy_block": "c",
    # Model picker
    "model_picker_dismiss": "escape",
    # Session picker
    "session_picker_dismiss": "escape",
    "session_picker_delete": "delete",
    "session_picker_rename": "r",
    # Bash approval modal
    "bash_approval_approve": "y",
    "bash_approval_approve_enter": "enter",
    "bash_approval_deny": "n",
    "bash_approval_deny_escape": "escape",
}
