"""Searchable per-directory session picker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePath
from typing import ClassVar, override

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option

from mother.picker_search import PickerSearchField, filter_picker_items
from mother.session import SessionEntry, SessionManager

_PREVIEW_LENGTH = 100


@dataclass(frozen=True, slots=True)
class SessionPickerEntry:
    """Display and search data for one persisted session."""

    session: SessionManager
    title: str
    preview: str
    search_text: str
    message_count: int

    @property
    def label(self) -> str:
        """Return the two-line label shown in the picker."""
        count_label = f"{self.message_count} message{'s' if self.message_count != 1 else ''}"
        model = self.session.header.get("model") or "unknown model"
        name = f" · {self.session.name}" if self.session.name else ""
        return f"{self.title} · {model} · {count_label}{name}\n{self.preview}"


def build_session_picker_entry(session: SessionManager) -> SessionPickerEntry:
    """Build a compact, human-readable picker entry from a session transcript."""
    entries = session.load_entries()
    return SessionPickerEntry(
        session=session,
        title=_format_session_date(session),
        preview=_first_user_prompt(entries),
        search_text=_transcript_text(entries),
        message_count=sum(1 for entry in entries if entry["type"] == "message"),
    )


def _format_session_date(session: SessionManager) -> str:
    created = session.header.get("created")
    if isinstance(created, str):
        try:
            return datetime.fromisoformat(created).astimezone().strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass
    return session.path.stem


def _first_user_prompt(entries: list[SessionEntry]) -> str:
    for entry in entries:
        if entry["type"] == "prompt" and entry["user_text"].strip():
            return _truncate_preview(entry["user_text"])
    for entry in entries:
        if entry["type"] == "message" and entry["role"] == "user" and entry["content"].strip():
            return _truncate_preview(entry["content"])
    return "No prompt recorded"


def _transcript_text(entries: list[SessionEntry]) -> str:
    """Return all persisted text that should be discoverable in picker search."""
    return "\n".join(entry["content"] for entry in entries if entry["type"] == "message")


def _truncate_preview(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= _PREVIEW_LENGTH:
        return normalized
    return f"{normalized[: _PREVIEW_LENGTH - 1].rstrip()}…"


class SessionPickerScreen(ModalScreen[SessionManager | None]):
    CSS_PATH: ClassVar[str | PurePath | list[str | PurePath] | None] = (
        Path(__file__).resolve().parent / "css" / "model_picker.tcss"
    )
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "dismiss", "Close"),
        ("delete", "delete", "Delete"),
        ("r", "rename", "Rename"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.sessions: list[SessionPickerEntry] = [
            build_session_picker_entry(session) for session in SessionManager.list_sessions()
        ]
        self.matches: list[SessionPickerEntry] = self.sessions
        self.rename_target: SessionManager | None = None

    @override
    def compose(self) -> ComposeResult:
        with Container(id="model-picker"):
            with Vertical():
                yield Input(placeholder="Search sessions...", id="model-query")
                yield OptionList(id="model-options")

    def on_mount(self) -> None:
        self.refresh_sessions("")
        _ = self.query_one(Input).focus()

    def refresh_sessions(self, query: str) -> None:
        self.matches = filter_picker_items(
            self.sessions,
            query,
            lambda entry: (
                PickerSearchField(entry.session.name or entry.preview, primary=True),
                PickerSearchField(entry.preview),
                PickerSearchField(entry.search_text),
                PickerSearchField(entry.title),
                PickerSearchField(str(entry.session.path)),
            ),
        )
        options = self.query_one(OptionList)
        _ = options.clear_options()
        _ = options.add_options(
            Option(Text(entry.label), id=str(i)) for i, entry in enumerate(self.matches)
        )
        options.highlighted = 0 if self.matches else None

    def on_input_changed(self, event: Input.Changed) -> None:
        self.refresh_sessions(event.value)

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        input_widget = self.query_one(Input)
        if self.rename_target is not None:
            try:
                self.rename_target.set_name(input_widget.value)
            except ValueError:
                return
            self.rename_target = None
            input_widget.value = ""
            input_widget.placeholder = "Search sessions..."
            self.refresh_sessions("")
            return
        options = self.query_one(OptionList)
        if options.highlighted is not None:
            _ = self.dismiss(self.matches[options.highlighted].session)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is not None:
            _ = self.dismiss(self.matches[int(str(event.option.id))].session)

    def action_rename(self) -> None:
        options = self.query_one(OptionList)
        if options.highlighted is None:
            return
        self.rename_target = self.matches[options.highlighted].session
        input_widget = self.query_one(Input)
        input_widget.value = self.rename_target.name or ""
        input_widget.placeholder = "Enter session name, then press Enter..."
        _ = input_widget.focus()

    def action_delete(self) -> None:
        options = self.query_one(OptionList)
        if options.highlighted is None:
            return
        selected = self.matches[options.highlighted]
        selected.session.delete()
        self.sessions.remove(selected)
        self.refresh_sessions(self.query_one(Input).value)
