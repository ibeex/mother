"""Searchable per-directory session picker."""

from __future__ import annotations

from pathlib import Path, PurePath
from typing import ClassVar, override

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option

from mother.picker_search import PickerSearchField, filter_picker_items
from mother.session import SessionManager


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
        self.sessions: list[SessionManager] = SessionManager.list_sessions()
        self.matches: list[SessionManager] = self.sessions
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
            lambda s: (
                PickerSearchField(s.name or s.path.stem, primary=True),
                PickerSearchField(str(s.path)),
            ),
        )
        options = self.query_one(OptionList)
        _ = options.clear_options()
        _ = options.add_options(
            Option(f"{s.name or s.path.stem} — {s.header.get('model', '')}", id=str(i))
            for i, s in enumerate(self.matches)
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
            _ = self.dismiss(self.matches[options.highlighted])

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is not None:
            _ = self.dismiss(self.matches[int(str(event.option.id))])

    def action_rename(self) -> None:
        options = self.query_one(OptionList)
        if options.highlighted is None:
            return
        self.rename_target = self.matches[options.highlighted]
        input_widget = self.query_one(Input)
        input_widget.value = self.rename_target.name or ""
        input_widget.placeholder = "Enter session name, then press Enter..."
        _ = input_widget.focus()

    def action_delete(self) -> None:
        options = self.query_one(OptionList)
        if options.highlighted is None:
            return
        selected = self.matches[options.highlighted]
        selected.delete()
        self.sessions.remove(selected)
        self.refresh_sessions(self.query_one(Input).value)
