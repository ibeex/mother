"""Model picker screen and command palette providers."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path, PurePath
from typing import ClassVar, Protocol, cast, override

from textual.app import ComposeResult
from textual.binding import BindingType
from textual.command import Hit, Provider
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option

from mother.config import MotherConfig
from mother.models import get_all_entries
from mother.picker_search import PickerSearchField, filter_picker_items

_CSS_DIR = Path(__file__).resolve().parent / "css"


class _MotherAppProto(Protocol):
    agent_mode: bool
    config: MotherConfig

    def action_toggle_agent_mode(self) -> None: ...
    def action_show_models(self) -> None: ...


def get_available_models() -> list[tuple[str, str]]:
    """Return available configured models for the picker."""
    return [(entry.id, f"{entry.id} — {entry.name}") for entry in get_all_entries()]


def filter_available_models(
    query: str,
    available_models: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Return available models matching a search query.

    Model id matches are preferred, followed by label matches. Exact, prefix,
    substring, and fuzzy subsequence matches are supported.
    """
    models = available_models or get_available_models()
    return filter_picker_items(
        models,
        query,
        lambda model: (
            PickerSearchField(model[0], primary=True),
            PickerSearchField(model[1]),
        ),
    )


class AgentModeProvider(Provider):
    """Command palette provider for toggling agent mode."""

    @override
    async def search(self, query: str) -> AsyncGenerator[Hit, None]:
        app = cast(_MotherAppProto, cast(object, self.app))
        matcher = self.matcher(query)
        label = "Agent: off" if app.agent_mode else "Agent: on"
        score = matcher.match(label)
        if score > 0 or not query:
            yield Hit(
                score or 1.0,
                matcher.highlight(label),
                app.action_toggle_agent_mode,
                help="Toggle agent mode (tool use)",
            )


class ModelProvider(Provider):
    """Command palette provider for opening the model picker."""

    @override
    async def search(self, query: str) -> AsyncGenerator[Hit, None]:
        app = cast(_MotherAppProto, cast(object, self.app))
        matcher = self.matcher(query)
        label = "Models"
        score = matcher.match(label)
        if score > 0 or not query:
            yield Hit(
                score or 1.0,
                matcher.highlight(label),
                app.action_show_models,
                help=f"Browse and switch models (current: {app.config.model})",
            )


class ModelPickerScreen(ModalScreen[str | None]):
    """Modal screen for searching and selecting available models."""

    CSS_PATH: ClassVar[str | PurePath | list[str | PurePath] | None] = (
        _CSS_DIR / "model_picker.tcss"
    )

    BINDINGS: ClassVar[list[BindingType]] = [("escape", "dismiss", "Close")]

    def __init__(self, current_model: str) -> None:
        super().__init__()
        self.current_model: str = current_model
        self._all_models: list[tuple[str, str]] = get_available_models()

    @override
    def compose(self) -> ComposeResult:
        with Container(id="model-picker"):
            with Vertical():
                yield Input(placeholder="Search models...", id="model-query")
                yield OptionList(id="model-options")

    def on_mount(self) -> None:
        self._refresh_options("")
        _ = self.query_one(Input).focus()

    def _refresh_options(self, query: str) -> None:
        option_list = self.query_one(OptionList)
        matching_models = filter_available_models(query, self._all_models)
        _ = option_list.clear_options()
        if not matching_models:
            _ = option_list.add_option(Option("No models found", disabled=True))
            option_list.highlighted = None
            return
        _ = option_list.add_options(
            Option(
                f"★ {label}" if model_id == self.current_model else label,
                id=model_id,
            )
            for model_id, label in matching_models
        )
        option_list.highlighted = 0

    def _select_highlighted_model(self) -> None:
        option_list = self.query_one(OptionList)
        if option_list.highlighted is None:
            return
        option = option_list.get_option_at_index(option_list.highlighted)
        if option.id is None:
            return
        _ = self.dismiss(option.id)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "model-query":
            self._refresh_options(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "model-query":
            self._select_highlighted_model()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "model-options" and event.option.id is not None:
            _ = self.dismiss(event.option.id)
