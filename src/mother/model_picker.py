"""Model picker screen and command palette providers."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path, PurePath
from typing import ClassVar, Protocol, cast, override

import llm
from textual.app import ComposeResult
from textual.binding import BindingType
from textual.command import Hit, Provider
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList
from textual.widgets.option_list import Option

from mother.config import MotherConfig

_CSS_DIR = Path(__file__).resolve().parent / "css"


class _MotherAppProto(Protocol):
    agent_mode: bool
    config: MotherConfig

    def action_toggle_agent_mode(self) -> None: ...
    def action_show_models(self) -> None: ...


def get_available_models() -> list[tuple[str, str]]:
    """Return available models for the picker.

    Prefer custom models configured via ``extra-openai-models.yaml`` (they have
    a custom ``api_base``). If none are configured, fall back to all models.
    """
    all_models = list(llm.get_models())
    preferred_models = [model for model in all_models if getattr(model, "api_base", None)]
    source_models = preferred_models or all_models

    seen: set[str] = set()
    available_models: list[tuple[str, str]] = []
    for model in source_models:
        model_id = model.model_id
        if model_id in seen:
            continue
        seen.add(model_id)
        model_name = getattr(model, "model_name", None)
        label = f"{model_id} — {model_name}" if model_name else model_id
        available_models.append((model_id, label))
    return available_models


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


class ModelSwitchConfirmScreen(ModalScreen[bool]):
    """Modal screen asking the user to confirm a context-resetting model switch."""

    CSS_PATH: ClassVar[str | PurePath | list[str | PurePath] | None] = (
        _CSS_DIR / "model_picker.tcss"
    )

    BINDINGS: ClassVar[list[BindingType]] = [
        ("escape", "cancel", "Cancel"),
        ("enter", "confirm_selection", "Select"),
    ]

    def __init__(self, target_model: str) -> None:
        super().__init__()
        self.target_model: str = target_model

    @override
    def compose(self) -> ComposeResult:
        with Container(id="model-switch-confirm"):
            with Vertical():
                yield Label(f"Switch to {self.target_model}?", id="model-switch-confirm-title")
                yield Label(
                    "This starts a fresh context. The new model won't know earlier messages.",
                    id="model-switch-confirm-message",
                )
                yield OptionList(
                    Option("Cancel", id="cancel"),
                    Option("Switch model", id="confirm"),
                    id="model-switch-confirm-options",
                )

    def on_mount(self) -> None:
        option_list = self.query_one(OptionList)
        option_list.highlighted = 0
        _ = option_list.focus()

    def action_cancel(self) -> None:
        _ = self.dismiss(False)

    def action_confirm_selection(self) -> None:
        option_list = self.query_one(OptionList)
        if option_list.highlighted is None:
            _ = self.dismiss(False)
            return
        option = option_list.get_option_at_index(option_list.highlighted)
        _ = self.dismiss(option.id == "confirm")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "model-switch-confirm-options":
            return
        _ = self.dismiss(event.option.id == "confirm")


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
        normalized_query = query.strip().lower()
        matching_models = [
            (model_id, label)
            for model_id, label in self._all_models
            if not normalized_query
            or normalized_query in model_id.lower()
            or normalized_query in label.lower()
        ]
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
