"""Mother TUI chatbot — a Textual interface for chatting with an LLM."""

from collections.abc import AsyncGenerator
from typing import override

import click
import llm
import pyperclip
from textual import work
from textual.app import App, ComposeResult
from textual.command import Hit, Provider
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Markdown, OptionList, TextArea
from textual.widgets.markdown import MarkdownBlock, MarkdownFence
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerState

from mother.config import MotherConfig, apply_cli_overrides, load_config
from mother.tools import get_default_tools


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


class Prompt(Markdown):
    """Markdown for the user prompt."""


class Response(Markdown):
    """Markdown for the reply from the LLM, with block-level copy support."""

    BORDER_TITLE = "Mother"
    BINDINGS = [
        ("j", "cursor_down", "Next block"),
        ("k", "cursor_up", "Prev block"),
        ("c", "copy_block", "Copy"),
    ]
    can_focus = True

    def __init__(self, markdown: str = "") -> None:
        super().__init__(markdown)
        self._cursor = 0
        self._raw = markdown

    def _blocks(self) -> list[MarkdownBlock]:
        return list(self.query(MarkdownBlock))

    def _refresh_highlight(self) -> None:
        for i, block in enumerate(self._blocks()):
            block.set_class(i == self._cursor, "highlight")

    def on_focus(self) -> None:
        self._refresh_highlight()

    def on_blur(self) -> None:
        for block in self._blocks():
            block.remove_class("highlight")

    def action_cursor_down(self) -> None:
        blocks = self._blocks()
        if self._cursor < len(blocks) - 1:
            self._cursor += 1
            self._refresh_highlight()

    def action_cursor_up(self) -> None:
        if self._cursor > 0:
            self._cursor -= 1
            self._refresh_highlight()

    def _block_text(self, block: MarkdownBlock) -> str:
        if isinstance(block, MarkdownFence):
            return block.code
        return block.source or self._raw

    def action_copy_block(self) -> None:
        text = self.screen.get_selected_text()
        if not text:
            blocks = self._blocks()
            text = self._block_text(blocks[self._cursor]) if blocks else self._raw
        try:
            pyperclip.copy(text)
        except Exception:
            self.app.copy_to_clipboard(text)
        self.notify("Copied!")


class ModelProvider(Provider):
    """Command palette provider for opening the model picker."""

    async def search(self, query: str) -> AsyncGenerator[Hit, None]:
        assert isinstance(self.app, MotherApp)
        app: MotherApp = self.app
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


class ModelPickerScreen(ModalScreen[None]):
    """Modal screen for searching and selecting available models."""

    CSS = """
    ModelPickerScreen {
        align: center middle;
        background: $background 60%;
    }

    #model-picker {
        width: 72;
        height: 24;
        border: round $primary;
        background: $surface;
        padding: 1;
    }

    #model-query {
        margin-bottom: 1;
    }

    #model-options {
        height: 1fr;
    }
    """

    BINDINGS = [("escape", "dismiss", "Close")]

    def __init__(self, current_model: str) -> None:
        super().__init__()
        self.current_model = current_model
        self._all_models = get_available_models()

    @override
    def compose(self) -> ComposeResult:
        with Container(id="model-picker"):
            with Vertical():
                yield Input(placeholder="Search models...", id="model-query")
                yield OptionList(id="model-options")

    def on_mount(self) -> None:
        self._refresh_options("")
        self.query_one(Input).focus()

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
        option_list.clear_options()
        if not matching_models:
            option_list.add_option(Option("No models found", disabled=True))
            option_list.highlighted = None
            return
        option_list.add_options(
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
        assert isinstance(self.app, MotherApp)
        app: MotherApp = self.app
        app.action_switch_model(option.id)
        self.dismiss()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "model-query":
            self._refresh_options(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "model-query":
            self._select_highlighted_model()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "model-options" and event.option.id is not None:
            assert isinstance(self.app, MotherApp)
            app: MotherApp = self.app
            app.action_switch_model(event.option.id)
            self.dismiss()


class MotherApp(App[None]):
    """Simple app for chatting with an LLM via a conversation."""

    AUTO_FOCUS = "TextArea"

    BINDINGS = [("ctrl+enter", "submit", "Send")]

    COMMANDS = App.COMMANDS | {ModelProvider}

    CSS = """
    Prompt {
        background: $primary 10%;
        color: $text;
        margin: 1;
        margin-right: 8;
        padding: 1 2 0 2;
    }

    Response {
        border: wide $success;
        background: $success 10%;
        color: $text;
        margin: 1;
        margin-left: 8;
        padding: 1 2 0 2;
    }

    Response:focus {
        border: wide $warning;
    }

    MarkdownBlock.highlight {
        background: $accent 20%;
    }

    TextArea {
        height: 6;
        border: tall $primary;
    }

    #model-picker {
        width: 72;
        height: 24;
    }
    """

    def __init__(
        self,
        config: MotherConfig | None = None,
        model_name: str | None = None,
        system: str | None = None,
    ) -> None:
        super().__init__()
        base = config or MotherConfig()
        self.config = apply_cli_overrides(base, model_name, system)

    @override
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="chat-view"):
            yield Response("INTERFACE 2037 READY FOR INQUIRY")
        yield TextArea()
        yield Footer()

    def on_mount(self) -> None:
        self.model = llm.get_model(self.config.model)
        self.conversation = self.model.conversation()
        self.query_one("#chat-view").anchor()
        self.sub_title = self.config.model

    def action_show_models(self) -> None:
        """Open the model picker."""
        self.push_screen(ModelPickerScreen(self.config.model))

    def action_switch_model(self, model_id: str) -> None:
        """Switch to a different LLM model and start a fresh conversation."""
        self.config = MotherConfig(
            model=model_id,
            system_prompt=self.config.system_prompt,
            tools_enabled=self.config.tools_enabled,
        )
        self.model = llm.get_model(model_id)
        self.conversation = self.model.conversation()
        self.sub_title = model_id
        self.notify(f"Switched to {model_id}", title="Model changed")

    async def action_submit(self) -> None:
        """When the user hits Ctrl+Enter."""
        text_area = self.query_one(TextArea)
        value = text_area.text.strip()
        if not value:
            return
        chat_view = self.query_one("#chat-view")
        text_area.read_only = True
        text_area.clear()
        await chat_view.mount(Prompt(value))
        await chat_view.mount(response := Response())
        self.send_prompt(value, response)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Re-enable input when the worker finishes."""
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            text_area = self.query_one(TextArea)
            text_area.read_only = False
            text_area.focus()

    @work(thread=True)
    def send_prompt(self, prompt: str, response: Response) -> None:
        """Get the response in a thread, maintaining conversation history."""
        self.call_from_thread(response.update, "*Thinking...*")
        tool_registry = get_default_tools()
        kwargs: dict = {"system": self.config.system_prompt}
        if not tool_registry.is_empty():
            kwargs["tools"] = tool_registry.tools()
        try:
            llm_response = self.conversation.prompt(prompt, **kwargs)
            full_text = ""
            for chunk in llm_response:
                full_text += chunk
                self.call_from_thread(response.update, full_text)
        except Exception as exc:
            error_text = f"**Error:** {exc}"
            self.call_from_thread(response.update, error_text)
            response._raw = error_text
            response._cursor = 0
            return
        response._raw = full_text
        response._cursor = 0


@click.command()
@click.option("--model", "-m", default=None, help="LLM model to use.")
@click.option("--system", "-s", default=None, help="System prompt.")
def cli(model: str | None, system: str | None) -> None:
    """Launch the Mother TUI chatbot."""
    config = load_config()
    config = apply_cli_overrides(config, model, system)
    app = MotherApp(config=config)
    app.run()
