"""Mother TUI chatbot — a Textual interface for chatting with an LLM."""

from typing import override

import click
import llm
import pyperclip
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Markdown, TextArea
from textual.widgets.markdown import MarkdownBlock, MarkdownFence
from textual.worker import Worker, WorkerState

DEFAULT_MODEL = "4-claude"

DEFAULT_SYSTEM = (
    "Formulate all responses as if you where the sentient AI named Mother from the Alien movies."
)


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


class MotherApp(App[None]):
    """Simple app for chatting with an LLM via a conversation."""

    AUTO_FOCUS = "TextArea"

    BINDINGS = [("ctrl+enter", "submit", "Send")]

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
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, system: str = DEFAULT_SYSTEM) -> None:
        super().__init__()
        self.model_name = model_name
        self.system_prompt = system

    @override
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="chat-view"):
            yield Response("INTERFACE 2037 READY FOR INQUIRY")
        yield TextArea()
        yield Footer()

    def on_mount(self) -> None:
        self.model = llm.get_model(self.model_name)
        self.conversation = self.model.conversation()
        self.query_one("#chat-view").anchor()

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
        llm_response = self.conversation.prompt(prompt, system=self.system_prompt)
        full_text = ""
        for chunk in llm_response:
            full_text += chunk
            self.call_from_thread(response.update, full_text)
        response._raw = full_text
        response._cursor = 0


@click.command()
@click.option("--model", "-m", default=DEFAULT_MODEL, help="LLM model to use.")
@click.option("--system", "-s", default=DEFAULT_SYSTEM, help="System prompt.")
def cli(model: str, system: str) -> None:
    """Launch the Mother TUI chatbot."""
    app = MotherApp(model_name=model, system=system)
    app.run()


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
