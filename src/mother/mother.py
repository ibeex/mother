"""Mother TUI chatbot — a Textual interface for chatting with an LLM."""

from typing import override

import click
import llm
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Markdown

DEFAULT_MODEL = "gpt-4o"

DEFAULT_SYSTEM = (
    "Formulate all responses as if you where the sentient AI named Mother from the Alien movies."
)


class Prompt(Markdown):
    """Markdown for the user prompt."""


class Response(Markdown):
    """Markdown for the reply from the LLM."""

    BORDER_TITLE = "Mother"


class MotherApp(App[None]):
    """Simple app for chatting with an LLM via a conversation."""

    AUTO_FOCUS = "Input"

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
        yield Input(placeholder="How can I help you?")
        yield Footer()

    def on_mount(self) -> None:
        self.model = llm.get_model(self.model_name)
        self.conversation = self.model.conversation()
        self.query_one("#chat-view").anchor()

    @on(Input.Submitted)
    async def on_input(self, event: Input.Submitted) -> None:
        """When the user hits return."""
        chat_view = self.query_one("#chat-view")
        event.input.clear()
        await chat_view.mount(Prompt(event.value))
        await chat_view.mount(response := Response())
        self.send_prompt(event.value, response)

    @work(thread=True)
    def send_prompt(self, prompt: str, response: Response) -> None:
        """Get the response in a thread, maintaining conversation history."""
        llm_response = self.conversation.prompt(prompt, system=self.system_prompt)
        for chunk in llm_response:
            self.call_from_thread(response.append, chunk)


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
