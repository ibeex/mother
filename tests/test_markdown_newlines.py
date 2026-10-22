"""Tests that typed newlines survive Markdown rendering in prompts and responses."""

import asyncio
from typing import final, override

from textual.app import App, ComposeResult
from textual.widgets._markdown import MarkdownParagraph

from mother.widgets import (
    Prompt,
    Response,
    _markdown_with_preserved_newlines,  # pyright: ignore[reportPrivateUsage]
)


@final
class _MarkdownApp(App[None]):
    def __init__(self, content: str) -> None:
        super().__init__()
        self._content = content

    @override
    def compose(self) -> ComposeResult:
        yield Prompt(self._content)
        yield Response(self._content)


def _paragraph_texts(app: App[None]) -> list[str]:
    return [
        paragraph._content.plain  # pyright: ignore[reportPrivateUsage]
        for paragraph in app.query(MarkdownParagraph)
    ]


def test_parser_converts_softbreaks_into_hardbreaks() -> None:
    parser = _markdown_with_preserved_newlines()
    tokens = parser.parse("line one\nline two\nline three\n")

    inline = next(token for token in tokens if token.type == "inline")
    child_types = [child.type for child in inline.children or []]

    assert child_types == ["text", "hardbreak", "text", "hardbreak", "text"]


def test_parser_leaves_fenced_code_untouched() -> None:
    parser = _markdown_with_preserved_newlines()
    tokens = parser.parse("```sh\necho one\necho two\n```\n")

    fence = next(token for token in tokens if token.type == "fence")

    assert fence.content == "echo one\necho two\n"


def test_prompt_and_response_render_typed_newlines() -> None:
    async def run() -> None:
        app = _MarkdownApp("first line\nsecond line\nthird line")

        async with app.run_test() as pilot:
            await pilot.pause()
            paragraphs = _paragraph_texts(app)

        assert "first line\nsecond line\nthird line" in paragraphs

    asyncio.run(run())


def test_rendering_newlines_does_not_change_stored_markdown() -> None:
    async def run() -> None:
        raw = "first line\nsecond line"
        app = _MarkdownApp(raw)

        async with app.run_test() as pilot:
            prompt = app.query_one(Prompt)
            response = app.query_one(Response)
            await pilot.pause()

        assert prompt.source == raw
        assert response.raw_markdown == raw

    asyncio.run(run())
