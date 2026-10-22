"""Tests that typed newlines and indentation survive Markdown rendering."""

import asyncio
from typing import final, override

from markdown_it import MarkdownIt
from textual.app import App, ComposeResult
from textual.widgets._markdown import MarkdownBulletList, MarkdownFence, MarkdownParagraph

from mother.widgets import (
    Prompt,
    Response,
    _markdown_with_preserved_layout,  # pyright: ignore[reportPrivateUsage]
)

_NBSP = "\u00a0"


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
    parser = _markdown_with_preserved_layout()
    tokens = parser.parse("line one\nline two\nline three\n")

    inline = next(token for token in tokens if token.type == "inline")
    child_types = [child.type for child in inline.children or []]

    assert child_types == ["text", "hardbreak", "text", "hardbreak", "text"]


def test_parser_preserves_leading_indentation() -> None:
    parser = _markdown_with_preserved_layout()
    tokens = parser.parse("heading\n   indented line\n    - item\n")

    inline = next(token for token in tokens if token.type == "inline")
    text_contents = [child.content for child in inline.children or [] if child.type == "text"]

    assert f"{_NBSP * 3}indented line" in text_contents
    assert f"{_NBSP * 4}- item" in text_contents


def test_parser_preserves_structural_indentation() -> None:
    examples = [
        "- parent\n  - child\n  - sibling\n- other",
        "1. parent\n   1. child\n   2. sibling\n2. other",
        "- first paragraph\n\n  second paragraph\n\n  ```python\n  x = 1\n  ```",
        "1. example\n\n   - nested\n\n     ```python\n     x = 1\n     ```",
        "  > quoted\n  > continuation\n\n  > second paragraph",
        "  # heading\n\n  ---",
        "  | one | two |\n  | --- | --- |\n  | a | b |",
        "[example][ref]\n\n  [ref]: https://example.com\n    'title'",
    ]
    for raw in examples:
        original = MarkdownIt("gfm-like").parse(raw)
        preserved = _markdown_with_preserved_layout().parse(raw)
        assert [(token.type, token.map, token.content) for token in preserved] == [
            (token.type, token.map, token.content) for token in original
        ]


def test_parser_registers_link_definitions_only_once() -> None:
    parser = _markdown_with_preserved_layout()
    environment: dict[str, object] = {}
    tokens = parser.parse("[example][ref]\n\n  [ref]: https://example.com", environment)
    inline = next(token for token in tokens if token.type == "inline")
    link = next(child for child in inline.children or [] if child.type == "link_open")

    assert link.attrGet("href") == "https://example.com"
    assert "duplicate_refs" not in environment


def test_parser_preserves_internal_whitespace_runs() -> None:
    parser = _markdown_with_preserved_layout()
    tokens = parser.parse("col1    col2   col3\n")

    inline = next(token for token in tokens if token.type == "inline")
    text = next(child.content for child in inline.children or [] if child.type == "text")

    assert text == "col1    col2   col3"


def test_parser_leaves_fenced_code_untouched() -> None:
    parser = _markdown_with_preserved_layout()
    tokens = parser.parse("```sh\n  echo one\n    echo two\n```\n")

    fence = next(token for token in tokens if token.type == "fence")

    assert fence.content == "  echo one\n    echo two\n"


def test_prompt_and_response_render_typed_newlines() -> None:
    async def run() -> None:
        app = _MarkdownApp("first line\nsecond line\nthird line")

        async with app.run_test() as pilot:
            await pilot.pause()
            paragraphs = _paragraph_texts(app)

        assert "first line\nsecond line\nthird line" in paragraphs

    asyncio.run(run())


def test_prompt_and_response_render_indentation() -> None:
    async def run() -> None:
        # Four-space bullets are paragraph text; up to three spaces starts a Markdown list.
        app = _MarkdownApp("output\n   package foo\n    - foo\n    -   spaced")

        async with app.run_test() as pilot:
            await pilot.pause()
            paragraphs = _paragraph_texts(app)

        expected = f"output\n{_NBSP * 3}package foo\n{_NBSP * 4}- foo\n{_NBSP * 4}-   spaced"
        assert expected in paragraphs

    asyncio.run(run())


def test_rendering_preserves_internal_whitespace_runs() -> None:
    async def run() -> None:
        raw = "col1    col2   col3\nnext    row"
        app = _MarkdownApp(raw)

        async with app.run_test() as pilot:
            await pilot.pause()
            paragraphs = _paragraph_texts(app)

        assert paragraphs == [raw, raw]

    asyncio.run(run())


def test_rendering_expands_leading_tabs_to_consistent_columns() -> None:
    async def run() -> None:
        app = _MarkdownApp("output\n\tvalue\n \tvalue\n        value\n\t value")

        async with app.run_test() as pilot:
            await pilot.pause()
            paragraphs = _paragraph_texts(app)

        expected = "output\n" + "\n".join(f"{_NBSP * width}value" for width in (8, 8, 8, 9))
        assert paragraphs == [expected, expected]

    asyncio.run(run())


def test_rendering_keeps_nested_lists() -> None:
    async def run() -> None:
        app = _MarkdownApp("- parent\n  - child\n  - sibling\n- other")

        async with app.run_test() as pilot:
            await pilot.pause()
            lists = list(app.query(MarkdownBulletList))
            paragraphs = _paragraph_texts(app)

        assert len(lists) == 4  # One outer and one nested list per widget.
        assert paragraphs == ["parent", "child", "sibling", "other"] * 2

    asyncio.run(run())


def test_rendering_preserves_fenced_code_verbatim() -> None:
    async def run() -> None:
        app = _MarkdownApp("```sh\n  echo one\n\techo two\n```")

        async with app.run_test() as pilot:
            await pilot.pause()
            fences = [str(fence.code) for fence in app.query(MarkdownFence)]

        assert fences == ["  echo one\n\techo two", "  echo one\n\techo two"]

    asyncio.run(run())


def test_rendering_newlines_does_not_change_stored_markdown() -> None:
    async def run() -> None:
        raw = "   first line\n     second line"
        app = _MarkdownApp(raw)

        async with app.run_test() as pilot:
            prompt = app.query_one(Prompt)
            response = app.query_one(Response)
            await pilot.pause()

        assert prompt.source == raw
        assert response.raw_markdown == raw

    asyncio.run(run())


def test_block_source_keeps_original_indentation() -> None:
    async def run() -> None:
        raw = "output\n   package foo\n    - foo"
        app = _MarkdownApp(raw)

        async with app.run_test() as pilot:
            await pilot.pause()
            response = app.query_one(Response)
            sources = [
                block.source
                for block in response._blocks()  # pyright: ignore[reportPrivateUsage]
            ]

        assert raw in sources

    asyncio.run(run())
