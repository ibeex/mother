"""Tests for streamed markdown rendering behavior."""

import asyncio
from typing import final, override
from unittest.mock import AsyncMock

from textual.app import App, ComposeResult
from textual.widgets.markdown import MarkdownFence

from mother.widgets import Response


@final
class _FailingMarkdownStream:
    def __init__(self) -> None:
        self.stop: AsyncMock = AsyncMock()

    async def write(self, _fragment: str) -> None:
        raise AssertionError("fenced fragments must be rendered via full markdown update")


class _MarkdownStreamingApp(App[None]):
    @override
    def compose(self) -> ComposeResult:
        yield Response("")


def test_response_streaming_reparses_fragments_that_contain_complete_fences() -> None:
    async def run() -> None:
        app = _MarkdownStreamingApp()

        async with app.run_test() as pilot:
            response = app.query_one(Response)

            await response.append_fragment("Before\n\n")
            await pilot.pause()
            assert response.raw_markdown == "Before\n\n"
            response._stream = _FailingMarkdownStream()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

            await response.append_fragment("```kusto\nprint 1\n```\n\nAfter\n")
            await pilot.pause()

            fence = response.query_one(MarkdownFence)
            assert fence.code == "print 1"
            assert response.raw_markdown.endswith("After\n")

    asyncio.run(run())


def test_response_streaming_preserves_fenced_code_blocks_across_fragments() -> None:
    async def run() -> None:
        app = _MarkdownStreamingApp()

        async with app.run_test() as pilot:
            response = app.query_one(Response)

            await response.append_fragment("```python\n")
            await pilot.pause()
            assert response.query_one(MarkdownFence).code == ""

            await response.append_fragment("print('hello')\n")
            await pilot.pause()
            assert response.query_one(MarkdownFence).code == "print('hello')"

            await response.append_fragment("```\n")
            await pilot.pause()
            assert response.query_one(MarkdownFence).code == "print('hello')"

            await response.stop_stream()
            await pilot.pause()
            assert response.query_one(MarkdownFence).code == "print('hello')"

    asyncio.run(run())
