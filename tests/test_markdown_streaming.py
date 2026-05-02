"""Tests for streamed markdown rendering behavior."""

import asyncio

from textual.app import App, ComposeResult
from textual.widgets.markdown import MarkdownFence

from mother.widgets import Response


class _MarkdownStreamingApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Response("")


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
