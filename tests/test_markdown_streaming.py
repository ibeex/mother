"""Tests for streamed markdown rendering behavior."""

import asyncio
from typing import final, override
from unittest.mock import AsyncMock, patch

from textual.app import App, ComposeResult
from textual.await_complete import AwaitComplete
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


@final
class _RecordingResponse(Response):
    """Response test double that records rendered markdown with configurable delay."""

    def __init__(self) -> None:
        super().__init__("")
        self.rendered_markdown: list[str] = []

    @override
    def update(self, markdown: str) -> AwaitComplete:
        async def record() -> None:
            if markdown == "```bash\n":
                await asyncio.sleep(0.02)
            self.rendered_markdown.append(markdown)

        return AwaitComplete(record())


def test_response_streaming_serializes_fenced_code_reparses() -> None:
    async def run() -> None:
        response = _RecordingResponse()

        first_update = asyncio.create_task(response.append_fragment("```bash\n"))
        await asyncio.sleep(0)
        final_update = asyncio.create_task(response.append_fragment("echo ok\n```\n"))
        _ = await asyncio.gather(first_update, final_update)

        assert response.raw_markdown == "```bash\necho ok\n```\n"
        assert response.rendered_markdown[-1] == response.raw_markdown

    asyncio.run(run())


def test_response_streaming_ignores_stale_prefix_snapshots() -> None:
    async def run() -> None:
        response = _RecordingResponse()
        final_markdown = "```bash\necho ok\n```\n"

        await response.update_streamed_markdown(final_markdown)
        await response.update_streamed_markdown("```bash\n")

        assert response.raw_markdown == final_markdown
        assert response.rendered_markdown[-1] == final_markdown

    asyncio.run(run())


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


def test_response_streaming_keeps_text_after_closed_fence_out_of_code_block() -> None:
    async def run() -> None:
        app = _MarkdownStreamingApp()

        async with app.run_test() as pilot:
            response = app.query_one(Response)

            await response.append_fragment("```bash\n")
            await response.append_fragment("echo ok\n")
            await response.append_fragment("```\n")
            await response.append_fragment("\nAfter\n")
            await pilot.pause()

            assert response.query_one(MarkdownFence).code == "echo ok"
            assert response.raw_markdown.endswith("After\n")

    asyncio.run(run())


def test_response_streaming_throttles_rapid_fenced_reparses_until_flush() -> None:
    async def run() -> None:
        response = _RecordingResponse()
        clock_values = iter((0.0, 0.01, 0.02, 0.20))

        large_line = "x" * 2_100 + "\n"

        with patch("mother.widgets.perf_counter", side_effect=lambda: next(clock_values)):
            await response.append_fragment("```python\n")
            await response.append_fragment(large_line)
            await response.append_fragment(large_line)

            assert response.rendered_markdown == ["```python\n"]
            assert response.raw_markdown == f"```python\n{large_line}{large_line}"

            await response.stop_stream()

        assert response.rendered_markdown[-1] == response.raw_markdown

    asyncio.run(run())


def test_response_stop_stream_forces_final_full_render_for_fenced_markdown() -> None:
    async def run() -> None:
        response = _RecordingResponse()

        await response.append_fragment("```python\n")
        await response.append_fragment("print('hello')\n")
        await response.append_fragment("```\n")
        render_count_before_stop = len(response.rendered_markdown)

        await response.stop_stream()

        assert len(response.rendered_markdown) == render_count_before_stop + 1
        assert response.rendered_markdown[-1] == response.raw_markdown

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
