"""Tests for the pydantic-ai request flow in MotherApp."""

import asyncio
from collections.abc import Awaitable, Callable
from inspect import isawaitable
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import cast, final
from unittest.mock import patch

from pydantic_ai import Tool
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from textual.worker import Worker

from mother import MotherApp, MotherConfig
from mother.council import (
    CouncilAggregateRanking,
    CouncilCandidateResponse,
    CouncilPeerReview,
    CouncilProgressUpdate,
    CouncilResult,
)
from mother.interrupts import UserInterruptedError
from mother.models import ModelEntry
from mother.runtime import RuntimePartialRunError, RuntimeResponse, RuntimeToolEvent
from mother.stats import TurnUsage
from mother.tools.bash_capture import BashResult
from mother.widgets import PromptTextArea, Response, ShellOutput, ThinkingOutput


@final
class _FakeResponse:
    def __init__(self) -> None:
        self.updated_texts: list[str] = []
        self.reset_texts: list[str] = []
        self.active_classes: set[str] = set()
        self.raw_markdown = ""
        self.appended_fragments: list[str] = []
        self.replaced_texts: list[str] = []
        self.stop_stream_calls = 0

    def update(self, text: str) -> None:
        self.updated_texts.append(text)

    async def append_fragment(self, fragment: str) -> None:
        self.appended_fragments.append(fragment)
        self.raw_markdown += fragment
        self.updated_texts.append(self.raw_markdown)

    async def replace_markdown(self, markdown: str) -> None:
        self.replaced_texts.append(markdown)
        self.raw_markdown = markdown
        self.updated_texts.append(markdown)

    async def stop_stream(self) -> None:
        self.stop_stream_calls += 1

    def reset_state(self, text: str) -> None:
        self.reset_texts.append(text)
        self.raw_markdown = text

    def add_class(self, class_name: str) -> None:
        self.active_classes.add(class_name)

    def remove_class(self, class_name: str) -> None:
        self.active_classes.discard(class_name)

    def set_class(self, add: bool, class_name: str) -> None:
        if add:
            self.active_classes.add(class_name)
            return
        self.active_classes.discard(class_name)


@final
class _FakeThinkingOutput:
    def __init__(self) -> None:
        self.display = False
        self.updated_texts: list[str] = []
        self.started = 0
        self.finished = 0

    def start_streaming(self) -> None:
        self.started += 1

    def finish_streaming(self) -> None:
        self.finished += 1

    def set_text(self, text: str) -> None:
        self.updated_texts.append(text)
        self.display = bool(text.strip())


@final
class _FakeRuntime:
    def __init__(
        self,
        response: RuntimeResponse,
        updates: list[tuple[str, str]],
        tool_events: list[RuntimeToolEvent] | None = None,
    ) -> None:
        self.response = response
        self.updates = updates
        self.tool_events = tool_events or []
        self.calls: list[dict[str, object]] = []

    async def run_stream(
        self,
        *,
        prompt_text: str,
        system_prompt: str,
        message_history: list[object],
        attachments: list[Path],
        tools: list[object],
        model_settings: dict[str, object],
        tool_call_limit: int | None = None,
        allow_tool_fallback: bool = True,
        on_text_update: Callable[[str], None] | None = None,
        on_thinking_update: Callable[[str], None] | None = None,
        on_tool_event: Callable[[RuntimeToolEvent], None] | None = None,
    ) -> RuntimeResponse:
        self.calls.append(
            {
                "prompt_text": prompt_text,
                "system_prompt": system_prompt,
                "message_history": list(message_history),
                "attachments": attachments,
                "tools": tools,
                "model_settings": model_settings,
                "tool_call_limit": tool_call_limit,
                "allow_tool_fallback": allow_tool_fallback,
                "has_tool_callback": on_tool_event is not None,
            }
        )
        latest_text = ""
        latest_thinking = ""
        for event in self.tool_events:
            if on_tool_event is not None:
                on_tool_event(event)
        for text, thinking in self.updates:
            if on_thinking_update is not None and thinking != latest_thinking:
                latest_thinking = thinking
                on_thinking_update(thinking)
            if on_text_update is not None and text != latest_text:
                latest_text = text
                on_text_update(text)
        return self.response


@final
class _FailingRuntime:
    async def run_stream(self, **_: object) -> RuntimeResponse:
        raise RuntimeError("boom")


@final
class _InterruptedRuntime:
    async def run_stream(
        self,
        **kwargs: object,
    ) -> RuntimeResponse:
        on_text_update = cast(Callable[[str], None] | None, kwargs.get("on_text_update"))
        if on_text_update is not None:
            on_text_update("partial answer")
        raise UserInterruptedError()


@final
class _PartialHistoryRuntime:
    def __init__(self, partial_messages: list[ModelMessage]) -> None:
        self.partial_messages = partial_messages

    async def run_stream(self, **_: object) -> RuntimeResponse:
        raise RuntimePartialRunError(RuntimeError("boom"), self.partial_messages)


@final
class _FakeCouncilRunner:
    def __init__(self, result: CouncilResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def run(
        self,
        *,
        user_question: str,
        conversation_context: str = "",
        supplemental_context: str = "",
    ) -> CouncilResult:
        self.calls.append(
            {
                "user_question": user_question,
                "conversation_context": conversation_context,
                "supplemental_context": supplemental_context,
            }
        )
        return self.result


def _call_from_thread(callback: object, *args: object) -> object:
    result = cast(Callable[..., object], callback)(*args)
    if not isawaitable(result):
        return result

    async def _await_result(awaitable: Awaitable[object]) -> object:
        return await awaitable

    outcomes: Queue[tuple[bool, object]] = Queue(maxsize=1)

    def _runner() -> None:
        try:
            resolved_value: object = asyncio.run(_await_result(cast(Awaitable[object], result)))
        except BaseException as exc:  # pragma: no cover - failure path is asserted via re-raise
            outcomes.put((False, exc))
            return
        outcomes.put((True, resolved_value))

    thread = Thread(target=_runner)
    thread.start()
    thread.join()
    succeeded, value = outcomes.get()
    if succeeded:
        return value
    raise cast(BaseException, value)


def _make_app(
    *,
    reasoning: bool = False,
    agent_mode: bool = False,
    openai_reasoning_summary: str = "auto",
) -> MotherApp:
    app = MotherApp(
        config=MotherConfig(
            model="test-model",
            reasoning_effort="high",
            openai_reasoning_summary=openai_reasoning_summary,
        )
    )
    app.current_model_entry = ModelEntry(
        id="test-model",
        name="test-model",
        api_type="openai-responses",
        supports_reasoning=reasoning,
        supports_tools=True,
        supports_images=True,
    )
    app.agent_mode = agent_mode
    return app


def test_run_runtime_request_streams_thinking_and_updates_usage() -> None:
    app = _make_app(reasoning=True)
    app.conversation_state.message_history = [cast(ModelMessage, object())]
    response = _FakeResponse()
    thinking_output = _FakeThinkingOutput()
    fake_runtime = _FakeRuntime(
        RuntimeResponse(
            text="final answer",
            all_messages=[cast(ModelMessage, object())],
            usage=TurnUsage(
                request_tokens=123,
                response_tokens=45,
                cache_read_tokens=6,
                duration_seconds=1.5,
                model_id="test-model",
                provider="openai-responses",
            ),
            agent_mode_used=False,
        ),
        [
            ("", "step 1\n"),
            ("", "step 1\nstep 2"),
            ("final ", "step 1\nstep 2"),
            ("final answer", "step 1\nstep 2"),
        ],
    )

    with (
        patch("mother.mother.ChatRuntime", return_value=fake_runtime),
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
        patch.object(app, "notify"),
        patch.object(app, "_scroll_chat_to_end"),
    ):
        full_text = asyncio.run(
            app._run_runtime_request(  # pyright: ignore[reportPrivateUsage]
                "hello",
                cast(Response, cast(object, response)),
                "system",
                tools=[],
                attachments=[Path("/tmp/pasted.png")],
                thinking_output=cast(ThinkingOutput, cast(object, thinking_output)),
            )
        )

    assert full_text == "final answer"
    assert thinking_output.started == 1
    assert thinking_output.finished == 1
    assert thinking_output.updated_texts[-1] == "step 1\nstep 2"
    assert response.updated_texts == ["final ", "final answer"]
    assert response.appended_fragments == ["final ", "answer"]
    assert response.stop_stream_calls == 1
    assert response.reset_texts == ["final answer"]
    assert app._last_context_tokens == 123  # pyright: ignore[reportPrivateUsage]
    assert app._session_input_tokens == 123  # pyright: ignore[reportPrivateUsage]
    assert app._session_output_tokens == 45  # pyright: ignore[reportPrivateUsage]
    assert app._session_cached_tokens == 6  # pyright: ignore[reportPrivateUsage]
    assert app._last_response_time_seconds == 1.5  # pyright: ignore[reportPrivateUsage]
    assert len(app.conversation_state.message_history) == 1
    assert fake_runtime.calls == [
        {
            "prompt_text": "hello",
            "system_prompt": "system",
            "message_history": fake_runtime.calls[0]["message_history"],
            "attachments": [Path("/tmp/pasted.png")],
            "tools": [],
            "model_settings": {"openai_reasoning_effort": "high"},
            "tool_call_limit": None,
            "allow_tool_fallback": True,
            "has_tool_callback": True,
        }
    ]
    message_history = fake_runtime.calls[0]["message_history"]
    assert isinstance(message_history, list)
    history_list = cast(list[object], message_history)
    assert len(history_list) == 1


def test_response_output_replaces_markdown_when_streamed_text_is_rewritten() -> None:
    app = _make_app()
    response = _FakeResponse()

    with patch.object(app, "_should_follow_chat_updates", return_value=False):
        asyncio.run(
            app._update_response_output(  # pyright: ignore[reportPrivateUsage]
                cast(Response, cast(object, response)),
                "hello",
            )
        )
        asyncio.run(
            app._update_response_output(  # pyright: ignore[reportPrivateUsage]
                cast(Response, cast(object, response)),
                "hullo",
            )
        )

    assert response.appended_fragments == ["hello"]
    assert response.replaced_texts == ["hullo"]
    assert response.updated_texts == ["hello", "hullo"]


def test_reasoning_options_include_openai_reasoning_summary() -> None:
    app = _make_app(reasoning=True, openai_reasoning_summary="detailed")

    assert app._reasoning_options() == {  # pyright: ignore[reportPrivateUsage]
        "openai_reasoning_effort": "high",
        "openai_reasoning_summary": "detailed",
    }


def test_standard_agent_runtime_request_limits_tool_calls_to_one() -> None:
    app = _make_app(agent_mode=True)
    response = _FakeResponse()
    fake_runtime = _FakeRuntime(
        RuntimeResponse(
            text="done",
            all_messages=[cast(ModelMessage, object())],
            usage=TurnUsage(model_id="test-model", provider="openai-responses"),
            agent_mode_used=True,
        ),
        [("done", "")],
    )

    with (
        patch("mother.mother.ChatRuntime", return_value=fake_runtime),
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
    ):
        _ = asyncio.run(
            app._run_runtime_request(  # pyright: ignore[reportPrivateUsage]
                "hello",
                cast(Response, cast(object, response)),
                "system",
                tools=[Tool(lambda: "ok")],
                attachments=[],
                thinking_output=None,
            )
        )

    assert fake_runtime.calls[0]["tool_call_limit"] == 1


def test_deep_research_runtime_request_allows_multi_step_tool_calls() -> None:
    app = _make_app(agent_mode=True)
    app.agent_profile = "deep_research"
    response = _FakeResponse()
    fake_runtime = _FakeRuntime(
        RuntimeResponse(
            text="done",
            all_messages=[cast(ModelMessage, object())],
            usage=TurnUsage(model_id="test-model", provider="openai-responses"),
            agent_mode_used=True,
        ),
        [("done", "")],
    )

    with (
        patch("mother.mother.ChatRuntime", return_value=fake_runtime),
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
    ):
        _ = asyncio.run(
            app._run_runtime_request(  # pyright: ignore[reportPrivateUsage]
                "hello",
                cast(Response, cast(object, response)),
                "system",
                tools=[Tool(lambda: "ok")],
                attachments=[],
                thinking_output=None,
            )
        )

    assert fake_runtime.calls[0]["tool_call_limit"] is None


def test_run_runtime_request_disables_agent_mode_after_tool_fallback() -> None:
    app = _make_app(agent_mode=True)
    response = _FakeResponse()
    fake_runtime = _FakeRuntime(
        RuntimeResponse(
            text="fallback response",
            all_messages=[cast(ModelMessage, object())],
            usage=TurnUsage(model_id="test-model", provider="openai-responses"),
            agent_mode_used=False,
        ),
        [("fallback response", "")],
    )

    with (
        patch("mother.mother.ChatRuntime", return_value=fake_runtime),
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
        patch.object(app, "notify") as notify,
    ):
        full_text = asyncio.run(
            app._run_runtime_request(  # pyright: ignore[reportPrivateUsage]
                "hello",
                cast(Response, cast(object, response)),
                "system",
                tools=[Tool(lambda: "ok")],
                attachments=[],
                thinking_output=None,
            )
        )

    assert full_text == "fallback response"
    assert app.agent_mode is False
    notify.assert_called_with(
        "test-model does not support tools — agent mode disabled",
        title="Agent mode",
        severity="warning",
    )


def test_run_runtime_request_appends_clipboard_notice_for_blocked_bash_tool() -> None:
    app = _make_app(agent_mode=True)
    response = _FakeResponse()
    blocked_event = RuntimeToolEvent(
        phase="finished",
        tool_name="bash",
        tool_call_id="bash-1",
        arguments={"command": "python3 --version"},
        output=(
            "Warning: bash guard blocked this command. It was not executed.\n\n"
            "Command:\n```bash\npython3 --version\n```\n\n"
            "Guard model: test-guard\n"
            "The exact command has been copied to the clipboard."
        ),
    )
    fake_runtime = _FakeRuntime(
        RuntimeResponse(
            text="I tried, but the local safety guard blocked the Python command.",
            all_messages=[cast(ModelMessage, object())],
            usage=TurnUsage(model_id="test-model", provider="openai-responses"),
            agent_mode_used=True,
        ),
        [("I tried, but the local safety guard blocked the Python command.", "")],
        tool_events=[blocked_event],
    )

    with (
        patch("mother.mother.ChatRuntime", return_value=fake_runtime),
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
        patch.object(app, "_handle_runtime_tool_event") as handle_tool_event,
    ):
        full_text = asyncio.run(
            app._run_runtime_request(  # pyright: ignore[reportPrivateUsage]
                "hello",
                cast(Response, cast(object, response)),
                "system",
                tools=[Tool(lambda: "ok")],
                attachments=[],
                thinking_output=None,
            )
        )

    assert full_text is not None
    assert "already in your clipboard" in full_text
    assert "`!<command>`" in full_text
    assert "`!!<command>`" in full_text
    assert response.reset_texts == [full_text]
    handle_tool_event.assert_called_once_with(blocked_event)


def test_run_runtime_request_does_not_append_clipboard_notice_when_response_mentions_it() -> None:
    app = _make_app(agent_mode=True)
    response = _FakeResponse()
    blocked_event = RuntimeToolEvent(
        phase="finished",
        tool_name="bash",
        tool_call_id="bash-1",
        arguments={"command": "python3 --version"},
        output=(
            "Warning: bash guard blocked this command. It was not executed.\n\n"
            "Command:\n```bash\npython3 --version\n```\n\n"
            "Guard model: test-guard\n"
            "The exact command has been copied to the clipboard."
        ),
    )
    response_text = "The exact blocked command is already in your clipboard for review."
    fake_runtime = _FakeRuntime(
        RuntimeResponse(
            text=response_text,
            all_messages=[cast(ModelMessage, object())],
            usage=TurnUsage(model_id="test-model", provider="openai-responses"),
            agent_mode_used=True,
        ),
        [(response_text, "")],
        tool_events=[blocked_event],
    )

    with (
        patch("mother.mother.ChatRuntime", return_value=fake_runtime),
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
        patch.object(app, "_handle_runtime_tool_event"),
    ):
        full_text = asyncio.run(
            app._run_runtime_request(  # pyright: ignore[reportPrivateUsage]
                "hello",
                cast(Response, cast(object, response)),
                "system",
                tools=[Tool(lambda: "ok")],
                attachments=[],
                thinking_output=None,
            )
        )

    assert full_text == response_text
    assert response.reset_texts == [response_text]


def test_run_runtime_request_shows_error_when_runtime_fails() -> None:
    app = _make_app()
    response = _FakeResponse()

    with (
        patch("mother.mother.ChatRuntime", return_value=_FailingRuntime()),
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
    ):
        full_text = asyncio.run(
            app._run_runtime_request(  # pyright: ignore[reportPrivateUsage]
                "hello",
                cast(Response, cast(object, response)),
                "system",
                tools=[],
                attachments=[],
                thinking_output=None,
            )
        )

    assert full_text is None
    assert response.updated_texts == ["**Error:** boom"]
    assert response.stop_stream_calls == 1
    assert response.reset_texts == ["**Error:** boom"]


def test_run_runtime_request_preserves_partial_output_when_interrupted() -> None:
    app = _make_app()
    response = _FakeResponse()

    with (
        patch("mother.mother.ChatRuntime", return_value=_InterruptedRuntime()),
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
    ):
        full_text = asyncio.run(
            app._run_runtime_request(  # pyright: ignore[reportPrivateUsage]
                "hello",
                cast(Response, cast(object, response)),
                "system",
                tools=[],
                attachments=[],
                thinking_output=None,
            )
        )

    assert full_text is None
    assert response.updated_texts == ["partial answer", "partial answer\n\n_Interrupted by user._"]
    assert response.appended_fragments == ["partial answer", "\n\n_Interrupted by user._"]
    assert response.stop_stream_calls == 1
    assert response.reset_texts == ["partial answer\n\n_Interrupted by user._"]


def test_run_runtime_request_preserves_partial_history_on_failure() -> None:
    app = _make_app()
    response = _FakeResponse()
    partial_messages = [
        cast(
            ModelMessage,
            ModelRequest(
                parts=[ToolReturnPart(tool_name="bash", content="README.md", tool_call_id="call-1")]
            ),
        )
    ]

    with (
        patch("mother.mother.ChatRuntime", return_value=_PartialHistoryRuntime(partial_messages)),
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
    ):
        full_text = asyncio.run(
            app._run_runtime_request(  # pyright: ignore[reportPrivateUsage]
                "hello",
                cast(Response, cast(object, response)),
                "system",
                tools=[],
                attachments=[],
                thinking_output=None,
            )
        )

    assert full_text is None
    assert app.conversation_state.message_history == partial_messages
    assert response.updated_texts == ["**Error:** boom"]
    assert response.stop_stream_calls == 1
    assert response.reset_texts == ["**Error:** boom"]


def test_run_council_request_appends_only_synthesized_turn_to_main_context() -> None:
    app = _make_app()
    response = _FakeResponse()
    app.conversation_state.append_synthetic_turn("Earlier question", "Earlier answer")
    council_context = app._build_council_context()  # pyright: ignore[reportPrivateUsage]
    council_result = CouncilResult(
        final_text="Final council answer",
        judge_model_id="opus",
        stage1=(
            CouncilCandidateResponse(
                label="Response A",
                model_id="gpt-5",
                text="Draft answer A",
            ),
            CouncilCandidateResponse(
                label="Response B",
                model_id="opus",
                text="Draft answer B",
            ),
        ),
        stage2=(
            CouncilPeerReview(
                reviewer_model_id="g3",
                text="Response B is stronger.\n\nFINAL RANKING:\n1. Response B\n2. Response A",
                parsed_ranking=("Response B", "Response A"),
            ),
        ),
        aggregate_rankings=(
            CouncilAggregateRanking(label="Response B", average_rank=1.0, rankings_count=1),
        ),
        label_to_model={"Response A": "gpt-5", "Response B": "opus"},
    )
    fake_runner = _FakeCouncilRunner(council_result)

    with (
        patch("mother.mother.CouncilRunner", return_value=fake_runner),
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
    ):
        full_text = asyncio.run(
            app._run_council_request(  # pyright: ignore[reportPrivateUsage]
                user_question="Need a launch plan",
                response=cast(Response, cast(object, response)),
                conversation_context=council_context,
                supplemental_context="Shell command: git status",
                council_members=(
                    ModelEntry(
                        id="gpt-5",
                        name="gpt-5",
                        api_type="openai-responses",
                    ),
                    ModelEntry(
                        id="g3",
                        name="g3",
                        api_type="openai-responses",
                    ),
                    ModelEntry(
                        id="opus",
                        name="opus",
                        api_type="anthropic",
                    ),
                ),
                council_judge=ModelEntry(
                    id="opus",
                    name="opus",
                    api_type="anthropic",
                ),
            )
        )

    assert full_text == "Final council answer"
    assert response.updated_texts == ["Final council answer"]
    assert response.stop_stream_calls == 1
    assert response.reset_texts == ["Final council answer"]
    assert fake_runner.calls == [
        {
            "user_question": "Need a launch plan",
            "conversation_context": "User: Earlier question\n\nAssistant: Earlier answer",
            "supplemental_context": "Shell command: git status",
        }
    ]
    assert len(app.conversation_state.transcript_messages) == 4
    assert app.conversation_state.transcript_messages[-2].content == "Need a launch plan"
    assert app.conversation_state.transcript_messages[-1].content == "Final council answer"
    latest_request = cast(ModelRequest, app.conversation_state.message_history[-2])
    latest_response = cast(ModelResponse, app.conversation_state.message_history[-1])
    request_part = latest_request.parts[0]
    response_part = latest_response.parts[0]
    assert isinstance(request_part, UserPromptPart)
    assert request_part.content == "Need a launch plan"
    assert isinstance(response_part, TextPart)
    assert response_part.content == "Final council answer"


def test_run_council_request_updates_waiting_message_from_progress_callbacks() -> None:
    app = _make_app()
    response = _FakeResponse()
    council_result = CouncilResult(
        final_text="Final council answer",
        judge_model_id="judge",
        stage1=(),
        stage2=(),
        aggregate_rankings=(),
        label_to_model={},
    )
    progress_updates = [
        CouncilProgressUpdate.stage1(1, 3),
        CouncilProgressUpdate.stage2(2, 3),
        CouncilProgressUpdate.stage3(),
    ]

    class _ProgressRunner:
        def __init__(self, **kwargs: object) -> None:
            self.on_progress: Callable[[CouncilProgressUpdate], None] = cast(
                Callable[[CouncilProgressUpdate], None],
                kwargs["on_progress"],
            )

        async def run(
            self,
            *,
            user_question: str,
            conversation_context: str = "",
            supplemental_context: str = "",
        ) -> CouncilResult:
            _ = user_question
            _ = conversation_context
            _ = supplemental_context
            for update in progress_updates:
                self.on_progress(update)
            return council_result

    with (
        patch("mother.mother.CouncilRunner", _ProgressRunner),
        patch.object(app, "call_from_thread", side_effect=_call_from_thread),
        patch.object(app, "_set_response_waiting_message") as set_waiting_message,
        patch.object(app, "_show_council_trace"),
    ):
        full_text = asyncio.run(
            app._run_council_request(  # pyright: ignore[reportPrivateUsage]
                user_question="Need a launch plan",
                response=cast(Response, cast(object, response)),
                conversation_context="Context",
                supplemental_context="Shell command: git status",
                council_members=(
                    ModelEntry(
                        id="gpt-5",
                        name="gpt-5",
                        api_type="openai-responses",
                    ),
                ),
                council_judge=ModelEntry(
                    id="judge",
                    name="judge",
                    api_type="anthropic",
                ),
            )
        )

    assert full_text == "Final council answer"
    assert [call.args[1] for call in set_waiting_message.call_args_list] == [
        update.status_text() for update in progress_updates
    ]


def test_handle_interrupt_escape_requires_double_press() -> None:
    app = _make_app()

    assert app.handle_interrupt_escape() is False

    app._active_prompt_worker = cast(Worker[None], object())  # pyright: ignore[reportPrivateUsage]

    with (
        patch.object(app, "notify"),
        patch.object(app, "_interrupt_active_request") as interrupt_active_request,
    ):
        assert app.handle_interrupt_escape() is True
        interrupt_active_request.assert_not_called()
        assert app.handle_interrupt_escape() is True

    interrupt_active_request.assert_called_once_with()


def test_double_escape_interrupts_direct_shell_command() -> None:
    async def slow_execute_bash(_: str) -> BashResult:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError as exc:
            raise UserInterruptedError() from exc
        return BashResult(output="done", exit_code=0)

    async def run() -> None:
        app = _make_app()

        with patch("mother.mother.execute_bash", side_effect=slow_execute_bash):
            async with app.run_test() as pilot:
                text_area = app.query_one(PromptTextArea)
                text_area.load_text("!sleep 30")
                await pilot.pause()

                submission_task = asyncio.create_task(app.action_submit())
                await pilot.pause()

                shell_output = app.query_one(ShellOutput)
                assert text_area.read_only is True

                await pilot.press("escape")
                await pilot.pause()
                assert text_area.read_only is True

                await pilot.press("escape")
                for _ in range(10):
                    await pilot.pause()
                    if not text_area.read_only:
                        break

                await submission_task
                assert text_area.read_only is False
                assert "_Interrupted by user._" in shell_output.text

    asyncio.run(run())
