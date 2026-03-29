"""Tests for status-line formatting and related app status text."""

from mother import MotherApp, MotherConfig
from mother.models import ModelEntry
from mother.widgets import StatusLine


def _reasoning_entry() -> ModelEntry:
    return ModelEntry(
        id="test-model",
        name="test-model",
        api_type="openai-responses",
        supports_reasoning=True,
        supports_tools=True,
        supports_images=True,
    )


def _anthropic_reasoning_entry() -> ModelEntry:
    return ModelEntry(
        id="claude",
        name="haiku",
        api_type="anthropic",
        supports_reasoning=True,
        supports_tools=True,
        supports_images=True,
    )


def test_subtitle_shows_agent_indicator() -> None:
    app = MotherApp()
    app.agent_mode = False
    app.config = MotherConfig(model="test-model")
    app._update_subtitle()  # pyright: ignore[reportPrivateUsage]
    assert "[AGENT]" not in app.sub_title

    app.agent_mode = True
    app._update_subtitle()  # pyright: ignore[reportPrivateUsage]
    assert "[AGENT]" in app.sub_title


def test_subtitle_shows_research_indicator() -> None:
    app = MotherApp()
    app.agent_mode = True
    app.agent_profile = "deep_research"
    app.config = MotherConfig(model="test-model")
    app._update_subtitle()  # pyright: ignore[reportPrivateUsage]
    assert "[RESEARCH]" in app.sub_title


def test_config_tools_enabled_sets_initial_mode() -> None:
    config = MotherConfig(tools_enabled=True)
    app = MotherApp(config=config)
    assert app.agent_mode is True


def test_statusline_formats_minimal_state() -> None:
    assert StatusLine.format_status("test-model", False, None) == "test-model · A:off · C:?"


def test_statusline_formats_full_state() -> None:
    assert (
        StatusLine.format_status(
            "test-model",
            True,
            12345,
            False,
            "high",
            1.25,
            12345,
            678,
            9000,
            "research",
        )
        == "test-model · A:research · C:12.3k · Tok:12.3k/678/9.0k · Man · R:high · 1.2s"
    )


def test_statusline_formats_response_time_variants() -> None:
    assert StatusLine.format_response_time(0.806) == "0.8s"
    assert StatusLine.format_response_time(61.0) == "1m 1s"
    assert StatusLine.format_response_time(3661.0) == "1h 1m 1s"


def test_status_reasoning_effort_visible_for_reasoning_models() -> None:
    app = MotherApp(config=MotherConfig(reasoning_effort="none"))
    app.current_model_entry = _reasoning_entry()
    assert app._status_reasoning_effort() == "off"  # pyright: ignore[reportPrivateUsage]


def test_status_reasoning_effort_shows_openai_summary_mode() -> None:
    app = MotherApp(
        config=MotherConfig(reasoning_effort="medium", openai_reasoning_summary="detailed")
    )
    app.current_model_entry = _reasoning_entry()
    assert app._status_reasoning_effort() == "medium/detailed"  # pyright: ignore[reportPrivateUsage]


def test_status_reasoning_effort_shows_anthropic_thinking_mode() -> None:
    app = MotherApp(config=MotherConfig(reasoning_effort="medium"))
    app.current_model_entry = _anthropic_reasoning_entry()
    assert app._status_reasoning_effort() == "medium/thinking"  # pyright: ignore[reportPrivateUsage]
