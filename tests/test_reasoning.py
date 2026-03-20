"""Tests for reasoning-effort helpers."""

from mother.models import ModelEntry
from mother.reasoning import (
    build_reasoning_options,
    format_reasoning_effort,
    normalize_reasoning_effort,
    parse_reasoning_effort,
    supported_reasoning_efforts,
    supports_reasoning_effort,
)

_REASONING_MODEL = ModelEntry(
    id="gpt-5",
    name="gpt-5.4",
    api_type="openai-responses",
    supports_reasoning=True,
)

_ANTHROPIC_REASONING_MODEL = ModelEntry(
    id="claude",
    name="sonnet",
    api_type="anthropic",
    supports_reasoning=True,
)

_PLAIN_MODEL = ModelEntry(
    id="plain",
    name="plain",
    api_type="openai-chat",
    supports_reasoning=False,
)


def test_normalize_reasoning_effort_accepts_aliases() -> None:
    assert normalize_reasoning_effort("off") == "none"
    assert normalize_reasoning_effort("default") == "auto"
    assert normalize_reasoning_effort("MEDIUM") == "medium"


def test_parse_reasoning_effort_rejects_unknown_values() -> None:
    for value in ("turbo", "minimal"):
        try:
            _ = parse_reasoning_effort(value)
        except ValueError as exc:
            assert "reasoning_effort" in str(exc)
        else:
            raise AssertionError("Expected invalid reasoning effort to raise ValueError")


def test_supports_reasoning_effort_checks_model_capability() -> None:
    assert supports_reasoning_effort(_REASONING_MODEL) is True
    assert supports_reasoning_effort(_PLAIN_MODEL) is False
    assert supports_reasoning_effort(None) is False


def test_supported_reasoning_efforts_returns_mother_supported_values() -> None:
    assert supported_reasoning_efforts(_REASONING_MODEL) == (
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
    )


def test_build_reasoning_options_for_openai_models() -> None:
    assert build_reasoning_options(_REASONING_MODEL, "medium") == {
        "openai_reasoning_effort": "medium"
    }
    assert build_reasoning_options(_REASONING_MODEL, "xhigh") == {
        "openai_reasoning_effort": "xhigh"
    }
    assert build_reasoning_options(_REASONING_MODEL, "auto") == {}
    assert build_reasoning_options(_REASONING_MODEL, "off") == {"openai_reasoning_effort": "none"}
    assert build_reasoning_options(_PLAIN_MODEL, "medium") == {}


def test_build_reasoning_options_for_anthropic_models() -> None:
    assert build_reasoning_options(_ANTHROPIC_REASONING_MODEL, "high") == {
        "anthropic_effort": "high"
    }
    assert build_reasoning_options(_ANTHROPIC_REASONING_MODEL, "xhigh") == {
        "anthropic_effort": "max"
    }
    assert build_reasoning_options(_ANTHROPIC_REASONING_MODEL, "off") == {}


def test_format_reasoning_effort_uses_user_facing_labels() -> None:
    assert format_reasoning_effort("auto") == "auto"
    assert format_reasoning_effort("none") == "off"
    assert format_reasoning_effort("high") == "high"
