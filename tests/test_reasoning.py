"""Tests for reasoning-effort helpers."""

from mother.models import ModelEntry
from mother.reasoning import (
    build_reasoning_options,
    format_reasoning_effort,
    normalize_openai_reasoning_summary,
    normalize_reasoning_effort,
    parse_openai_reasoning_summary,
    parse_reasoning_effort,
    supported_reasoning_efforts,
    supports_openai_reasoning_summary,
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


def test_normalize_openai_reasoning_summary_accepts_supported_values() -> None:
    assert normalize_openai_reasoning_summary("AUTO") == "auto"
    assert normalize_openai_reasoning_summary("detailed") == "detailed"


def test_parse_openai_reasoning_summary_rejects_unknown_values() -> None:
    try:
        _ = parse_openai_reasoning_summary("verbose")
    except ValueError as exc:
        assert "openai_reasoning_summary" in str(exc)
    else:
        raise AssertionError("Expected invalid OpenAI reasoning summary to raise ValueError")


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


def test_supports_openai_reasoning_summary_checks_model_capability() -> None:
    assert supports_openai_reasoning_summary(_REASONING_MODEL) is True
    assert supports_openai_reasoning_summary(_ANTHROPIC_REASONING_MODEL) is False
    assert supports_openai_reasoning_summary(_PLAIN_MODEL) is False
    assert supports_openai_reasoning_summary(None) is False


def test_build_reasoning_options_for_openai_models() -> None:
    assert build_reasoning_options(_REASONING_MODEL, "medium") == {
        "openai_reasoning_effort": "medium"
    }
    assert build_reasoning_options(_REASONING_MODEL, "xhigh") == {
        "openai_reasoning_effort": "xhigh"
    }
    assert build_reasoning_options(_REASONING_MODEL, "auto") == {}
    assert build_reasoning_options(_REASONING_MODEL, "off") == {"openai_reasoning_effort": "none"}
    assert build_reasoning_options(_REASONING_MODEL, "auto", "detailed") == {
        "openai_reasoning_summary": "detailed"
    }
    assert build_reasoning_options(_REASONING_MODEL, "medium", "concise") == {
        "openai_reasoning_effort": "medium",
        "openai_reasoning_summary": "concise",
    }
    assert build_reasoning_options(_PLAIN_MODEL, "medium", "detailed") == {}


def test_build_reasoning_options_for_anthropic_models() -> None:
    assert build_reasoning_options(_ANTHROPIC_REASONING_MODEL, "low") == {
        "anthropic_thinking": {"type": "enabled", "budget_tokens": 1024}
    }
    assert build_reasoning_options(_ANTHROPIC_REASONING_MODEL, "medium") == {
        "anthropic_thinking": {"type": "enabled", "budget_tokens": 2048}
    }
    assert build_reasoning_options(_ANTHROPIC_REASONING_MODEL, "high") == {
        "anthropic_thinking": {"type": "enabled", "budget_tokens": 3072}
    }
    assert build_reasoning_options(_ANTHROPIC_REASONING_MODEL, "xhigh") == {
        "anthropic_thinking": {"type": "enabled", "budget_tokens": 3584}
    }
    assert build_reasoning_options(_ANTHROPIC_REASONING_MODEL, "auto") == {}
    assert build_reasoning_options(_ANTHROPIC_REASONING_MODEL, "off") == {}
    assert build_reasoning_options(_ANTHROPIC_REASONING_MODEL, "high", "detailed") == {
        "anthropic_thinking": {"type": "enabled", "budget_tokens": 3072}
    }


def test_format_reasoning_effort_uses_user_facing_labels() -> None:
    assert format_reasoning_effort("auto") == "auto"
    assert format_reasoning_effort("none") == "off"
    assert format_reasoning_effort("high") == "high"
