"""Tests for reasoning-effort helpers."""

from typing import ClassVar, cast, final

from llm.models import Model

from mother.reasoning import (
    build_reasoning_options,
    format_reasoning_effort,
    normalize_reasoning_effort,
    parse_reasoning_effort,
    supported_reasoning_efforts,
    supports_reasoning_effort,
)


@final
class _ReasoningOptions:
    model_fields: ClassVar[dict[str, object]] = {"reasoning_effort": object()}


@final
class _PlainOptions:
    model_fields: ClassVar[dict[str, object]] = {"temperature": object()}


@final
class _ReasoningModel:
    Options: ClassVar[type[_ReasoningOptions]] = _ReasoningOptions


@final
class _PlainModel:
    Options: ClassVar[type[_PlainOptions]] = _PlainOptions


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


def test_supports_reasoning_effort_checks_model_options() -> None:
    assert supports_reasoning_effort(cast(Model, cast(object, _ReasoningModel()))) is True
    assert supports_reasoning_effort(cast(Model, cast(object, _PlainModel()))) is False
    assert supports_reasoning_effort(None) is False


def test_supported_reasoning_efforts_returns_mother_supported_values() -> None:
    reasoning_model = cast(Model, cast(object, _ReasoningModel()))

    assert supported_reasoning_efforts(reasoning_model) == (
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
    )


def test_build_reasoning_options_only_for_reasoning_models() -> None:
    reasoning_model = cast(Model, cast(object, _ReasoningModel()))
    plain_model = cast(Model, cast(object, _PlainModel()))

    assert build_reasoning_options(reasoning_model, "medium") == {"reasoning_effort": "medium"}
    assert build_reasoning_options(reasoning_model, "xhigh") == {"reasoning_effort": "xhigh"}
    assert build_reasoning_options(reasoning_model, "auto") == {}
    assert build_reasoning_options(reasoning_model, "off") == {"reasoning_effort": "none"}
    assert build_reasoning_options(plain_model, "medium") == {}


def test_format_reasoning_effort_uses_user_facing_labels() -> None:
    assert format_reasoning_effort("auto") == "auto"
    assert format_reasoning_effort("none") == "off"
    assert format_reasoning_effort("high") == "high"
