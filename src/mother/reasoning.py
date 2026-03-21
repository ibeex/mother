"""Helpers for reasoning-capable models."""

from __future__ import annotations

from typing import Final

from mother.models import ModelEntry

DEFAULT_REASONING_EFFORT: Final[str] = "medium"
_SUPPORTED_REASONING_EFFORTS: Final[tuple[str, ...]] = (
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
)
REASONING_EFFORT_CHOICES: Final[tuple[str, ...]] = ("auto", *_SUPPORTED_REASONING_EFFORTS)
_REASONING_EFFORT_ALIASES: Final[dict[str, str]] = {
    "default": "auto",
    "off": "none",
}
REASONING_EFFORT_HELP: Final[str] = "auto|off|low|medium|high|xhigh"

DEFAULT_OPENAI_REASONING_SUMMARY: Final[str] = "auto"
OPENAI_REASONING_SUMMARY_CHOICES: Final[tuple[str, ...]] = (
    "auto",
    "concise",
    "detailed",
)
OPENAI_REASONING_SUMMARY_HELP: Final[str] = "auto|concise|detailed"


def normalize_reasoning_effort(value: str) -> str | None:
    """Normalize a user/config reasoning value to a supported canonical value."""
    normalized = value.strip().lower()
    if not normalized:
        return None
    normalized = _REASONING_EFFORT_ALIASES.get(normalized, normalized)
    if normalized not in REASONING_EFFORT_CHOICES:
        return None
    return normalized


def parse_reasoning_effort(value: str) -> str:
    """Validate and normalize a reasoning setting, raising on invalid input."""
    normalized = normalize_reasoning_effort(value)
    if normalized is None:
        choices = ", ".join(REASONING_EFFORT_CHOICES)
        raise ValueError(f"Invalid reasoning_effort {value!r}. Expected one of: {choices}")
    return normalized


def normalize_openai_reasoning_summary(value: str) -> str | None:
    """Normalize a user/config OpenAI reasoning-summary value."""
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized not in OPENAI_REASONING_SUMMARY_CHOICES:
        return None
    return normalized


def parse_openai_reasoning_summary(value: str) -> str:
    """Validate and normalize an OpenAI reasoning-summary setting."""
    normalized = normalize_openai_reasoning_summary(value)
    if normalized is None:
        choices = ", ".join(OPENAI_REASONING_SUMMARY_CHOICES)
        raise ValueError(f"Invalid openai_reasoning_summary {value!r}. Expected one of: {choices}")
    return normalized


def supports_reasoning_effort(model: ModelEntry | None) -> bool:
    """Return whether the configured model supports reasoning control."""
    return model is not None and model.supports_reasoning


def supports_openai_reasoning_summary(model: ModelEntry | None) -> bool:
    """Return whether the configured model supports OpenAI reasoning summaries."""
    return (
        supports_reasoning_effort(model)
        and model is not None
        and model.api_type == "openai-responses"
    )


def supported_reasoning_efforts(model: ModelEntry | None) -> tuple[str, ...]:
    """Return the reasoning-effort values Mother supports for reasoning models."""
    if not supports_reasoning_effort(model):
        return ()
    return _SUPPORTED_REASONING_EFFORTS


def build_reasoning_options(
    model: ModelEntry | None,
    reasoning_effort: str,
    openai_reasoning_summary: str = DEFAULT_OPENAI_REASONING_SUMMARY,
) -> dict[str, object]:
    """Return provider-specific request settings for reasoning-capable models."""
    if not supports_reasoning_effort(model):
        return {}

    normalized_effort = normalize_reasoning_effort(reasoning_effort)
    normalized_summary = normalize_openai_reasoning_summary(openai_reasoning_summary)
    options: dict[str, object] = {}

    if model is not None and model.api_type == "anthropic":
        anthropic_mapping: dict[str, str | None] = {
            "none": None,
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "max",
        }
        if normalized_effort is None:
            return options
        mapped = anthropic_mapping.get(normalized_effort)
        if mapped is not None:
            options["anthropic_effort"] = mapped
        return options

    if normalized_effort is not None and normalized_effort != "auto":
        options["openai_reasoning_effort"] = normalized_effort

    if supports_openai_reasoning_summary(model) and normalized_summary not in (None, "auto"):
        options["openai_reasoning_summary"] = normalized_summary

    return options


def format_reasoning_effort(reasoning_effort: str) -> str:
    """Return a user-facing label for the current reasoning setting."""
    if reasoning_effort == "auto":
        return "auto"
    if reasoning_effort == "none":
        return "off"
    return reasoning_effort
