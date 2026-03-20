"""Helpers for reasoning-effort capable models."""

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


def supports_reasoning_effort(model: ModelEntry | None) -> bool:
    """Return whether the configured model supports reasoning control."""
    return model is not None and model.supports_reasoning


def supported_reasoning_efforts(model: ModelEntry | None) -> tuple[str, ...]:
    """Return the reasoning-effort values Mother supports for reasoning models."""
    if not supports_reasoning_effort(model):
        return ()
    return _SUPPORTED_REASONING_EFFORTS


def build_reasoning_options(model: ModelEntry | None, reasoning_effort: str) -> dict[str, object]:
    """Return provider-specific request settings for reasoning-capable models."""
    if not supports_reasoning_effort(model):
        return {}

    normalized = normalize_reasoning_effort(reasoning_effort)
    if normalized is None or normalized == "auto":
        return {}

    if model is not None and model.api_type == "anthropic":
        anthropic_mapping: dict[str, str | None] = {
            "none": None,
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "max",
        }
        mapped = anthropic_mapping.get(normalized)
        if mapped is None:
            return {}
        return {"anthropic_effort": mapped}

    return {"openai_reasoning_effort": normalized}


def format_reasoning_effort(reasoning_effort: str) -> str:
    """Return a user-facing label for the current reasoning setting."""
    if reasoning_effort == "auto":
        return "auto"
    if reasoning_effort == "none":
        return "off"
    return reasoning_effort
