"""Helpers for reasoning-effort capable models."""

from __future__ import annotations

from typing import Final

from llm.models import Model

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


def supports_reasoning_effort(model: Model | None) -> bool:
    """Return whether the model exposes a ``reasoning_effort`` option."""
    if model is None:
        return False
    options_class = getattr(model, "Options", None)
    model_fields = getattr(options_class, "model_fields", None)
    return isinstance(model_fields, dict) and "reasoning_effort" in model_fields


def supported_reasoning_efforts(model: Model | None) -> tuple[str, ...]:
    """Return the reasoning-effort values Mother supports for reasoning models."""
    if not supports_reasoning_effort(model):
        return ()
    return _SUPPORTED_REASONING_EFFORTS


def build_reasoning_options(model: Model | None, reasoning_effort: str) -> dict[str, object]:
    """Return request options for reasoning-capable models."""
    supported = supported_reasoning_efforts(model)
    if not supported:
        return {}
    normalized = normalize_reasoning_effort(reasoning_effort)
    if normalized is None or normalized == "auto" or normalized not in supported:
        return {}
    return {"reasoning_effort": normalized}


def format_reasoning_effort(reasoning_effort: str) -> str:
    """Return a user-facing label for the current reasoning setting."""
    if reasoning_effort == "auto":
        return "auto"
    if reasoning_effort == "none":
        return "off"
    return reasoning_effort
