"""Shared agent-mode definitions and formatting helpers."""

from __future__ import annotations

from typing import Literal

AgentProfile = Literal["standard", "deep_research"]
RuntimeMode = Literal["chat", "agent", "deep_research"]

DEFAULT_AGENT_PROFILE: AgentProfile = "standard"

_STANDARD_AGENT_PROFILE_ALIASES: frozenset[str] = frozenset(
    {"agent", "default", "normal", "standard"}
)
_DEEP_RESEARCH_PROFILE_ALIASES: frozenset[str] = frozenset({"deep", "deep research", "research"})


def _normalize_phrase(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def normalize_agent_profile(value: str) -> AgentProfile | None:
    """Normalize a user-facing agent profile label to an internal value."""
    normalized = _normalize_phrase(value)
    if normalized in _STANDARD_AGENT_PROFILE_ALIASES:
        return "standard"
    if normalized in _DEEP_RESEARCH_PROFILE_ALIASES:
        return "deep_research"
    return None


def format_agent_profile(profile: AgentProfile) -> str:
    """Return the user-facing label for an agent profile."""
    if profile == "deep_research":
        return "deep research"
    return "standard"


def resolve_runtime_mode(*, agent_enabled: bool, agent_profile: AgentProfile) -> RuntimeMode:
    """Return the effective runtime mode shown to the model and UI."""
    if not agent_enabled:
        return "chat"
    if agent_profile == "deep_research":
        return "deep_research"
    return "agent"


def format_runtime_mode(mode: RuntimeMode) -> str:
    """Return the user-facing label for a runtime mode."""
    if mode == "deep_research":
        return "deep research"
    return mode


def format_agent_status(agent_enabled: bool, agent_profile: AgentProfile) -> str:
    """Return the compact status-line value for the current agent state."""
    if not agent_enabled:
        return "off"
    if agent_profile == "deep_research":
        return "research"
    return "on"
