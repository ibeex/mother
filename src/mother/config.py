"""Configuration management for Mother TUI."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from mother.models import ModelEntry, default_model_entries, load_model_entries
from mother.reasoning import (
    DEFAULT_OPENAI_REASONING_SUMMARY,
    DEFAULT_REASONING_EFFORT,
    parse_openai_reasoning_summary,
    parse_reasoning_effort,
)
from mother.session import default_markdown_export_dir
from mother.system_prompt import DEFAULT_BASE_SYSTEM

CONFIG_DIR = Path.home() / ".config" / "mother"
CONFIG_FILE = CONFIG_DIR / "config.toml"
LEGACY_CONFIG_FILE = Path.home() / ".mother" / "config.toml"

_DEFAULT_SYSTEM = DEFAULT_BASE_SYSTEM

_DEFAULT_CONFIG_TEMPLATE = """\
# Mother TUI configuration

# Selected model id from the [[models]] registry below.
# Leave empty until you configure your models.
model = ""

# Built-in Textual theme.
# theme = "catppuccin-mocha"

# Base system prompt sent with every conversation.
# Mother appends runtime context such as date, OS, current directory, mode, and tools.
# system_prompt = "You are Mother, a concise and helpful assistant."

# Reasoning effort for models that support it.
# Supported values: "auto", "none", "low", "medium", "high", "xhigh"
# reasoning_effort = "medium"

# OpenAI Responses reasoning summary visibility.
# Supported values: "auto", "concise", "detailed"
# Set this to "detailed" if you want visible streamed reasoning summaries when available.
# openai_reasoning_summary = "auto"

# Enable agent mode (allows LLM to run shell commands via the bash tool)
# tools_enabled = false

# Optional CA bundle path for web_search and web_fetch.
# Leave empty to use only Python/system certificates.
# Set this when your network uses SSL inspection with a custom root CA.
# ca_bundle_path = "/etc/ssl/certs/ib_cert.pem"

# Directory used by /save, Ctrl+S, and `mother --save` markdown exports.
# session_markdown_dir = "~/Documents/mother"

# Legacy allowlist from the old regex-based bash guard.
# Retained for backwards compatibility but ignored by the current
# LLM-based bash guard.
# allowlist = ["ls", "cat"]

# Anonymous multi-model council used by /council.
# The ids here must match [[models]] entries below.
# [council]
# members = ["gpt-5", "g3", "opus"]
# judge = "opus"
# max_context_turns = 8
# max_context_chars = 12000

# Add your own [[models]] entries below.
# Default config intentionally ships with no models.
# Put named secrets in ~/.config/mother/keys.json and reference them here.
# Example keys.json:
# {
#   "CODY_KEY": "paste-remote-key-here",
#   "LOCAL_KEY": "lm-studio"
# }
#
# Example model:
# [[models]]
# id = "my-model"
# name = "provider-model-name"
# api_type = "openai-responses"  # openai-responses | openai-chat | anthropic
# base_url = "https://example.com/"
# api_key = "CODY_KEY"
# supports_tools = true
# supports_reasoning = true
# supports_images = true
"""


@dataclass(frozen=True, slots=True)
class CouncilConfig:
    """Configuration for the anonymous /council workflow."""

    members: tuple[str, ...] = ("gpt-5", "g3", "opus")
    judge: str = "opus"
    max_context_turns: int = 8
    max_context_chars: int = 12000


@dataclass
class MotherConfig:
    model: str = ""
    theme: str = "catppuccin-mocha"
    system_prompt: str = field(default=_DEFAULT_SYSTEM)
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    openai_reasoning_summary: str = DEFAULT_OPENAI_REASONING_SUMMARY
    tools_enabled: bool = False
    ca_bundle_path: str = ""
    session_markdown_dir: str = field(default_factory=lambda: str(default_markdown_export_dir()))
    allowlist: frozenset[str] = field(default_factory=lambda: frozenset({"ls", "cat"}))
    models: list[ModelEntry] = field(default_factory=default_model_entries)
    council: CouncilConfig = field(default_factory=CouncilConfig)


DEFAULT_MODEL = MotherConfig().model
DEFAULT_SYSTEM = MotherConfig().system_prompt


def save_default_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(_DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")


def _parse_council_members(value: object, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list):
        raise ValueError("Config field 'council.members' must be an array of strings.")

    members: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str):
            raise ValueError("Config field 'council.members' must be an array of strings.")
        stripped = item.strip()
        if stripped:
            members.append(stripped)
    return tuple(members)


def _parse_council_judge(value: object, *, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError("Config field 'council.judge' must be a string.")
    return value.strip()


def _parse_council_limit(value: object, *, field_name: str, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Config field {field_name!r} must be a non-negative integer.")
    return value


def _load_council_config(data: dict[str, object]) -> CouncilConfig:
    raw_council = data.get("council")
    default = CouncilConfig()
    if raw_council is None:
        return default
    if not isinstance(raw_council, dict):
        raise ValueError("Config field 'council' must be a table.")

    council_data = cast(dict[str, object], raw_council)
    return CouncilConfig(
        members=_parse_council_members(
            council_data.get("members"),
            default=default.members,
        ),
        judge=_parse_council_judge(council_data.get("judge"), default=default.judge),
        max_context_turns=_parse_council_limit(
            council_data.get("max_context_turns"),
            field_name="council.max_context_turns",
            default=default.max_context_turns,
        ),
        max_context_chars=_parse_council_limit(
            council_data.get("max_context_chars"),
            field_name="council.max_context_chars",
            default=default.max_context_chars,
        ),
    )


def load_config(path: Path | None = None) -> MotherConfig:
    resolved = path or CONFIG_FILE
    if path is None and not resolved.exists() and LEGACY_CONFIG_FILE.exists():
        resolved = LEGACY_CONFIG_FILE
    if not resolved.exists():
        save_default_config(CONFIG_FILE if path is None else resolved)
        return MotherConfig()

    with resolved.open("rb") as f:
        data = tomllib.load(f)

    raw_allowlist = cast(list[str] | None, data.get("allowlist"))
    allowlist = frozenset(raw_allowlist) if raw_allowlist is not None else MotherConfig().allowlist
    raw_reasoning_effort = cast(str | None, data.get("reasoning_effort"))
    reasoning_effort = (
        DEFAULT_REASONING_EFFORT
        if raw_reasoning_effort is None
        else parse_reasoning_effort(raw_reasoning_effort)
    )
    raw_openai_reasoning_summary = cast(str | None, data.get("openai_reasoning_summary"))
    openai_reasoning_summary = (
        DEFAULT_OPENAI_REASONING_SUMMARY
        if raw_openai_reasoning_summary is None
        else parse_openai_reasoning_summary(raw_openai_reasoning_summary)
    )
    return MotherConfig(
        model=cast(str, data.get("model", MotherConfig.model)),
        theme=cast(str, data.get("theme", MotherConfig.theme)),
        system_prompt=cast(str, data.get("system_prompt", MotherConfig.system_prompt)),
        reasoning_effort=reasoning_effort,
        openai_reasoning_summary=openai_reasoning_summary,
        tools_enabled=cast(bool, data.get("tools_enabled", MotherConfig.tools_enabled)),
        ca_bundle_path=cast(str, data.get("ca_bundle_path", MotherConfig.ca_bundle_path)),
        session_markdown_dir=cast(
            str,
            data.get("session_markdown_dir", MotherConfig().session_markdown_dir),
        ),
        allowlist=allowlist,
        models=load_model_entries(cast(dict[str, object], data)),
        council=_load_council_config(cast(dict[str, object], data)),
    )


def apply_cli_overrides(
    config: MotherConfig,
    model: str | None,
    system: str | None,
) -> MotherConfig:
    return MotherConfig(
        model=model if model is not None else config.model,
        theme=config.theme,
        system_prompt=system if system is not None else config.system_prompt,
        reasoning_effort=config.reasoning_effort,
        openai_reasoning_summary=config.openai_reasoning_summary,
        tools_enabled=config.tools_enabled,
        ca_bundle_path=config.ca_bundle_path,
        session_markdown_dir=config.session_markdown_dir,
        allowlist=config.allowlist,
        models=list(config.models),
        council=config.council,
    )
