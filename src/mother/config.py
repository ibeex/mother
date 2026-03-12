"""Configuration management for Mother TUI."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "mother"
CONFIG_FILE = CONFIG_DIR / "config.toml"

_DEFAULT_SYSTEM = (
    "Formulate all responses as if you were the sentient AI named Mother from the Alien movies."
)

_DEFAULT_CONFIG_TEMPLATE = """\
# Mother TUI configuration

# LLM model to use (e.g. "gpt-4o", "gpt-5", "claude-3-5-sonnet-latest")
# model = "gpt-5"

# System prompt sent with every conversation
# system_prompt = "Formulate all responses as if you were the sentient AI named Mother from the Alien movies."

# Enable tool use (web search, file read, etc.) — not yet implemented
# tools_enabled = false
"""


@dataclass
class MotherConfig:
    model: str = "gpt-5"
    system_prompt: str = field(default=_DEFAULT_SYSTEM)
    tools_enabled: bool = False


DEFAULT_MODEL = MotherConfig().model
DEFAULT_SYSTEM = MotherConfig().system_prompt


def save_default_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")


def load_config(path: Path | None = None) -> MotherConfig:
    resolved = path or CONFIG_FILE
    if not resolved.exists():
        save_default_config(resolved)
        return MotherConfig()

    with resolved.open("rb") as f:
        data = tomllib.load(f)

    return MotherConfig(
        model=data.get("model", MotherConfig.model),
        system_prompt=data.get("system_prompt", MotherConfig.system_prompt),
        tools_enabled=data.get("tools_enabled", MotherConfig.tools_enabled),
    )


def apply_cli_overrides(
    config: MotherConfig,
    model: str | None,
    system: str | None,
) -> MotherConfig:
    return MotherConfig(
        model=model if model is not None else config.model,
        system_prompt=system if system is not None else config.system_prompt,
        tools_enabled=config.tools_enabled,
    )
