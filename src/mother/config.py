"""Configuration management for Mother TUI."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from mother.system_prompt import DEFAULT_BASE_SYSTEM

CONFIG_DIR = Path.home() / ".config" / "mother"
CONFIG_FILE = CONFIG_DIR / "config.toml"

_DEFAULT_SYSTEM = DEFAULT_BASE_SYSTEM

_DEFAULT_CONFIG_TEMPLATE = """\
# Mother TUI configuration

# LLM model to use (e.g. "gpt-4o", "gpt-5", "claude-3-5-sonnet-latest")
# model = "gpt-5"

# Base system prompt sent with every conversation.
# Mother appends runtime context such as date, OS, current directory, mode, and tools.
# system_prompt = "You are Mother, a concise and helpful assistant."

# Enable agent mode (allows LLM to run shell commands via the bash tool)
# tools_enabled = false

# Optional CA bundle path for web_search and web_fetch.
# Leave empty to use only Python/system certificates.
# Set this when your network uses SSL inspection with a custom root CA.
# ca_bundle_path = "/etc/ssl/certs/ib_cert.pem"

# Legacy allowlist from the old regex-based bash guard.
# Retained for backwards compatibility but ignored by the current
# LLM-based bash guard.
# allowlist = ["ls", "cat"]
"""


@dataclass
class MotherConfig:
    model: str = "gpt-5"
    system_prompt: str = field(default=_DEFAULT_SYSTEM)
    tools_enabled: bool = False
    ca_bundle_path: str = ""
    allowlist: frozenset[str] = field(default_factory=lambda: frozenset({"ls", "cat"}))


DEFAULT_MODEL = MotherConfig().model
DEFAULT_SYSTEM = MotherConfig().system_prompt


def save_default_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(_DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")


def load_config(path: Path | None = None) -> MotherConfig:
    resolved = path or CONFIG_FILE
    if not resolved.exists():
        save_default_config(resolved)
        return MotherConfig()

    with resolved.open("rb") as f:
        data = tomllib.load(f)

    raw_allowlist = cast(list[str] | None, data.get("allowlist"))
    allowlist = frozenset(raw_allowlist) if raw_allowlist is not None else MotherConfig().allowlist
    return MotherConfig(
        model=cast(str, data.get("model", MotherConfig.model)),
        system_prompt=cast(str, data.get("system_prompt", MotherConfig.system_prompt)),
        tools_enabled=cast(bool, data.get("tools_enabled", MotherConfig.tools_enabled)),
        ca_bundle_path=cast(str, data.get("ca_bundle_path", MotherConfig.ca_bundle_path)),
        allowlist=allowlist,
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
        ca_bundle_path=config.ca_bundle_path,
        allowlist=config.allowlist,
    )
