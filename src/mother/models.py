"""Config-driven model registry for Mother."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal, cast

from pydantic_ai.models import Model as PydanticModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

ApiType = Literal["openai-responses", "openai-chat", "anthropic"]


@dataclass(frozen=True, slots=True)
class ModelEntry:
    id: str
    name: str
    api_type: ApiType
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = ""
    supports_tools: bool = False
    supports_reasoning: bool = False
    supports_images: bool = False


_DEFAULT_MODEL_ENTRIES: tuple[ModelEntry, ...] = ()

_VALID_API_TYPES: frozenset[str] = frozenset({"openai-responses", "openai-chat", "anthropic"})
_DEFAULT_CONFIG_FILE = Path.home() / ".config" / "mother" / "config.toml"
_DEFAULT_KEYS_FILE = Path.home() / ".config" / "mother" / "keys.json"


def default_model_entries() -> list[ModelEntry]:
    """Return built-in example model entries used as sane defaults."""
    return list(_DEFAULT_MODEL_ENTRIES)


def fallback_model_entry(model_id: str) -> ModelEntry:
    """Return a minimal fallback entry for ad-hoc model ids."""
    return ModelEntry(
        id=model_id,
        name=model_id,
        api_type="openai-responses",
    )


def _required_str(raw_entry: dict[str, object], key: str) -> str:
    value = raw_entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Model config field {key!r} must be a non-empty string.")
    return value.strip()


def _optional_str(raw_entry: dict[str, object], key: str) -> str:
    value = raw_entry.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"Model config field {key!r} must be a string.")
    return value.strip()


def _optional_bool(raw_entry: dict[str, object], key: str) -> bool:
    value = raw_entry.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f"Model config field {key!r} must be a boolean.")
    return value


def load_model_entries(toml_data: dict[str, object]) -> list[ModelEntry]:
    """Parse ``[[models]]`` entries from TOML data."""
    raw_models = toml_data.get("models")
    if raw_models is None:
        return default_model_entries()
    if not isinstance(raw_models, list):
        raise ValueError("Config field 'models' must be an array of tables.")
    if not raw_models:
        return default_model_entries()

    entries: list[ModelEntry] = []
    seen_ids: set[str] = set()
    for raw_item in cast(list[object], raw_models):
        if not isinstance(raw_item, dict):
            raise ValueError("Each [[models]] entry must be a table.")
        raw_entry = cast(dict[str, object], raw_item)
        model_id = _required_str(raw_entry, "id")
        if model_id in seen_ids:
            raise ValueError(f"Duplicate model id {model_id!r} in config.")
        seen_ids.add(model_id)

        api_type_raw = _required_str(raw_entry, "api_type")
        if api_type_raw not in _VALID_API_TYPES:
            valid_values = ", ".join(sorted(_VALID_API_TYPES))
            raise ValueError(
                f"Invalid api_type {api_type_raw!r} for model {model_id!r}. Expected one of: {valid_values}"
            )

        entries.append(
            ModelEntry(
                id=model_id,
                name=_required_str(raw_entry, "name"),
                api_type=cast(ApiType, api_type_raw),
                base_url=_optional_str(raw_entry, "base_url"),
                api_key=_optional_str(raw_entry, "api_key"),
                api_key_env=_optional_str(raw_entry, "api_key_env"),
                supports_tools=_optional_bool(raw_entry, "supports_tools"),
                supports_reasoning=_optional_bool(raw_entry, "supports_reasoning"),
                supports_images=_optional_bool(raw_entry, "supports_images"),
            )
        )
    return entries


@cache
def load_key_aliases(path: Path | None = None) -> dict[str, str]:
    """Load named API keys from ``keys.json``."""
    resolved = path or _DEFAULT_KEYS_FILE
    if not resolved.exists():
        return {}
    with resolved.open(encoding="utf-8") as handle:
        raw_data = json.load(handle)
    if not isinstance(raw_data, dict):
        raise ValueError("keys.json must contain a JSON object of key names to secret strings.")

    aliases: dict[str, str] = {}
    for raw_key, raw_value in raw_data.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            raise ValueError("keys.json keys and values must both be strings.")
        aliases[raw_key] = raw_value
    return aliases


def resolve_api_key(entry: ModelEntry) -> str:
    """Resolve the configured API key for a model entry."""
    if entry.api_key:
        aliases = load_key_aliases()
        return aliases.get(entry.api_key, entry.api_key)
    if entry.api_key_env:
        return os.environ.get(entry.api_key_env, "")
    return ""


@cache
def create_pydantic_model(entry: ModelEntry) -> PydanticModel:
    """Create and cache a pydantic-ai model instance for a registry entry."""
    api_key = resolve_api_key(entry) or None
    base_url = entry.base_url or None

    if entry.api_type == "openai-responses":
        provider = OpenAIProvider(base_url=base_url, api_key=api_key)
        return OpenAIResponsesModel(entry.name, provider=provider)
    if entry.api_type == "openai-chat":
        provider = OpenAIProvider(base_url=base_url, api_key=api_key)
        return OpenAIChatModel(entry.name, provider=provider)

    provider = AnthropicProvider(base_url=base_url, api_key=api_key)
    return AnthropicModel(entry.name, provider=provider)


def find_model_entry(model_id: str, entries: list[ModelEntry]) -> ModelEntry | None:
    """Return the configured entry for a model id, if present."""
    for entry in entries:
        if entry.id == model_id:
            return entry
    return None


def resolve_model_entry(model_id: str, entries: list[ModelEntry]) -> ModelEntry:
    """Return the configured entry or a minimal fallback entry."""
    return find_model_entry(model_id, entries) or fallback_model_entry(model_id)


def get_model_entry(model_id: str, entries: list[ModelEntry] | None = None) -> ModelEntry | None:
    """Look up a model entry by id."""
    source_entries = entries if entries is not None else get_all_entries()
    return find_model_entry(model_id, source_entries)


def get_all_entries(path: Path | None = None) -> list[ModelEntry]:
    """Load all configured model entries from Mother's config."""
    resolved = path or _DEFAULT_CONFIG_FILE
    if not resolved.exists():
        return default_model_entries()
    with resolved.open("rb") as handle:
        data = tomllib.load(handle)
    return load_model_entries(cast(dict[str, object], data))
