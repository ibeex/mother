"""Tests for pydantic-ai model construction and provider defaults."""

from mother.models import ApiType, ModelEntry, create_pydantic_model

_GATEWAY_BASE_URL = "https://gateway.example.test/"


def _entry(api_type: ApiType, name: str = "gpt-6-luna") -> ModelEntry:
    return ModelEntry(id=name, name=name, api_type=api_type, base_url=_GATEWAY_BASE_URL)


def test_openai_responses_model_disables_prompt_cache_diagnostics() -> None:
    """OpenAI-compatible gateways reject the diagnostic comparison id, so opt out."""
    create_pydantic_model.cache_clear()
    model = create_pydantic_model(_entry("openai-responses"))

    settings = dict(model.settings or {})
    assert settings["openai_prompt_cache_diagnostics"] is False


def test_openai_chat_model_keeps_default_settings() -> None:
    create_pydantic_model.cache_clear()
    model = create_pydantic_model(_entry("openai-chat"))

    settings = dict(model.settings or {})
    assert "openai_prompt_cache_diagnostics" not in settings
