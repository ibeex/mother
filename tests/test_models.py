"""Tests for pydantic-ai model construction and provider defaults."""

import asyncio

import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from mother.models import (
    ApiType,
    ModelEntry,
    close_cached_models_for_current_loop,
    create_pydantic_model,
)

_GATEWAY_BASE_URL = "https://gateway.example.test/"


def _entry(api_type: ApiType, name: str = "gpt-6-luna") -> ModelEntry:
    return ModelEntry(
        id=name, name=name, api_type=api_type, base_url=_GATEWAY_BASE_URL, api_key="test-key"
    )


def test_openai_responses_model_disables_prompt_cache_diagnostics() -> None:
    """OpenAI-compatible gateways reject the diagnostic comparison id, so opt out."""
    model = create_pydantic_model(_entry("openai-responses"))

    settings = dict(model.settings or {})
    assert settings["openai_prompt_cache_diagnostics"] is False


def test_openai_chat_model_keeps_default_settings() -> None:
    model = create_pydantic_model(_entry("openai-chat"))

    settings = dict(model.settings or {})
    assert "openai_prompt_cache_diagnostics" not in settings


@pytest.mark.parametrize("api_type", ["openai-responses", "openai-chat", "anthropic"])
def test_model_cache_is_scoped_to_the_running_event_loop(api_type: ApiType) -> None:
    """A cached async model must never be reused after its event loop closes."""

    async def create_and_close_model() -> object:
        entry = _entry(api_type)
        model = create_pydantic_model(entry)
        assert isinstance(model, OpenAIResponsesModel | OpenAIChatModel | AnthropicModel)
        provider = model.provider
        assert provider is not None
        try:
            assert create_pydantic_model(entry) is model
            assert not provider.client.is_closed()
        finally:
            await close_cached_models_for_current_loop()
        assert provider.client.is_closed()
        await close_cached_models_for_current_loop()
        return model

    first_model = asyncio.run(create_and_close_model())
    second_model = asyncio.run(create_and_close_model())

    assert first_model is not second_model
