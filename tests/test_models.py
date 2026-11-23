# pyright: reportPrivateUsage=false
"""Tests for pydantic-ai model construction and provider defaults."""

import asyncio

import httpx2
import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from mother import models
from mother.models import (
    ApiType,
    ModelEntry,
    _build_http_client,
    _stamp_user_agent,
    close_cached_models_for_current_loop,
    create_pydantic_model,
)
from mother.user_agent import user_agent

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


def test_user_agent_hook_overrides_the_sdk_header() -> None:
    """Gateways must see `mother/<version>`, not `AsyncOpenAI/Python <version>`."""
    sent_headers: dict[str, str] = {}

    def _handler(request: httpx2.Request) -> httpx2.Response:
        sent_headers.update(dict(request.headers))
        return httpx2.Response(200, json={})

    async def send_with_sdk_header() -> None:
        client = httpx2.AsyncClient(
            transport=httpx2.MockTransport(_handler),
            event_hooks={"request": [_stamp_user_agent]},
        )
        # The SDK writes its own User-Agent as a request header, which wins over
        # any header configured on the client itself.
        _ = await client.post(
            "https://gateway.example.test/v1/chat",
            headers={"User-Agent": "AsyncOpenAI/Python 2.0.0"},
        )
        await client.aclose()

    asyncio.run(send_with_sdk_header())

    assert sent_headers["user-agent"] == user_agent()
    assert sent_headers["user-agent"].startswith("mother/")


@pytest.mark.parametrize("api_type", ["openai-responses", "openai-chat", "anthropic"])
def test_every_provider_uses_the_user_agent_stamping_client(
    api_type: ApiType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither SDK may fall back to the HTTP client it would create itself."""
    built: list[httpx2.AsyncClient] = []

    def _spy_build_http_client(ca_bundle_path: str) -> httpx2.AsyncClient:
        client = _build_http_client(ca_bundle_path)
        built.append(client)
        return client

    monkeypatch.setattr(models, "_build_http_client", _spy_build_http_client)
    model = create_pydantic_model(_entry(api_type))

    assert isinstance(model, OpenAIResponsesModel | OpenAIChatModel | AnthropicModel)
    assert len(built) == 1
    assert _stamp_user_agent in built[0].event_hooks["request"]

    asyncio.run(built[0].aclose())
