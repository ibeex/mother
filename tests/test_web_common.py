"""Tests for shared web-tool helpers."""

from mother.tools.web_common import should_retry_with_jina_api_key


def test_should_retry_with_jina_api_key_for_rate_limit():
    assert should_retry_with_jina_api_key(429, "rate limit exceeded") is True


def test_should_retry_with_jina_api_key_for_auth_required_detail():
    detail = (
        "AuthenticationRequiredError: Authentication is required to use this endpoint. "
        "Please provide a valid API key via Authorization header"
    )
    assert should_retry_with_jina_api_key(401, detail) is True


def test_should_not_retry_with_jina_api_key_for_unrelated_error():
    assert should_retry_with_jina_api_key(403, "forbidden") is False
