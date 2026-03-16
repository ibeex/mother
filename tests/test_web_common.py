"""Tests for shared web tool helpers."""

from __future__ import annotations

import ssl
from pathlib import Path
from typing import cast, final
from unittest.mock import patch

from mother.tools.web_common import build_ssl_context, should_retry_with_jina_api_key


@final
class _FakeSSLContext:
    def __init__(self) -> None:
        self.loaded_cafiles: list[str] = []
        self.verify_flags: int = 0b1111

    def load_verify_locations(self, cafile: str) -> None:
        self.loaded_cafiles.append(cafile)


def test_should_retry_with_jina_api_key_for_rate_limit() -> None:
    assert should_retry_with_jina_api_key(429, "rate limit exceeded") is True


def test_should_retry_with_jina_api_key_for_auth_required_detail() -> None:
    detail = (
        "AuthenticationRequiredError: Authentication is required to use this endpoint. "
        "Please provide a valid API key via Authorization header"
    )
    assert should_retry_with_jina_api_key(401, detail) is True


def test_should_not_retry_with_jina_api_key_for_unrelated_error() -> None:
    assert should_retry_with_jina_api_key(403, "forbidden") is False


def test_build_ssl_context_disables_strict_x509_verification_when_available() -> None:
    fake_context_impl = _FakeSSLContext()
    fake_context = cast(ssl.SSLContext, cast(object, fake_context_impl))

    with (
        patch("mother.tools.web_common.ssl.create_default_context", return_value=fake_context),
        patch("mother.tools.web_common.ssl.VERIFY_X509_STRICT", 0b0100),
    ):
        context = build_ssl_context()

    assert context is fake_context
    assert fake_context_impl.verify_flags == 0b1011


def test_build_ssl_context_does_not_load_extra_ca_bundle_when_not_configured() -> None:
    fake_context_impl = _FakeSSLContext()
    fake_context = cast(ssl.SSLContext, cast(object, fake_context_impl))

    with patch("mother.tools.web_common.ssl.create_default_context", return_value=fake_context):
        context = build_ssl_context()

    assert context is fake_context
    assert fake_context_impl.loaded_cafiles == []


def test_build_ssl_context_uses_configured_ca_bundle_path(tmp_path: Path) -> None:
    fake_context_impl = _FakeSSLContext()
    fake_context = cast(ssl.SSLContext, cast(object, fake_context_impl))
    ca_bundle = tmp_path / "office.pem"
    _ = ca_bundle.write_text("dummy cert", encoding="utf-8")

    with patch("mother.tools.web_common.ssl.create_default_context", return_value=fake_context):
        context = build_ssl_context(str(ca_bundle))

    assert context is fake_context
    assert fake_context_impl.loaded_cafiles == [str(ca_bundle)]


def test_build_ssl_context_rejects_missing_configured_ca_bundle() -> None:
    fake_context_impl = _FakeSSLContext()
    fake_context = cast(ssl.SSLContext, cast(object, fake_context_impl))

    with patch("mother.tools.web_common.ssl.create_default_context", return_value=fake_context):
        try:
            _ = build_ssl_context("/tmp/does-not-exist.pem")
        except RuntimeError as exc:
            assert str(exc) == "Configured CA bundle was not found: /tmp/does-not-exist.pem"
        else:
            raise AssertionError("Expected RuntimeError for missing configured CA bundle")
