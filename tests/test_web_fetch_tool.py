"""Tests for the HTTP fetch tool."""

from __future__ import annotations

import io
import ssl
import subprocess
from email.message import Message
from typing import final
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

from mother.tools import get_default_tools
from mother.tools.web_fetch_tool import MAX_TIMEOUT, make_web_fetch_tool


@final
class _FakeResponse:
    def __init__(
        self, content: bytes, *, status: int = 200, content_type: str = "text/plain"
    ) -> None:
        self._content: bytes = content
        self.status: int = status
        self.headers: dict[str, str] = {"Content-Type": content_type}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        _ = exc_type
        _ = exc
        _ = tb
        return False

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            return self._content
        return self._content[:amount]


def _get_header(request: Request, name: str) -> str | None:
    normalized_name = name.lower()
    for header_name, header_value in request.header_items():
        if header_name.lower() == normalized_name:
            return header_value
    return None


def test_web_fetch_tool_raw_mode_uses_urllib_request():
    captured_requests: list[Request] = []

    def _fake_urlopen(
        request: Request, *, timeout: float, context: ssl.SSLContext
    ) -> _FakeResponse:
        captured_requests.append(request)
        assert timeout == 12.0
        assert context is not None
        return _FakeResponse(b'{"ok": true}', content_type="application/json")

    with patch("mother.tools.web_fetch_tool.urllib.request.urlopen", side_effect=_fake_urlopen):
        tool = make_web_fetch_tool()
        output = tool(
            "http://localhost:8000/api",
            mode="raw",
            method="POST",
            headers_json='{"Accept": "application/json"}',
            body='{"hello": "world"}',
            timeout=12.0,
        )

    assert output.startswith("## Fetch Result")
    assert "- Mode: raw" in output
    assert "- URL: http://localhost:8000/api" in output
    assert "- Status: 200" in output
    assert '{"ok": true}' in output
    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.full_url == "http://localhost:8000/api"
    assert request.method == "POST"
    assert _get_header(request, "Accept") == "application/json"
    assert request.data == b'{"hello": "world"}'


def test_web_fetch_tool_auto_mode_uses_raw_for_local_urls():
    captured_requests: list[Request] = []

    def _fake_urlopen(
        request: Request, *, timeout: float, context: ssl.SSLContext
    ) -> _FakeResponse:
        _ = timeout
        assert context is not None
        captured_requests.append(request)
        return _FakeResponse(b"local ok")

    with patch("mother.tools.web_fetch_tool.urllib.request.urlopen", side_effect=_fake_urlopen):
        tool = make_web_fetch_tool()
        output = tool("http://localhost:3000/health")

    assert "- Mode: raw" in output
    assert len(captured_requests) == 1
    assert captured_requests[0].full_url == "http://localhost:3000/health"


def test_web_fetch_tool_auto_mode_uses_jina_for_remote_pages():
    captured_requests: list[Request] = []

    def _fake_urlopen(
        request: Request, *, timeout: float, context: ssl.SSLContext
    ) -> _FakeResponse:
        _ = timeout
        assert context is not None
        captured_requests.append(request)
        return _FakeResponse(b"Readable content")

    with patch("mother.tools.web_fetch_tool.urllib.request.urlopen", side_effect=_fake_urlopen):
        tool = make_web_fetch_tool()
        output = tool("https://example.com/docs")

    assert "- Mode: jina" in output
    assert "Readable content" in output
    assert len(captured_requests) == 1
    assert captured_requests[0].full_url == "https://r.jina.ai/https://example.com/docs"
    assert _get_header(captured_requests[0], "Authorization") is None


def test_web_fetch_tool_retries_jina_with_api_key_after_rate_limit():
    captured_requests: list[Request] = []
    rate_limit_error = HTTPError(
        url="https://r.jina.ai/https://example.com/docs",
        code=429,
        msg="Too Many Requests",
        hdrs=Message(),
        fp=io.BytesIO(b"rate limit exceeded"),
    )

    def _fake_urlopen(
        request: Request, *, timeout: float, context: ssl.SSLContext
    ) -> _FakeResponse:
        _ = timeout
        assert context is not None
        captured_requests.append(request)
        if len(captured_requests) == 1:
            raise rate_limit_error
        return _FakeResponse(b"Retried readable content")

    completed = subprocess.CompletedProcess(
        args=["pass", "api/jina"], returncode=0, stdout="secret-key\n"
    )

    with (
        patch("mother.tools.web_common.subprocess.run", return_value=completed),
        patch("mother.tools.web_fetch_tool.urllib.request.urlopen", side_effect=_fake_urlopen),
    ):
        tool = make_web_fetch_tool()
        output = tool("https://example.com/docs", mode="jina")

    assert "Retried readable content" in output
    assert len(captured_requests) == 2
    assert _get_header(captured_requests[0], "Authorization") is None
    assert _get_header(captured_requests[1], "Authorization") == "Bearer secret-key"


def test_web_fetch_tool_raw_mode_retries_cloudflare_challenge_with_honest_user_agent():
    captured_requests: list[Request] = []
    challenge_headers = Message()
    challenge_headers["cf-mitigated"] = "challenge"
    challenge_error = HTTPError(
        url="https://example.com/protected",
        code=403,
        msg="Forbidden",
        hdrs=challenge_headers,
        fp=io.BytesIO(b"challenge"),
    )

    def _fake_urlopen(
        request: Request, *, timeout: float, context: ssl.SSLContext
    ) -> _FakeResponse:
        _ = timeout
        assert context is not None
        captured_requests.append(request)
        if len(captured_requests) == 1:
            raise challenge_error
        return _FakeResponse(b"Retried raw content")

    with patch("mother.tools.web_fetch_tool.urllib.request.urlopen", side_effect=_fake_urlopen):
        tool = make_web_fetch_tool()
        output = tool("https://example.com/protected", mode="raw")

    assert "Retried raw content" in output
    assert len(captured_requests) == 2
    first_user_agent = _get_header(captured_requests[0], "User-Agent")
    second_user_agent = _get_header(captured_requests[1], "User-Agent")
    assert first_user_agent is not None
    assert second_user_agent == "mother/1.0"
    assert second_user_agent != first_user_agent


def test_web_fetch_tool_caps_timeout_to_maximum():
    captured_timeouts: list[float] = []

    def _fake_urlopen(
        request: Request, *, timeout: float, context: ssl.SSLContext
    ) -> _FakeResponse:
        _ = request
        assert context is not None
        captured_timeouts.append(timeout)
        return _FakeResponse(b"Readable content")

    with patch("mother.tools.web_fetch_tool.urllib.request.urlopen", side_effect=_fake_urlopen):
        tool = make_web_fetch_tool()
        output = tool("https://example.com/docs", timeout=999.0)

    assert "Readable content" in output
    assert captured_timeouts == [MAX_TIMEOUT]


def test_web_fetch_tool_rejects_non_positive_timeout():
    tool = make_web_fetch_tool()
    output = tool("https://example.com/docs", timeout=0)
    assert output == "Error: timeout must be a positive number."


def test_web_fetch_tool_truncates_raw_response_when_size_limit_is_exceeded():
    def _fake_urlopen(
        request: Request, *, timeout: float, context: ssl.SSLContext
    ) -> _FakeResponse:
        _ = request
        _ = timeout
        assert context is not None
        return _FakeResponse(b"abcdefghijk")

    with (
        patch("mother.tools.web_fetch_tool.MAX_CONTENT_BYTES", 5),
        patch("mother.tools.web_fetch_tool.urllib.request.urlopen", side_effect=_fake_urlopen),
    ):
        tool = make_web_fetch_tool()
        output = tool("https://example.com/api", mode="raw")

    assert "abcde" in output
    assert "[Content truncated due to size limit]" in output


def test_web_fetch_tool_truncates_jina_response_when_size_limit_is_exceeded():
    def _fake_urlopen(
        request: Request, *, timeout: float, context: ssl.SSLContext
    ) -> _FakeResponse:
        _ = request
        _ = timeout
        assert context is not None
        return _FakeResponse(b"abcdefghijk")

    with (
        patch("mother.tools.web_fetch_tool.MAX_CONTENT_BYTES", 5),
        patch("mother.tools.web_fetch_tool.urllib.request.urlopen", side_effect=_fake_urlopen),
    ):
        tool = make_web_fetch_tool()
        output = tool("https://example.com/docs", mode="jina")

    assert "abcde" in output
    assert "[Content truncated due to size limit]" in output


def test_web_fetch_tool_rejects_invalid_headers_json():
    tool = make_web_fetch_tool()
    output = tool("https://example.com/api", mode="raw", headers_json="[]")
    assert output == "Error: Invalid headers_json: expected a JSON object."


def test_web_fetch_tool_rejects_local_jina_mode():
    tool = make_web_fetch_tool()
    output = tool("http://localhost:8000/health", mode="jina")
    assert output == "Error: jina mode cannot fetch local URLs. Use mode=raw instead."


def test_web_fetch_tool_registered_in_registry():
    registry = get_default_tools(tools_enabled=True)
    assert not registry.is_empty()
    tool_names = [getattr(tool, "__name__", "") for tool in registry.tools()]
    assert "bash" in tool_names
    assert "web_search" in tool_names
    assert "web_fetch" in tool_names
