"""Tests for the Jina-powered web search tool."""

from __future__ import annotations

import io
import subprocess
from email.message import Message
from typing import final
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from mother.tools import get_default_tools
from mother.tools.web_search_tool import make_web_search_tool


@final
class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self._content: bytes = content

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        _ = exc_type
        _ = exc
        _ = tb
        return False

    def read(self) -> bytes:
        return self._content


def test_web_search_tool_uses_pass_and_returns_search_results():
    captured_requests: list[Request] = []

    def _fake_urlopen(request: Request, *, timeout: float) -> _FakeResponse:
        captured_requests.append(request)
        assert timeout == 30.0
        return _FakeResponse(b"Result one\nResult two")

    completed = subprocess.CompletedProcess(
        args=["pass", "api/jina"], returncode=0, stdout="secret-key\n"
    )

    with (
        patch("mother.tools.web_search_tool.subprocess.run", return_value=completed),
        patch("mother.tools.web_search_tool.urllib.request.urlopen", side_effect=_fake_urlopen),
    ):
        tool = make_web_search_tool()
        output = tool("Jina AI")

    assert output.startswith("## Search Results")
    assert "Result one" in output
    assert "Result two" in output
    assert len(captured_requests) == 1
    captured_request = captured_requests[0]
    assert captured_request.full_url == "https://s.jina.ai/?q=Jina%20AI"
    assert captured_request.headers["Authorization"] == "Bearer secret-key"
    assert captured_request.headers["X-respond-with"] == "no-content"
    assert captured_request.headers["Accept"] == "text/plain"


def test_web_search_tool_handles_missing_pass_command():
    with patch(
        "mother.tools.web_search_tool.subprocess.run",
        side_effect=FileNotFoundError("pass not found"),
    ):
        tool = make_web_search_tool()
        output = tool("test")

    assert output == "Error: `pass` command is not installed or not available in PATH."


def test_web_search_tool_handles_http_error():
    error = HTTPError(
        url="https://s.jina.ai/?q=test",
        code=401,
        msg="Unauthorized",
        hdrs=Message(),
        fp=io.BytesIO(b"bad key"),
    )
    completed = subprocess.CompletedProcess(
        args=["pass", "api/jina"], returncode=0, stdout="secret-key\n"
    )

    with (
        patch("mother.tools.web_search_tool.subprocess.run", return_value=completed),
        patch("mother.tools.web_search_tool.urllib.request.urlopen", side_effect=error),
    ):
        tool = make_web_search_tool()
        output = tool("test")

    assert output == "Error: HTTP 401 - Unauthorized\nbad key"


def test_web_search_tool_handles_network_error():
    completed = subprocess.CompletedProcess(
        args=["pass", "api/jina"], returncode=0, stdout="secret-key\n"
    )

    with (
        patch("mother.tools.web_search_tool.subprocess.run", return_value=completed),
        patch(
            "mother.tools.web_search_tool.urllib.request.urlopen",
            side_effect=URLError("network unreachable"),
        ),
    ):
        tool = make_web_search_tool()
        output = tool("test")

    assert output == "Error: network unreachable"


def test_web_search_tool_requires_non_empty_query():
    tool = make_web_search_tool()
    assert tool("   ") == "Error: query must not be empty."


def test_web_search_tool_registered_in_registry():
    registry = get_default_tools(tools_enabled=True)
    assert not registry.is_empty()
    tool_names = [getattr(tool, "__name__", "") for tool in registry.tools()]
    assert "bash" in tool_names
    assert "web_search" in tool_names
