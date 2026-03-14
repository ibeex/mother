"""Shared helpers for web search and fetch tools."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from typing import Protocol, cast
from urllib.parse import urlparse

DEFAULT_TIMEOUT = 30.0
DEFAULT_PASS_PATH = "api/jina"
JINA_SEARCH_URL = "https://s.jina.ai/"
JINA_READER_URL = "https://r.jina.ai/"
DEFAULT_USER_AGENT = "mother/1.0"


class ReadableResponse(Protocol):
    status: int
    headers: Mapping[str, str]

    def read(self) -> bytes: ...


class ResponseContext(Protocol):
    def __enter__(self) -> ReadableResponse: ...
    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool | None: ...


def get_jina_api_key(pass_path: str = DEFAULT_PASS_PATH) -> str:
    """Load the Jina API key from the local password store."""
    try:
        completed = subprocess.run(
            ["pass", pass_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("`pass` command is not installed or not available in PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raw_stderr = cast(object, exc.stderr)
        stderr = raw_stderr.strip() if isinstance(raw_stderr, str) else ""
        message = stderr or f"`pass {pass_path}` failed with exit code {exc.returncode}."
        raise RuntimeError(message) from exc

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"`pass {pass_path}` returned no secret.")
    return lines[0]


def should_retry_with_jina_api_key(status_code: int, detail: str) -> bool:
    """Return whether a Jina request should be retried with the API key."""
    normalized_detail = detail.lower()
    auth_required_markers = (
        "authentication is required",
        "authenticationrequirederror",
        "provide a valid api key",
        "authorization header",
    )
    return (
        status_code in {401, 429}
        or "rate limit" in normalized_detail
        or "too many requests" in normalized_detail
        or any(marker in normalized_detail for marker in auth_required_markers)
    )


def is_local_url(url: str) -> bool:
    """Return whether the URL points at the local machine."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        return False
    return hostname in {"localhost", "127.0.0.1", "::1"}


def parse_headers_json(headers_json: str) -> dict[str, str]:
    """Parse a JSON object containing HTTP headers."""
    normalized = headers_json.strip()
    if not normalized:
        return {}

    try:
        payload = cast(object, json.loads(normalized))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid headers_json: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Invalid headers_json: expected a JSON object.")

    raw_headers = cast(dict[object, object], payload)
    headers: dict[str, str] = {}
    for raw_key, raw_value in raw_headers.items():
        if not isinstance(raw_key, str):
            raise ValueError("Invalid headers_json: header names must be strings.")
        if not isinstance(raw_value, str):
            raise ValueError("Invalid headers_json: header values must be strings.")
        headers[raw_key] = raw_value
    return headers
