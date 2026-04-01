"""Shared helpers for web search and fetch tools."""

from __future__ import annotations

import json
import ssl
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

DEFAULT_TIMEOUT = 30.0
DEFAULT_PASS_PATH = "api/jina"
JINA_SEARCH_URL = "https://s.jina.ai/"
JINA_READER_URL = "https://r.jina.ai/"
DEFAULT_USER_AGENT = "mother/1.0"


class ReadableResponse(Protocol):
    status: int
    headers: Mapping[str, str]

    def read(self, amount: int = -1) -> bytes: ...


class ResponseContext(Protocol):
    def __enter__(self) -> ReadableResponse: ...
    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool | None: ...


class FetchResultLike(Protocol):
    @property
    def url(self) -> str: ...

    @property
    def mode(self) -> str: ...

    @property
    def content(self) -> str: ...

    @property
    def status(self) -> int | None: ...

    @property
    def content_type(self) -> str | None: ...


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


def build_ssl_context(ca_bundle_path: str = "") -> ssl.SSLContext:
    """Build an SSL context using system roots and an optional configured CA bundle.

    Python/OpenSSL can reject some enterprise interception certificates more
    strictly than curl, especially when the corporate CA omits extensions such as
    Authority Key Identifier. To stay compatible with those environments while
    still verifying certificates, this context disables OpenSSL's strict X.509
    verification flag when it is available.

    If ``ca_bundle_path`` is empty, only the default system trust store is used.
    If it is set, the referenced CA bundle is added to the context.
    """
    context = ssl.create_default_context()
    verify_x509_strict = getattr(ssl, "VERIFY_X509_STRICT", 0)
    if verify_x509_strict:
        context.verify_flags &= ~verify_x509_strict

    normalized_ca_bundle_path = ca_bundle_path.strip()
    if not normalized_ca_bundle_path:
        return context

    ca_bundle = Path(normalized_ca_bundle_path).expanduser()
    if not ca_bundle.is_file():
        raise RuntimeError(f"Configured CA bundle was not found: {ca_bundle}")

    context.load_verify_locations(cafile=str(ca_bundle))
    return context


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


def format_fetch_error(exc: Exception) -> str:
    """Return a readable error message for HTTP fetch failures."""
    if isinstance(exc, HTTPError):
        detail = exc.read().decode("utf-8", errors="replace").strip()
        if detail:
            return f"Error: HTTP {exc.code} - {exc.reason}\n{detail}"
        return f"Error: HTTP {exc.code} - {exc.reason}"
    if isinstance(exc, URLError):
        return f"Error: {exc.reason}"
    return f"Error: {exc}"


def fetch_result_metadata_lines(
    result: FetchResultLike,
    *,
    url_first: bool = False,
) -> tuple[str, ...]:
    """Return normalized metadata lines shared by fetch formatting callers."""
    fields = [
        ("Mode", result.mode),
        ("URL", result.url),
    ]
    if url_first:
        fields = [fields[1], fields[0]]
    if result.status is not None:
        fields.append(("Status", str(result.status)))
    if result.content_type is not None:
        fields.append(("Content-Type", result.content_type))
    return tuple(f"{label}: {value}" for label, value in fields)
