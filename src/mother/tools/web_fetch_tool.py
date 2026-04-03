"""HTTP fetch tool with raw urllib and Jina reader modes."""

from __future__ import annotations

import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, cast
from urllib.error import HTTPError
from urllib.parse import urlparse

from mother.tools.cleaners import clean_fetched_body
from mother.tools.web_common import (
    DEFAULT_PASS_PATH,
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
    JINA_READER_URL,
    ReadableResponse,
    ResponseContext,
    build_ssl_context,
    fetch_result_metadata_lines,
    format_fetch_error,
    get_jina_api_key,
    is_local_url,
    parse_headers_json,
    should_retry_with_jina_api_key,
)

FetchMode = Literal["auto", "raw", "jina"]


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    mode: FetchMode
    content: str
    status: int | None = None
    content_type: str | None = None


MAX_TIMEOUT = 120.0
MAX_CONTENT_BYTES = 512_000
_CONTENT_TRUNCATED_MARKER = "[Content truncated due to size limit]"
_CLOUDFLARE_FORBIDDEN = 403
_BROWSER_LIKE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _validate_url(url: str) -> str:
    normalized = url.strip()
    if not normalized:
        raise ValueError("url must not be empty.")

    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("url must use http:// or https://.")
    if not parsed.netloc:
        raise ValueError("url must include a hostname.")
    return normalized


def _resolve_timeout(timeout: float) -> float:
    if timeout <= 0:
        raise ValueError("timeout must be a positive number.")
    return min(timeout, MAX_TIMEOUT)


def _resolve_mode(
    url: str,
    mode: FetchMode,
    method: str,
    headers_json: str,
    body: str,
) -> FetchMode:
    if mode not in {"auto", "raw", "jina"}:
        raise ValueError("mode must be one of: auto, raw, jina.")

    if mode != "auto":
        return mode

    if method != "GET" or headers_json.strip() or body:
        return "raw"
    if is_local_url(url):
        return "raw"
    return "jina"


def _build_raw_headers(headers: Mapping[str, str], *, honest_user_agent: bool) -> dict[str, str]:
    request_headers = {
        "User-Agent": _BROWSER_LIKE_USER_AGENT,
        "Accept": (
            "application/json,text/plain,text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        **headers,
    }
    if honest_user_agent:
        request_headers["User-Agent"] = DEFAULT_USER_AGENT
    return request_headers


def _build_raw_request(
    url: str,
    method: str,
    headers: Mapping[str, str],
    body: str,
    *,
    honest_user_agent: bool,
) -> urllib.request.Request:
    data = body.encode("utf-8") if body else None
    request_headers = _build_raw_headers(headers, honest_user_agent=honest_user_agent)
    return urllib.request.Request(url, headers=request_headers, data=data, method=method)


def _open_request(
    request: urllib.request.Request, timeout: float, ca_bundle_path: str
) -> ResponseContext:
    ssl_context = build_ssl_context(ca_bundle_path)
    return cast(
        ResponseContext,
        urllib.request.urlopen(request, timeout=timeout, context=ssl_context),
    )


def _header_value(headers: object, name: str) -> str:
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return ""
    value = getter(name, "")
    if isinstance(value, str):
        return value
    return str(value)


def _read_response_content(response: ReadableResponse) -> str:
    content = response.read(MAX_CONTENT_BYTES + 1)
    content_bytes = content[:MAX_CONTENT_BYTES]
    body_text = content_bytes.decode("utf-8", errors="replace").strip()
    if len(content) > MAX_CONTENT_BYTES:
        if body_text:
            return f"{body_text}\n{_CONTENT_TRUNCATED_MARKER}"
        return _CONTENT_TRUNCATED_MARKER
    return body_text or "(empty response body)"


def _preprocess_fetched_body(url: str, body: str) -> str:
    cleaned_body = clean_fetched_body(url, body).strip()
    return cleaned_body or "(empty response body)"


def _should_retry_raw_with_honest_user_agent(exc: HTTPError) -> bool:
    if exc.code != _CLOUDFLARE_FORBIDDEN:
        return False
    return _header_value(exc.headers, "cf-mitigated").lower() == "challenge"


def _run_raw_request(
    url: str,
    method: str,
    headers: Mapping[str, str],
    body: str,
    timeout: float,
    ca_bundle_path: str,
) -> FetchResult:
    request = _build_raw_request(
        url,
        method,
        headers,
        body,
        honest_user_agent=False,
    )

    try:
        response_context = _open_request(request, timeout, ca_bundle_path)
    except HTTPError as exc:
        if not _should_retry_raw_with_honest_user_agent(exc):
            raise
        fallback_request = _build_raw_request(
            url,
            method,
            headers,
            body,
            honest_user_agent=True,
        )
        response_context = _open_request(fallback_request, timeout, ca_bundle_path)

    with response_context as response:
        content_type = _header_value(response.headers, "Content-Type") or "unknown"
        body_text = _preprocess_fetched_body(url, _read_response_content(response))
        return FetchResult(
            url=url,
            mode="raw",
            content=body_text,
            status=response.status,
            content_type=content_type,
        )


def _build_jina_reader_request(url: str, api_key: str | None = None) -> urllib.request.Request:
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "text/plain",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    return urllib.request.Request(f"{JINA_READER_URL}{url}", headers=headers)


def _run_jina_request(
    url: str, timeout: float, ca_bundle_path: str, api_key: str | None = None
) -> str:
    request = _build_jina_reader_request(url, api_key)
    response_context = _open_request(request, timeout, ca_bundle_path)
    with response_context as response:
        return _preprocess_fetched_body(url, _read_response_content(response))


def fetch_url(
    url: str,
    *,
    mode: FetchMode = "auto",
    method: str = "GET",
    headers_json: str = "",
    body: str = "",
    timeout: float = DEFAULT_TIMEOUT,
    pass_path: str = DEFAULT_PASS_PATH,
    ca_bundle_path: str = "",
) -> FetchResult:
    """Fetch a URL and return structured content for reuse outside tool mode."""
    normalized_method = method.strip().upper() or "GET"
    normalized_url = _validate_url(url)
    resolved_timeout = _resolve_timeout(timeout)
    resolved_mode = _resolve_mode(
        normalized_url,
        mode,
        normalized_method,
        headers_json,
        body,
    )
    if resolved_mode == "jina" and is_local_url(normalized_url):
        raise ValueError("jina mode cannot fetch local URLs. Use mode=raw instead.")

    if resolved_mode == "raw":
        headers = parse_headers_json(headers_json)
        return _run_raw_request(
            normalized_url,
            normalized_method,
            headers,
            body,
            resolved_timeout,
            ca_bundle_path,
        )

    try:
        content = _run_jina_request(
            normalized_url,
            resolved_timeout,
            ca_bundle_path,
        )
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        if not should_retry_with_jina_api_key(exc.code, detail):
            raise

        api_key = get_jina_api_key(pass_path)
        content = _run_jina_request(
            normalized_url,
            resolved_timeout,
            ca_bundle_path,
            api_key,
        )

    return FetchResult(
        url=normalized_url,
        mode="jina",
        content=content,
    )


def _format_fetch_result(result: FetchResult) -> str:
    metadata_lines = [f"- {line}" for line in fetch_result_metadata_lines(result)]
    lines = ["## Fetch Result", "", *metadata_lines, "", result.content]
    return "\n".join(lines)


def make_web_fetch_tool(
    pass_path: str = DEFAULT_PASS_PATH, ca_bundle_path: str = ""
) -> Callable[..., str]:
    """Factory returning a callable llm tool for HTTP fetches."""

    def web_fetch(
        url: str,
        mode: FetchMode = "auto",
        method: str = "GET",
        headers_json: str = "",
        body: str = "",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> str:
        """Fetch a web page or HTTP endpoint.

        Use this when you already know the URL and want to open it.

        Choosing a mode:
        - mode="auto": best default. Uses raw HTTP for localhost, APIs, custom headers,
          non-GET methods, or request bodies. Uses jina for normal public web pages.
        - mode="raw": direct urllib request. Best for APIs, localhost, JSON endpoints,
          POST requests, custom headers, or exact HTTP behavior.
        - mode="jina": readable page fetch for public websites. Best for articles,
          documentation pages, and general web content you want converted to plain text.
          Do not use jina mode for localhost or private local services.

        Args:
            url: Required HTTP or HTTPS URL.
            mode: One of "auto", "raw", or "jina".
            method: HTTP method for raw requests, such as GET or POST.
            headers_json: Optional JSON object string of request headers. Example:
                '{"Accept": "application/json", "Authorization": "Bearer ..."}'
            body: Optional request body string for raw requests.
            timeout: Network timeout in seconds. Values above 120 seconds are capped.

        Returns:
            Retrieved content formatted for chat, or a readable error message.
        """
        try:
            result = fetch_url(
                url,
                mode=mode,
                method=method,
                headers_json=headers_json,
                body=body,
                timeout=timeout,
                pass_path=pass_path,
                ca_bundle_path=ca_bundle_path,
            )
        except Exception as exc:
            return format_fetch_error(exc)
        return _format_fetch_result(result)

    return web_fetch
