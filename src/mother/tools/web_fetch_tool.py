"""HTTP fetch tool with raw urllib and Jina reader modes."""

from __future__ import annotations

import urllib.request
from collections.abc import Callable, Mapping
from typing import Literal, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from mother.tools.web_common import (
    DEFAULT_PASS_PATH,
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
    JINA_READER_URL,
    ResponseContext,
    build_ssl_context,
    get_jina_api_key,
    is_local_url,
    parse_headers_json,
    should_retry_with_jina_api_key,
)

FetchMode = Literal["auto", "raw", "jina"]


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


def _run_raw_request(
    url: str,
    method: str,
    headers: Mapping[str, str],
    body: str,
    timeout: float,
    ca_bundle_path: str,
) -> str:
    request_headers = {"User-Agent": DEFAULT_USER_AGENT, **headers}
    data = body.encode("utf-8") if body else None
    request = urllib.request.Request(url, headers=request_headers, data=data, method=method)
    ssl_context = build_ssl_context(ca_bundle_path)
    response_context = cast(
        ResponseContext,
        urllib.request.urlopen(request, timeout=timeout, context=ssl_context),
    )
    with response_context as response:
        content_type = response.headers.get("Content-Type", "unknown")
        content = response.read().decode("utf-8", errors="replace").strip()
        body_text = content or "(empty response body)"
        return "\n".join(
            [
                "## Fetch Result",
                "",
                "- Mode: raw",
                f"- URL: {url}",
                f"- Status: {response.status}",
                f"- Content-Type: {content_type}",
                "",
                body_text,
            ]
        )


def _build_jina_reader_request(url: str, api_key: str | None = None) -> urllib.request.Request:
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "text/plain",
    }
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    return urllib.request.Request(f"{JINA_READER_URL}{url}", headers=headers)


def _run_jina_request(
    url: str, timeout: float, ca_bundle_path: str, api_key: str | None = None
) -> str:
    request = _build_jina_reader_request(url, api_key)
    ssl_context = build_ssl_context(ca_bundle_path)
    response_context = cast(
        ResponseContext,
        urllib.request.urlopen(request, timeout=timeout, context=ssl_context),
    )
    with response_context as response:
        return response.read().decode("utf-8", errors="replace").strip()


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
            timeout: Network timeout in seconds.

        Returns:
            Retrieved content formatted for chat, or a readable error message.
        """
        normalized_method = method.strip().upper() or "GET"
        try:
            normalized_url = _validate_url(url)
            resolved_mode = _resolve_mode(
                normalized_url,
                mode,
                normalized_method,
                headers_json,
                body,
            )
            if resolved_mode == "jina" and is_local_url(normalized_url):
                return "Error: jina mode cannot fetch local URLs. Use mode=raw instead."

            if resolved_mode == "raw":
                headers = parse_headers_json(headers_json)
                return _run_raw_request(
                    normalized_url,
                    normalized_method,
                    headers,
                    body,
                    timeout,
                    ca_bundle_path,
                )

            try:
                content = _run_jina_request(normalized_url, timeout, ca_bundle_path)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace").strip()
                if not should_retry_with_jina_api_key(exc.code, detail):
                    if detail:
                        return f"Error: HTTP {exc.code} - {exc.reason}\n{detail}"
                    return f"Error: HTTP {exc.code} - {exc.reason}"

                api_key = get_jina_api_key(pass_path)
                content = _run_jina_request(normalized_url, timeout, ca_bundle_path, api_key)

            return "\n".join(
                [
                    "## Fetch Result",
                    "",
                    "- Mode: jina",
                    f"- URL: {normalized_url}",
                    "",
                    content or "(empty response body)",
                ]
            )
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            if detail:
                return f"Error: HTTP {exc.code} - {exc.reason}\n{detail}"
            return f"Error: HTTP {exc.code} - {exc.reason}"
        except URLError as exc:
            return f"Error: {exc.reason}"
        except (RuntimeError, ValueError) as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error: {exc}"

    return web_fetch
