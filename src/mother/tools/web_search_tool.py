"""Web search tool powered by the Jina Search API."""

from __future__ import annotations

import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import cast
from urllib.error import HTTPError, URLError

from mother.tools.web_common import (
    DEFAULT_PASS_PATH,
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
    JINA_SEARCH_URL,
    ResponseContext,
    get_jina_api_key,
)


def _build_search_request(query: str, api_key: str) -> urllib.request.Request:
    encoded_query = urllib.parse.quote(query)
    url = f"{JINA_SEARCH_URL}?q={encoded_query}"
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "X-Respond-With": "no-content",
        "Accept": "text/plain",
    }
    headers["Authorization"] = f"Bearer {api_key}"
    return urllib.request.Request(url, headers=headers)


def _run_search_request(query: str, timeout: float, api_key: str) -> str:
    request = _build_search_request(query, api_key)
    response_context = cast(
        ResponseContext,
        urllib.request.urlopen(request, timeout=timeout),
    )
    with response_context as response:
        return response.read().decode("utf-8", errors="replace").strip()


def make_web_search_tool(pass_path: str = DEFAULT_PASS_PATH) -> Callable[..., str]:
    """Factory returning a callable llm tool for web search."""

    def web_search(query: str, timeout: float = DEFAULT_TIMEOUT) -> str:
        """Search the public web by query.

        Use this when you do not yet know the exact URL and need to discover sources,
        pages, or recent public information. If you already have a URL, prefer
        web_fetch instead.

        Args:
            query: Search query string describing what to look for.
            timeout: Network timeout in seconds.

        Returns:
            Plain-text search results or a readable error message.
        """
        normalized_query = query.strip()
        if not normalized_query:
            return "Error: query must not be empty."

        try:
            api_key = get_jina_api_key(pass_path)
            content = _run_search_request(normalized_query, timeout, api_key)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            if detail:
                return f"Error: HTTP {exc.code} - {exc.reason}\n{detail}"
            return f"Error: HTTP {exc.code} - {exc.reason}"
        except RuntimeError as exc:
            return f"Error: {exc}"
        except URLError as exc:
            return f"Error: {exc.reason}"
        except Exception as exc:
            return f"Error: {exc}"

        if not content:
            return "No search results found. Try a different query."

        return "\n".join(["## Search Results", "", content])

    return web_search
