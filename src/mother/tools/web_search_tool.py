"""Web search tool powered by the Jina Search API."""

from __future__ import annotations

import subprocess
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Protocol, cast
from urllib.error import HTTPError, URLError

JINA_SEARCH_URL = "https://s.jina.ai/"
DEFAULT_TIMEOUT = 30.0
DEFAULT_PASS_PATH = "api/jina"


class _ReadableResponse(Protocol):
    def read(self) -> bytes: ...


class _ResponseContext(Protocol):
    def __enter__(self) -> _ReadableResponse: ...
    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool | None: ...


def _get_jina_api_key(pass_path: str = DEFAULT_PASS_PATH) -> str:
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


def _build_search_request(query: str, api_key: str) -> urllib.request.Request:
    encoded_query = urllib.parse.quote(query)
    url = f"{JINA_SEARCH_URL}?q={encoded_query}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "mother/1.0",
        "X-Respond-With": "no-content",
        "Accept": "text/plain",
    }
    return urllib.request.Request(url, headers=headers)


def make_web_search_tool(pass_path: str = DEFAULT_PASS_PATH) -> Callable[..., str]:
    """Factory returning a callable llm tool for web search."""

    def web_search(query: str, timeout: float = DEFAULT_TIMEOUT) -> str:
        """Search the web for public information.

        Args:
            query: Search query string.
            timeout: Network timeout in seconds.

        Returns:
            Plain-text Jina search results or a readable error message.
        """
        normalized_query = query.strip()
        if not normalized_query:
            return "Error: query must not be empty."

        try:
            api_key = _get_jina_api_key(pass_path)
            request = _build_search_request(normalized_query, api_key)
            response_context = cast(
                _ResponseContext,
                urllib.request.urlopen(request, timeout=timeout),
            )
            with response_context as response:
                content = response.read().decode("utf-8", errors="replace").strip()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            if detail:
                return f"Error: HTTP {exc.code} - {exc.reason}\n{detail}"
            return f"Error: HTTP {exc.code} - {exc.reason}"
        except URLError as exc:
            return f"Error: {exc.reason}"
        except RuntimeError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error: {exc}"

        if not content:
            return "No search results found. Try a different query."

        return "\n".join(["## Search Results", "", content])

    return web_search
