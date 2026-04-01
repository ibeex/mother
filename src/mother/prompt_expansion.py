"""Prompt preprocessing helpers for explicit inline web fetch expansion."""

from __future__ import annotations

import re
from dataclasses import dataclass

from mother.tools.web_common import fetch_result_metadata_lines, format_fetch_error
from mother.tools.web_fetch_tool import FetchResult, fetch_url

_FETCH_DIRECTIVE_PATTERN = re.compile(
    r"\[\[\s*fetch\s+(?P<url>https?://[^\s\]]+)\s*\]\]",
    re.IGNORECASE,
)
_MAX_FETCH_DIRECTIVES = 3
_MAX_FETCH_CHARS_PER_SOURCE = 12_000
_MAX_FETCH_CHARS_TOTAL = 24_000
_PROMPT_TRUNCATED_MARKER = "[Content truncated for prompt context]"


@dataclass(frozen=True, slots=True)
class PromptExpansionResult:
    prompt_text: str
    fetched_urls: tuple[str, ...] = ()


def _extract_fetch_urls(text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    urls: list[str] = []
    for match in _FETCH_DIRECTIVE_PATTERN.finditer(text):
        url = match.group("url").strip()
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return tuple(urls)


def _replace_fetch_directives_with_urls(text: str) -> str:
    replaced = _FETCH_DIRECTIVE_PATTERN.sub(lambda match: match.group("url").strip(), text)
    return replaced.strip()


def _truncate_for_prompt(content: str, limit: int) -> str:
    normalized = content.strip()
    if limit <= 0:
        return _PROMPT_TRUNCATED_MARKER
    if len(normalized) <= limit:
        return normalized
    truncated = normalized[:limit].rstrip()
    if truncated:
        return f"{truncated}\n{_PROMPT_TRUNCATED_MARKER}"
    return _PROMPT_TRUNCATED_MARKER


def _format_fetch_result_for_prompt(result: FetchResult, *, content_limit: int) -> str:
    lines = list(fetch_result_metadata_lines(result, url_first=True))
    lines.extend(["", _truncate_for_prompt(result.content, content_limit)])
    return "\n".join(lines)


def expand_prompt_fetch_directives(text: str, *, ca_bundle_path: str = "") -> PromptExpansionResult:
    """Expand ``[[fetch ...]]`` directives into fetched web context."""
    fetch_urls = _extract_fetch_urls(text)
    if not fetch_urls:
        return PromptExpansionResult(prompt_text=text)

    cleaned_text = _replace_fetch_directives_with_urls(text)
    sections: list[str] = [
        "Fetched web content explicitly requested by the user:",
    ]

    remaining_chars = _MAX_FETCH_CHARS_TOTAL
    limited_urls = fetch_urls[:_MAX_FETCH_DIRECTIVES]
    for index, url in enumerate(limited_urls, start=1):
        content_limit = min(_MAX_FETCH_CHARS_PER_SOURCE, remaining_chars)
        if content_limit <= 0:
            sections.append(
                "Additional fetched content was omitted because the prompt context limit was reached."
            )
            break

        try:
            result = fetch_url(url, ca_bundle_path=ca_bundle_path)
            formatted_result = _format_fetch_result_for_prompt(result, content_limit=content_limit)
            remaining_chars -= min(len(formatted_result), content_limit)
        except Exception as exc:
            formatted_result = f"URL: {url}\n\n{format_fetch_error(exc)}"
            remaining_chars -= content_limit

        sections.extend(["", f"Source {index}", formatted_result])

    ignored_count = len(fetch_urls) - len(limited_urls)
    if ignored_count > 0:
        sections.extend(
            [
                "",
                f"Note: ignored {ignored_count} additional [[fetch ...]] directive(s) after the first {_MAX_FETCH_DIRECTIVES}.",
            ]
        )

    expanded_prompt = "\n".join(sections)
    if cleaned_text:
        expanded_prompt = f"{expanded_prompt}\n\nUser prompt:\n{cleaned_text}"

    return PromptExpansionResult(prompt_text=expanded_prompt, fetched_urls=limited_urls)
