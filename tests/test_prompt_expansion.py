"""Tests for explicit inline web fetch prompt expansion."""

from __future__ import annotations

from unittest.mock import patch

from mother.prompt_expansion import PromptExpansionResult, expand_prompt_fetch_directives
from mother.tools.web_fetch_tool import FetchResult


def test_expand_prompt_fetch_directives_leaves_plain_text_unchanged() -> None:
    result = expand_prompt_fetch_directives("hello")

    assert result == PromptExpansionResult(prompt_text="hello")


def test_expand_prompt_fetch_directives_injects_fetched_content() -> None:
    with patch(
        "mother.prompt_expansion.fetch_url",
        return_value=FetchResult(
            url="https://example.com/docs",
            mode="jina",
            content="Example content",
        ),
    ):
        result = expand_prompt_fetch_directives(
            "Summarize this page: [[fetch https://example.com/docs]]"
        )

    assert result.fetched_urls == ("https://example.com/docs",)
    assert "Fetched web content explicitly requested by the user:" in result.prompt_text
    assert "Source 1" in result.prompt_text
    assert "URL: https://example.com/docs" in result.prompt_text
    assert "Mode: jina" in result.prompt_text
    assert "Example content" in result.prompt_text
    assert "User prompt:" in result.prompt_text
    assert "Summarize this page: https://example.com/docs" in result.prompt_text


def test_expand_prompt_fetch_directives_formats_fetch_failures_inline() -> None:
    with patch(
        "mother.prompt_expansion.fetch_url",
        side_effect=RuntimeError("boom"),
    ):
        result = expand_prompt_fetch_directives("[[fetch https://example.com/docs]]")

    assert result.fetched_urls == ("https://example.com/docs",)
    assert "Source 1" in result.prompt_text
    assert "URL: https://example.com/docs" in result.prompt_text
    assert "Error: boom" in result.prompt_text


def test_expand_prompt_fetch_directives_counts_failed_fetches_against_total_budget() -> None:
    long_content = "x" * 20_000
    with patch(
        "mother.prompt_expansion.fetch_url",
        side_effect=[
            RuntimeError("boom"),
            FetchResult(
                url="https://example.com/two",
                mode="jina",
                content=long_content,
            ),
            FetchResult(
                url="https://example.com/three",
                mode="jina",
                content="should not be fetched",
            ),
        ],
    ):
        result = expand_prompt_fetch_directives(
            " ".join(
                [
                    "[[fetch https://example.com/one]]",
                    "[[fetch https://example.com/two]]",
                    "[[fetch https://example.com/three]]",
                ]
            )
        )

    assert result.fetched_urls == (
        "https://example.com/one",
        "https://example.com/two",
        "https://example.com/three",
    )
    assert "Source 1" in result.prompt_text
    assert "Source 2" in result.prompt_text
    assert "Source 3" not in result.prompt_text
    assert "should not be fetched" not in result.prompt_text
    assert "Additional fetched content was omitted" in result.prompt_text
