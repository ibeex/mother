"""Site-specific cleanup helpers for fetched web content."""

from __future__ import annotations

from urllib.parse import urlparse

from mother.tools.cleaners.base import ContentCleaner
from mother.tools.cleaners.hacker_news import hacker_news_cleaner

# Add new cleaners here.
_CLEANERS: tuple[ContentCleaner, ...] = (hacker_news_cleaner,)

__all__ = ["ContentCleaner", "clean_fetched_body"]


def clean_fetched_body(url: str, body: str) -> str:
    """Apply site-specific cleanup to fetched body text when available."""
    try:
        parsed_url = urlparse(url)
    except ValueError:
        return body

    for cleaner in _CLEANERS:
        if cleaner.matches(parsed_url):
            return cleaner.clean(body)
    return body
