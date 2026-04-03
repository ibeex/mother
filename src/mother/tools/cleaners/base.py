"""Shared cleaner types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import ParseResult


@dataclass(frozen=True, slots=True)
class ContentCleaner:
    id: str
    matches: Callable[[ParseResult], bool]
    clean: Callable[[str], str]
