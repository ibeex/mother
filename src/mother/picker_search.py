"""Reusable fuzzy matching helpers for picker-style searches."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class PickerSearchField:
    """A searchable text field for picker matching."""

    text: str
    primary: bool = False


@dataclass(frozen=True, slots=True, order=True)
class _MatchRank:
    """Sort key describing how well a field matched a picker query."""

    kind: int
    field_priority: int
    start: int
    gaps: int
    span: int
    length: int


def filter_picker_items[T](
    items: Iterable[T],
    query: str,
    fields: Callable[[T], Iterable[PickerSearchField]],
) -> list[T]:
    """Return items whose searchable fields match the query.

    Matching prefers exact and prefix matches, then substring matches, then
    fuzzy subsequence matches such as ``lo3`` matching ``local_3``.
    """
    collected_items = list(items)
    stripped_query = query.strip()
    if not stripped_query:
        return collected_items

    folded_query = stripped_query.casefold()
    ranked_matches: list[tuple[_MatchRank, int, T]] = []
    for index, item in enumerate(collected_items):
        best_rank: _MatchRank | None = None
        for field in fields(item):
            rank = _rank_match(folded_query, field)
            if rank is None:
                continue
            if best_rank is None or rank < best_rank:
                best_rank = rank
        if best_rank is not None:
            ranked_matches.append((best_rank, index, item))

    ranked_matches.sort(key=lambda match: (match[0], match[1]))
    return [item for _, _, item in ranked_matches]


def _rank_match(folded_query: str, field: PickerSearchField) -> _MatchRank | None:
    """Return a rank for a single searchable field, if it matches."""
    folded_text = field.text.casefold()
    if not folded_text:
        return None

    field_priority = 0 if field.primary else 1
    if folded_text == folded_query:
        return _MatchRank(0, field_priority, 0, 0, len(folded_query), len(folded_text))
    if folded_text.startswith(folded_query):
        return _MatchRank(1, field_priority, 0, 0, len(folded_query), len(folded_text))

    substring_index = folded_text.find(folded_query)
    if substring_index >= 0:
        return _MatchRank(
            2,
            field_priority,
            substring_index,
            0,
            len(folded_query),
            len(folded_text),
        )

    subsequence_rank = _subsequence_rank(folded_query, folded_text)
    if subsequence_rank is None:
        return None
    start, gaps, span = subsequence_rank
    return _MatchRank(3, field_priority, start, gaps, span, len(folded_text))


def _subsequence_rank(query: str, text: str) -> tuple[int, int, int] | None:
    """Return the start, total gaps, and span for a subsequence match."""
    positions: list[int] = []
    start_index = 0
    for character in query:
        position = text.find(character, start_index)
        if position < 0:
            return None
        positions.append(position)
        start_index = position + 1

    start = positions[0]
    gaps = sum(current - previous - 1 for previous, current in pairwise(positions))
    span = positions[-1] - start + 1
    return start, gaps, span
