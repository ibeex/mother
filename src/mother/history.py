"""Persistent prompt-history storage for Mother."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import TypedDict, cast

from mother.picker_search import PickerSearchField, filter_picker_items

DEFAULT_PROMPT_HISTORY_FILE = Path.home() / ".mother" / "prompt_history.jsonl"


class PromptHistoryEntry(TypedDict):
    """A single prompt-history record persisted to JSONL."""

    input: str
    ts: str


@dataclass(frozen=True, slots=True)
class PromptHistoryMatch:
    """A searchable prompt-history entry addressed by reverse index."""

    index: int
    text: str


def default_prompt_history_path() -> Path:
    """Return the default JSONL path used for prompt history."""
    return DEFAULT_PROMPT_HISTORY_FILE


@dataclass(slots=True)
class PromptHistory:
    """Append-only prompt history with simple reverse search support."""

    path: Path = field(default_factory=default_prompt_history_path)
    _entries: list[str] = field(default_factory=list, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)
            entries: list[str] = []
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    candidate = line.strip()
                    if not candidate:
                        continue
                    try:
                        loaded = cast(object, json.loads(candidate))
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(loaded, dict):
                        continue
                    raw_entry = cast(dict[str, object], loaded)
                    entry_input = raw_entry.get("input")
                    if isinstance(entry_input, str) and entry_input:
                        entries.append(entry_input)
            self._entries = entries
            self._loaded = True

    @property
    def size(self) -> int:
        """Return the number of stored prompt-history entries."""
        self._ensure_loaded()
        return len(self._entries)

    def append(self, text: str) -> None:
        """Append a new prompt to history, keeping it available in-memory immediately."""
        if not text:
            return
        self._ensure_loaded()
        entry: PromptHistoryEntry = {
            "input": text,
            "ts": datetime.now(UTC).isoformat(),
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            self._entries.append(text)
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    _ = handle.write(f"{line}\n")
            except OSError:
                return

    def entry(self, index: int) -> str:
        """Return a history entry by reverse index.

        ``index=1`` returns the most recent prompt, ``2`` the one before that, and so on.
        """
        if index <= 0:
            raise IndexError("History index must be positive.")
        self._ensure_loaded()
        try:
            return self._entries[-index]
        except IndexError as exc:
            raise IndexError(f"No history entry at index {index}") from exc

    def find_previous(self, query: str, *, before_index: int = 0) -> tuple[int, str] | None:
        """Find the previous history entry containing ``query``.

        Args:
            query: Substring to search for.
            before_index: Reverse index of the current history selection. Search starts
                strictly before that entry; ``0`` starts from the newest entry.
        """
        normalized_query = query.casefold().strip()
        if not normalized_query:
            return None
        self._ensure_loaded()
        start_index = max(before_index + 1, 1)
        for index in range(start_index, len(self._entries) + 1):
            entry = self._entries[-index]
            if normalized_query in entry.casefold():
                return index, entry
        return None

    def search(self, query: str) -> list[PromptHistoryMatch]:
        """Return recent prompt-history matches ordered by fuzzy picker relevance."""
        self._ensure_loaded()
        matches = [
            PromptHistoryMatch(index=index, text=text)
            for index, text in enumerate(reversed(self._entries), start=1)
        ]
        return filter_picker_items(
            matches,
            query,
            lambda match: (PickerSearchField(match.text, primary=True),),
        )
