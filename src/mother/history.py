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


@dataclass(frozen=True, slots=True)
class _PromptHistoryRecord:
    """Internal persisted prompt-history record with stable timestamp data."""

    text: str
    ts: str


def default_prompt_history_path() -> Path:
    """Return the default JSONL path used for prompt history."""
    return DEFAULT_PROMPT_HISTORY_FILE


@dataclass(slots=True)
class PromptHistory:
    """Persistent prompt history with simple reverse search support."""

    path: Path = field(default_factory=default_prompt_history_path)
    _records: list[_PromptHistoryRecord] = field(default_factory=list, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)
            records: list[_PromptHistoryRecord] = []
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
                    entry_ts = raw_entry.get("ts")
                    if isinstance(entry_input, str) and entry_input:
                        ts = entry_ts if isinstance(entry_ts, str) and entry_ts else ""
                        records.append(_PromptHistoryRecord(text=entry_input, ts=ts))
            self._records = records
            self._loaded = True

    @property
    def size(self) -> int:
        """Return the number of stored prompt-history entries."""
        self._ensure_loaded()
        return len(self._records)

    def _write_records_locked(self) -> None:
        """Rewrite the history file from the current in-memory records."""
        serialized = "".join(
            f"{json.dumps(PromptHistoryEntry(input=record.text, ts=record.ts), ensure_ascii=False)}\n"
            for record in self._records
        )
        temporary_path = self.path.with_name(f"{self.path.name}.tmp")
        try:
            temporary_path.write_text(serialized, encoding="utf-8")
            temporary_path.replace(self.path)
        except OSError:
            if temporary_path.exists():
                temporary_path.unlink(missing_ok=True)
            raise

    def append(self, text: str) -> None:
        """Append a new prompt or move an exact duplicate to the newest position."""
        if not text:
            return
        self._ensure_loaded()
        entry = _PromptHistoryRecord(text=text, ts=datetime.now(UTC).isoformat())
        with self._lock:
            existing_index = next(
                (index for index, record in enumerate(self._records) if record.text == text),
                None,
            )
            if existing_index is None:
                self._records.append(entry)
                try:
                    line = json.dumps(
                        PromptHistoryEntry(input=entry.text, ts=entry.ts),
                        ensure_ascii=False,
                    )
                    with self.path.open("a", encoding="utf-8") as handle:
                        _ = handle.write(f"{line}\n")
                except OSError:
                    return
                return

            self._records = [record for record in self._records if record.text != text]
            self._records.append(entry)
            try:
                self._write_records_locked()
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
            return self._records[-index].text
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
        for index in range(start_index, len(self._records) + 1):
            entry = self._records[-index].text
            if normalized_query in entry.casefold():
                return index, entry
        return None

    def search(self, query: str) -> list[PromptHistoryMatch]:
        """Return recent prompt-history matches ordered by fuzzy picker relevance.

        Exact duplicate prompt texts are collapsed so the newest copy wins.
        """
        self._ensure_loaded()
        seen: set[str] = set()
        matches: list[PromptHistoryMatch] = []
        for index, record in enumerate(reversed(self._records), start=1):
            if record.text in seen:
                continue
            seen.add(record.text)
            matches.append(PromptHistoryMatch(index=index, text=record.text))
        return filter_picker_items(
            matches,
            query,
            lambda match: (PickerSearchField(match.text, primary=True),),
        )
