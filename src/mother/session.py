"""Session persistence and markdown export for Mother."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Literal, TypedDict, cast
from uuid import uuid4

SESSION_VERSION = 3
DEFAULT_SESSIONS_DIR = Path.home() / ".mother" / "sessions"
LAST_SESSION_FILE_NAME = "last"

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class SessionHeader(TypedDict, total=False):
    type: Literal["session"]
    version: int
    id: str
    created: str
    cwd: str
    model: str


class MessageEntry(TypedDict):
    type: Literal["message"]
    id: str
    ts: str
    role: Literal["user", "assistant"]
    content: str


class PromptEntry(TypedDict):
    type: Literal["prompt"]
    id: str
    ts: str
    user_text: str
    prompt_text: str
    system_prompt: str
    agent_mode: bool
    tool_names: list[str]
    attachment_paths: list[str]


class ToolCallEntry(TypedDict):
    type: Literal["tool_call"]
    id: str
    ts: str
    tool_name: str
    tool_call_id: str
    arguments: dict[str, JsonValue]


class ToolResultEntry(TypedDict):
    type: Literal["tool_result"]
    id: str
    ts: str
    tool_name: str
    tool_call_id: str
    arguments: dict[str, JsonValue]
    output: str
    is_error: bool


class EventEntry(TypedDict):
    type: Literal["event"]
    id: str
    ts: str
    name: str
    details: dict[str, JsonValue]


SessionEntry = MessageEntry | PromptEntry | ToolCallEntry | ToolResultEntry | EventEntry


def default_markdown_export_dir() -> Path:
    """Return the default directory for saved markdown transcripts."""
    debian_documents = Path.home() / "Debian" / "Documents"
    if debian_documents.exists():
        return debian_documents / "mother"
    return Path.home() / "Documents" / "mother"


def _normalize_json_value(value: object) -> JsonValue:
    """Convert Python values into a JSON-compatible representation."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        normalized: dict[str, JsonValue] = {}
        for key, item in cast(dict[object, object], value).items():
            normalized[str(key)] = _normalize_json_value(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in cast(list[object], value)]
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in cast(tuple[object, ...], value)]
    if isinstance(value, set):
        return [_normalize_json_value(item) for item in cast(set[object], value)]
    if isinstance(value, frozenset):
        return [_normalize_json_value(item) for item in value]
    return repr(value)


def _render_fenced_block(content: str, language: str = "text") -> str:
    """Render a fenced code block for markdown export."""
    trimmed = content.rstrip("\n")
    return f"```{language}\n{trimmed}\n```"


def _render_details(summary: str, body: str) -> list[str]:
    """Render a collapsible details block for markdown export."""
    return [
        "<details>",
        f"<summary>{summary}</summary>",
        "",
        body,
        "",
        "</details>",
    ]


@dataclass(frozen=True, slots=True)
class MarkdownFormatNotice:
    """Describe a non-fatal markdown formatting notice for the caller."""

    message: str
    severity: Literal["warning"] | None = None


def format_markdown_export(path: Path) -> MarkdownFormatNotice | None:
    """Format an exported markdown file with rumdl when uv is available."""
    if shutil.which("uv") is None:
        return MarkdownFormatNotice("Install uv to enable better markdown formatting on save.")

    try:
        _ = subprocess.run(
            ["uv", "run", "rumdl", "fmt", "--disable", "MD013", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr_value = cast(object, exc.stderr)
        stdout_value = cast(object, exc.stdout)
        stderr = stderr_value if isinstance(stderr_value, str) else ""
        stdout = stdout_value if isinstance(stdout_value, str) else ""
        detail = stderr.strip() or stdout.strip() or str(exc)
        return MarkdownFormatNotice(f"Markdown formatting failed: {detail}", severity="warning")

    return None


@dataclass(slots=True)
class SessionManager:
    """Create, append, export, and clean up append-only session files."""

    path: Path
    header: SessionHeader
    sessions_dir: Path = field(default=DEFAULT_SESSIONS_DIR)
    markdown_dir: Path = field(default_factory=default_markdown_export_dir)
    counter: int = 0
    _flushed: bool = False
    _lock: RLock = field(default_factory=RLock, repr=False)

    @property
    def last_file(self) -> Path:
        return self.sessions_dir / LAST_SESSION_FILE_NAME

    @classmethod
    def create(
        cls,
        *,
        sessions_dir: Path | None = None,
        markdown_dir: Path | None = None,
        cwd: Path | None = None,
        model_name: str | None = None,
    ) -> SessionManager:
        """Create a new session and delete the previous unsaved one, if present."""
        resolved_sessions_dir = (sessions_dir or DEFAULT_SESSIONS_DIR).expanduser()
        resolved_sessions_dir.mkdir(parents=True, exist_ok=True)
        cls._delete_last_if_unsaved(resolved_sessions_dir)

        created = datetime.now(UTC)
        session_id = uuid4().hex[:8]
        filename = f"{created.strftime('%Y-%m-%dT%H-%M-%S')}_{session_id}.jsonl"
        path = resolved_sessions_dir / filename

        header: SessionHeader = {
            "type": "session",
            "version": SESSION_VERSION,
            "id": session_id,
            "created": created.isoformat(),
            "cwd": str((cwd or Path.cwd()).resolve()),
        }
        if model_name:
            header["model"] = model_name

        manager = cls(
            path=path,
            header=header,
            sessions_dir=resolved_sessions_dir,
            markdown_dir=(markdown_dir or default_markdown_export_dir()).expanduser(),
        )
        _ = manager.last_file.write_text(f"{path}\n", encoding="utf-8")
        return manager

    @classmethod
    def save_last(
        cls,
        *,
        sessions_dir: Path | None = None,
        markdown_dir: Path | None = None,
    ) -> Path | None:
        """Save the last unsaved session to markdown, if one exists."""
        resolved_sessions_dir = (sessions_dir or DEFAULT_SESSIONS_DIR).expanduser()
        last_file = resolved_sessions_dir / LAST_SESSION_FILE_NAME
        if not last_file.exists():
            return None

        last_path = Path(last_file.read_text(encoding="utf-8").strip()).expanduser()
        if not last_path.exists():
            last_file.unlink(missing_ok=True)
            return None

        with last_path.open(encoding="utf-8") as handle:
            header = cast(SessionHeader, json.loads(handle.readline()))

        manager = cls(
            path=last_path,
            header=header,
            sessions_dir=resolved_sessions_dir,
            markdown_dir=(markdown_dir or default_markdown_export_dir()).expanduser(),
            _flushed=True,
        )
        output_path = manager.save_as_markdown()
        manager._clear_last_pointer()
        manager.path.unlink(missing_ok=True)
        manager._flushed = False
        manager.counter = 0
        return output_path

    def append(self, role: str, content: str) -> None:
        """Append a user or assistant message to the JSONL session."""
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unknown role: {role!r}")

        entry: MessageEntry = {
            "type": "message",
            "id": self._next_entry_id(),
            "ts": datetime.now(UTC).isoformat(),
            "role": cast(Literal["user", "assistant"], role),
            "content": content,
        }
        with self._lock:
            self._write(entry)

    def record_prompt(
        self,
        *,
        user_text: str,
        prompt_text: str,
        system_prompt: str,
        agent_mode: bool,
        tool_names: list[str],
        attachment_paths: list[str],
    ) -> None:
        """Record the exact prompt context sent to the LLM for a turn."""
        entry: PromptEntry = {
            "type": "prompt",
            "id": self._next_entry_id(),
            "ts": datetime.now(UTC).isoformat(),
            "user_text": user_text,
            "prompt_text": prompt_text,
            "system_prompt": system_prompt,
            "agent_mode": agent_mode,
            "tool_names": tool_names,
            "attachment_paths": attachment_paths,
        }
        with self._lock:
            self._write(entry)

    def record_tool_call(
        self,
        *,
        tool_name: str,
        tool_call_id: str | None,
        arguments: dict[str, object],
    ) -> None:
        """Record the start of a tool call."""
        entry: ToolCallEntry = {
            "type": "tool_call",
            "id": self._next_entry_id(),
            "ts": datetime.now(UTC).isoformat(),
            "tool_name": tool_name,
            "tool_call_id": tool_call_id or "",
            "arguments": cast(dict[str, JsonValue], _normalize_json_value(arguments)),
        }
        with self._lock:
            self._write(entry)

    def record_tool_result(
        self,
        *,
        tool_name: str,
        tool_call_id: str | None,
        arguments: dict[str, object],
        output: str,
        is_error: bool = False,
    ) -> None:
        """Record the output from a completed tool call."""
        entry: ToolResultEntry = {
            "type": "tool_result",
            "id": self._next_entry_id(),
            "ts": datetime.now(UTC).isoformat(),
            "tool_name": tool_name,
            "tool_call_id": tool_call_id or "",
            "arguments": cast(dict[str, JsonValue], _normalize_json_value(arguments)),
            "output": output,
            "is_error": is_error,
        }
        with self._lock:
            self._write(entry)

    def record_event(self, name: str, details: dict[str, object] | None = None) -> None:
        """Record a session lifecycle event such as model or mode changes."""
        entry: EventEntry = {
            "type": "event",
            "id": self._next_entry_id(),
            "ts": datetime.now(UTC).isoformat(),
            "name": name,
            "details": cast(dict[str, JsonValue], _normalize_json_value(details or {})),
        }
        with self._lock:
            self._write(entry)

    def save_as_markdown(self, output_dir: Path | None = None) -> Path:
        """Export the current session to markdown, overwriting the same file each time."""
        with self._lock:
            if not self.path.exists():
                raise RuntimeError("Nothing to save — no messages were sent yet.")

            resolved_output_dir = (output_dir or self.markdown_dir).expanduser()
            resolved_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = resolved_output_dir / self._markdown_filename()

            entries = self._load_entries()
            _ = output_path.write_text(self._render_markdown(entries), encoding="utf-8")
            return output_path

    def _next_entry_id(self) -> str:
        with self._lock:
            self.counter += 1
            return f"{self.counter:08d}"

    def _write(self, obj: SessionHeader | SessionEntry) -> None:
        if not self._flushed:
            with self.path.open("w", encoding="utf-8") as handle:
                _ = handle.write(json.dumps(self.header, ensure_ascii=False) + "\n")
            self._flushed = True

        with self.path.open("a", encoding="utf-8") as handle:
            _ = handle.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _load_entries(self) -> list[SessionEntry]:
        entries: list[SessionEntry] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                obj = cast(dict[str, object], json.loads(stripped))
                entry_type = obj.get("type")
                if entry_type == "message":
                    entries.append(cast(MessageEntry, cast(object, obj)))
                elif entry_type == "prompt":
                    entries.append(cast(PromptEntry, cast(object, obj)))
                elif entry_type == "tool_call":
                    entries.append(cast(ToolCallEntry, cast(object, obj)))
                elif entry_type == "tool_result":
                    entries.append(cast(ToolResultEntry, cast(object, obj)))
                elif entry_type == "event":
                    entries.append(cast(EventEntry, cast(object, obj)))
        return entries

    def _markdown_filename(self) -> str:
        created = str(self.header.get("created", ""))
        safe_created = created.replace(":", "-").replace(".", "-")
        return f"mother-{safe_created}.md"

    def _render_markdown(self, entries: list[SessionEntry]) -> str:
        created = str(self.header.get("created", ""))
        cwd = str(self.header.get("cwd", ""))
        model_name = self.header.get("model")

        lines = [
            "# Mother Session",
            "",
            f"**Date:** {created}  ",
            f"**Working directory:** `{cwd}`",
        ]
        if isinstance(model_name, str) and model_name:
            lines.append(f"**Model:** `{model_name}`")

        lines.extend(["", "## Session Summary", ""])
        lines.extend(f"- {item}" for item in self._build_summary(entries))

        first_prompt = self._first_prompt_entry(entries)
        if first_prompt is not None:
            lines.extend(
                [
                    "",
                    "## System Prompt",
                    "",
                    _render_fenced_block(first_prompt["system_prompt"]),
                ]
            )

        lines.extend(["", "---", ""])
        lines.extend(self._render_entries(entries))
        return "\n".join(lines).rstrip() + "\n"

    def _build_summary(self, entries: list[SessionEntry]) -> list[str]:
        user_messages = 0
        assistant_messages = 0
        prompt_count = 0
        tool_calls = 0
        tool_results = 0
        tools_seen: set[str] = set()
        models_seen: list[str] = []

        initial_model = self.header.get("model")
        if isinstance(initial_model, str) and initial_model:
            models_seen.append(initial_model)

        for entry in entries:
            if entry["type"] == "message":
                if entry["role"] == "user":
                    user_messages += 1
                else:
                    assistant_messages += 1
            elif entry["type"] == "prompt":
                prompt_count += 1
                tools_seen.update(entry["tool_names"])
            elif entry["type"] == "tool_call":
                tool_calls += 1
                tools_seen.add(entry["tool_name"])
            elif entry["type"] == "tool_result":
                tool_results += 1
                tools_seen.add(entry["tool_name"])
            elif entry["type"] == "event" and entry["name"] == "model_change":
                model_value = entry["details"].get("model")
                if isinstance(model_value, str) and model_value and model_value not in models_seen:
                    models_seen.append(model_value)

        summary = [
            f"{len(entries)} recorded entries",
            f"{user_messages} user messages",
            f"{assistant_messages} assistant messages",
            f"{prompt_count} prompt contexts",
            f"{tool_calls} tool calls",
            f"{tool_results} tool results",
        ]
        if tools_seen:
            summary.append(
                "Tools seen: " + ", ".join(f"`{tool_name}`" for tool_name in sorted(tools_seen))
            )
        if models_seen:
            summary.append(
                "Models seen: " + ", ".join(f"`{model_name}`" for model_name in models_seen)
            )
        return summary

    def _render_entries(self, entries: list[SessionEntry]) -> list[str]:
        lines: list[str] = []
        consumed_tool_results: set[int] = set()
        previous_system_prompt: str | None = None

        for index, entry in enumerate(entries):
            if entry["type"] == "tool_result" and index in consumed_tool_results:
                continue

            if entry["type"] == "tool_call":
                match_index = self._find_matching_tool_result_index(
                    entries,
                    start=index + 1,
                    tool_name=entry["tool_name"],
                    tool_call_id=entry["tool_call_id"],
                    consumed_tool_results=consumed_tool_results,
                )
                if match_index is not None:
                    consumed_tool_results.add(match_index)
                    matched_result = cast(ToolResultEntry, entries[match_index])
                    lines.extend(self._render_tool_call_entry(entry, matched_result))
                    continue

            if entry["type"] == "prompt":
                lines.extend(self._render_prompt_entry(entry, previous_system_prompt))
                previous_system_prompt = entry["system_prompt"]
                continue

            lines.extend(self._render_entry(entry))

        return lines

    def _find_matching_tool_result_index(
        self,
        entries: list[SessionEntry],
        *,
        start: int,
        tool_name: str,
        tool_call_id: str,
        consumed_tool_results: set[int],
    ) -> int | None:
        for index in range(start, len(entries)):
            entry = entries[index]
            if entry["type"] != "tool_result" or index in consumed_tool_results:
                continue
            if entry["tool_name"] != tool_name:
                continue
            if entry["tool_call_id"] != tool_call_id:
                continue
            return index
        return None

    def _first_prompt_entry(self, entries: list[SessionEntry]) -> PromptEntry | None:
        for entry in entries:
            if entry["type"] == "prompt":
                return entry
        return None

    def _render_entry(self, entry: SessionEntry) -> list[str]:
        if entry["type"] == "message":
            return self._render_message_entry(entry)
        if entry["type"] == "prompt":
            return self._render_prompt_entry(entry)
        if entry["type"] == "tool_call":
            return self._render_tool_call_entry(entry)
        if entry["type"] == "tool_result":
            return self._render_tool_result_entry(entry)
        return self._render_event_entry(entry)

    def _render_message_entry(self, entry: MessageEntry) -> list[str]:
        role = entry["role"].capitalize()
        return [
            f"## {role}",
            "",
            entry["content"],
            "",
            "---",
            "",
        ]

    def _render_prompt_entry(
        self,
        entry: PromptEntry,
        previous_system_prompt: str | None = None,
    ) -> list[str]:
        tools_available = (
            ", ".join(f"`{tool_name}`" for tool_name in entry["tool_names"])
            if entry["tool_names"]
            else "(none)"
        )
        attachment_paths = entry.get("attachment_paths", [])
        attachments = (
            ", ".join(f"`{path}`" for path in attachment_paths) if attachment_paths else "(none)"
        )
        lines = [
            "### Prompt Context",
            "",
            f"- Time: `{entry['ts']}`",
            f"- Mode: `{'agent' if entry['agent_mode'] else 'chat'}`",
            f"- Tools available: {tools_available}",
            f"- Attachments: {attachments}",
            "",
        ]
        if previous_system_prompt is not None and entry["system_prompt"] != previous_system_prompt:
            lines.extend(
                _render_details(
                    "System prompt updated",
                    _render_fenced_block(entry["system_prompt"]),
                )
            )
            lines.append("")
        if entry["prompt_text"] != entry["user_text"]:
            lines.extend(
                _render_details(
                    "Exact prompt sent to model",
                    _render_fenced_block(entry["prompt_text"]),
                )
            )
            lines.append("")
        lines.extend(["---", ""])
        return lines

    def _render_tool_call_entry(
        self,
        entry: ToolCallEntry,
        result: ToolResultEntry | None = None,
    ) -> list[str]:
        arguments_json = json.dumps(
            entry["arguments"], indent=2, sort_keys=True, ensure_ascii=False
        )
        lines = [
            f"### Tool Call · `{entry['tool_name']}`",
            "",
            f"- Time: `{entry['ts']}`",
        ]
        if entry["tool_call_id"]:
            lines.append(f"- Call ID: `{entry['tool_call_id']}`")
        if result is None:
            lines.append("- Status: `pending`")
            lines.extend(["", _render_fenced_block(arguments_json, "json"), "", "---", ""])
            return lines

        status = "error" if result["is_error"] else "ok"
        lines.append(f"- Status: `{status}`")
        lines.extend(["", "**Arguments**", "", _render_fenced_block(arguments_json, "json"), ""])
        lines.extend(_render_details("Tool output", _render_fenced_block(result["output"])))
        lines.extend(["", "---", ""])
        return lines

    def _render_tool_result_entry(self, entry: ToolResultEntry) -> list[str]:
        arguments_json = json.dumps(
            entry["arguments"], indent=2, sort_keys=True, ensure_ascii=False
        )
        status = "error" if entry["is_error"] else "ok"
        lines = [
            f"### Tool Result · `{entry['tool_name']}`",
            "",
            f"- Time: `{entry['ts']}`",
            f"- Status: `{status}`",
        ]
        if entry["tool_call_id"]:
            lines.append(f"- Call ID: `{entry['tool_call_id']}`")
        lines.extend(["", "**Arguments**", "", _render_fenced_block(arguments_json, "json"), ""])
        lines.extend(_render_details("Tool output", _render_fenced_block(entry["output"])))
        lines.extend(["", "---", ""])
        return lines

    def _render_event_entry(self, entry: EventEntry) -> list[str]:
        details_json = json.dumps(entry["details"], indent=2, sort_keys=True, ensure_ascii=False)
        return [
            f"### Event · `{entry['name']}`",
            "",
            f"- Time: `{entry['ts']}`",
            "",
            _render_fenced_block(details_json, "json"),
            "",
            "---",
            "",
        ]

    def _clear_last_pointer(self) -> None:
        if not self.last_file.exists():
            return
        last_path = self.last_file.read_text(encoding="utf-8").strip()
        if last_path == str(self.path):
            self.last_file.unlink(missing_ok=True)

    @classmethod
    def _delete_last_if_unsaved(cls, sessions_dir: Path) -> None:
        last_file = sessions_dir / LAST_SESSION_FILE_NAME
        if not last_file.exists():
            return

        last_path = Path(last_file.read_text(encoding="utf-8").strip()).expanduser()
        last_path.unlink(missing_ok=True)
        last_file.unlink(missing_ok=True)
