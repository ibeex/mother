"""Lightweight GitHub release update checks for Mother."""

from __future__ import annotations

import html
import json
import os
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

from mother.config import CONFIG_DIR

GITHUB_LATEST_RELEASE_URL = "https://api.github.com/repos/ibeex/mother/releases/latest"
GITHUB_RELEASES_URL = "https://api.github.com/repos/ibeex/mother/releases?per_page=20"
GITHUB_RELEASES_ATOM_URL = "https://github.com/ibeex/mother/releases.atom"
STATE_FILE = CONFIG_DIR / "state.json"
PACKAGE_NAME = "mother"


@dataclass(frozen=True, slots=True)
class ReleaseInfo:
    version: str
    title: str
    url: str
    body: str


@dataclass(frozen=True, slots=True)
class UpdateCheckResult:
    current_version: str
    latest: ReleaseInfo | None
    upgrade_command: str
    changelog_markdown: str | None = None

    @property
    def update_available(self) -> bool:
        return (
            self.latest is not None
            and _compare_versions(self.latest.version, self.current_version) > 0
        )


def installed_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0.0.0"


def check_for_updates() -> UpdateCheckResult | None:
    """Check GitHub releases and update local seen-version state.

    Returns None on network/parse errors so startup never fails because of this feature.
    """
    if _env_flag("MOTHER_OFFLINE") or _env_flag("MOTHER_SKIP_VERSION_CHECK"):
        return None

    current = installed_version()
    try:
        latest = _fetch_latest_release()
        state = _read_state()
        previous = cast(str | None, state.get("last_seen_version"))
        notified = cast(str | None, state.get("latest_notified_version"))
        last_changelog = cast(str | None, state.get("last_changelog_version"))
        upgraded_from_seen = previous is not None and _compare_versions(current, previous) > 0
        upgraded_from_notified = notified is not None and _compare_versions(current, notified) >= 0
        changelog = None
        if last_changelog != current and (upgraded_from_seen or upgraded_from_notified):
            changelog = _fetch_upgrade_changelog(previous, current)
            if changelog:
                state["last_changelog_version"] = current
        if _compare_versions(latest.version, current) > 0:
            state["latest_notified_version"] = latest.version
        state["last_seen_version"] = current
        _write_state(state)
        return UpdateCheckResult(
            current_version=current,
            latest=latest,
            upgrade_command=_upgrade_command(),
            changelog_markdown=changelog,
        )
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError):
        return None


def _fetch_latest_release() -> ReleaseInfo:
    try:
        data = _get_json(GITHUB_LATEST_RELEASE_URL)
        if not isinstance(data, dict):
            raise ValueError("GitHub latest release response was not an object")
        return _release_from_json(cast(dict[str, object], data))
    except urllib.error.HTTPError as exc:
        releases = _fetch_atom_releases()
        if not releases:
            raise ValueError("GitHub releases Atom feed was empty") from exc
        return releases[0]


def _fetch_upgrade_changelog(previous: str | None, current: str) -> str | None:
    try:
        releases = _fetch_api_releases()
    except urllib.error.HTTPError:
        releases = _fetch_atom_releases()
    sections: list[str] = []
    for release in releases:
        after_previous = previous is None or _compare_versions(release.version, previous) > 0
        if after_previous and _compare_versions(release.version, current) <= 0:
            body = _format_release_body(release.body)
            sections.append(f"## {release.title}\n\n{body}")
    return "\n\n".join(sections) if sections else None


def _fetch_api_releases() -> list[ReleaseInfo]:
    data = _get_json(GITHUB_RELEASES_URL)
    if not isinstance(data, list):
        return []
    releases: list[ReleaseInfo] = []
    for raw_item in cast(list[object], data):
        if isinstance(raw_item, dict):
            releases.append(_release_from_json(cast(dict[str, object], raw_item)))
    return releases


def _fetch_atom_releases() -> list[ReleaseInfo]:
    root = ET.fromstring(_get_text(GITHUB_RELEASES_ATOM_URL))
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    releases: list[ReleaseInfo] = []
    for entry in root.findall("atom:entry", namespace):
        entry_id = entry.findtext("atom:id", default="", namespaces=namespace)
        raw_version = entry_id.rsplit("/", maxsplit=1)[-1]
        if not raw_version:
            continue
        title = entry.findtext("atom:title", default=raw_version, namespaces=namespace)
        content = entry.findtext("atom:content", default="", namespaces=namespace)
        link = entry.find("atom:link[@rel='alternate']", namespace)
        url = link.get("href", "") if link is not None else ""
        releases.append(
            ReleaseInfo(
                version=raw_version.removeprefix("v"),
                title=title,
                url=url,
                body=_html_to_text(content),
            )
        )
    return releases


def _get_json(url: str) -> object:
    return cast(object, json.loads(_get_text(url, accept="application/vnd.github+json")))


def _get_text(url: str, *, accept: str | None = None) -> str:
    headers = {"User-Agent": "mother-update-check"}
    if accept is not None:
        headers["Accept"] = accept
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:  # pyright: ignore[reportAny]
        payload = response.read()  # pyright: ignore[reportAny]
    return cast(bytes, payload).decode("utf-8")


def _release_from_json(data: dict[str, object]) -> ReleaseInfo:
    raw_tag = data.get("tag_name")
    if not isinstance(raw_tag, str) or not raw_tag:
        raise ValueError("GitHub release missing tag_name")
    name = data.get("name")
    url = data.get("html_url")
    body = data.get("body")
    release_version = raw_tag.removeprefix("v")
    return ReleaseInfo(
        version=release_version,
        title=name if isinstance(name, str) and name else raw_tag,
        url=url if isinstance(url, str) else "",
        body=body if isinstance(body, str) else "",
    )


def _format_release_body(value: str) -> str:
    """Normalize release notes so Markdown renders each note on its own line."""
    text = value.strip()
    if not text:
        return "No release notes provided."
    text = re.sub(r"\s+(Full Changelog:)", r"\n\n\1", text)
    text = re.sub(
        r"(?<!-)\s+((?:feat|fix|docs|style|refactor|perf|test|build|ci|chore)(?:\(.+?\))?:)",
        r"\n\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?<!-)\s+(breaking change\b)", r"\n\1", text, flags=re.IGNORECASE)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        if re.match(r"(?:(?:feat|fix|docs|style|refactor|perf|test|build|ci|chore)(?:\(.+\))?:|breaking change\b)", line, re.IGNORECASE):
            line = f"- {line}"
        lines.append(line)
    return "\n".join(lines).strip()


def _html_to_text(value: str) -> str:
    text = re.sub(r"</(?:p|div|li|h[1-6])>", "\n", value)
    text = re.sub(r"<li>", "- ", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    lines = [line.strip() for line in html.unescape(text).splitlines()]
    return _format_release_body("\n".join(line for line in lines if line))


def _read_state() -> dict[str, object]:
    if not STATE_FILE.exists():
        return {}
    with STATE_FILE.open("r", encoding="utf-8") as file:
        data = cast(object, json.load(file))
    return cast(dict[str, object], data) if isinstance(data, dict) else {}


def _write_state(state: dict[str, object]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ = STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _upgrade_command() -> str:
    executable = Path(sys.executable).as_posix().lower()
    prefix = Path(sys.prefix).as_posix().lower()
    if "pipx" in executable or "pipx" in prefix:
        return "pipx upgrade mother"
    return "uv tool upgrade mother"


def _compare_versions(left: str, right: str) -> int:
    left_parts = _version_parts(left)
    right_parts = _version_parts(right)
    max_len = max(len(left_parts), len(right_parts))
    left_parts.extend([0] * (max_len - len(left_parts)))
    right_parts.extend([0] * (max_len - len(right_parts)))
    return (left_parts > right_parts) - (left_parts < right_parts)


def _version_parts(value: str) -> list[int]:
    clean = value.strip().removeprefix("v")
    parts: list[int] = []
    for piece in clean.split("."):
        digits = ""
        for char in piece:
            if not char.isdigit():
                break
            digits += char
        parts.append(int(digits or "0"))
    return parts


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}
