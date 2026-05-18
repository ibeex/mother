"""Lightweight GitHub release update checks for Mother."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

from mother.config import CONFIG_DIR

GITHUB_LATEST_RELEASE_URL = "https://api.github.com/repos/ibeex/mother/releases/latest"
GITHUB_RELEASES_URL = "https://api.github.com/repos/ibeex/mother/releases?per_page=20"
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
        changelog = None
        if previous and _compare_versions(current, previous) > 0:
            changelog = _fetch_upgrade_changelog(previous, current)
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
    data = _get_json(GITHUB_LATEST_RELEASE_URL)
    if not isinstance(data, dict):
        raise ValueError("GitHub latest release response was not an object")
    return _release_from_json(cast(dict[str, object], data))


def _fetch_upgrade_changelog(previous: str, current: str) -> str | None:
    data = _get_json(GITHUB_RELEASES_URL)
    if not isinstance(data, list):
        return None
    sections: list[str] = []
    for raw_item in cast(list[object], data):
        if not isinstance(raw_item, dict):
            continue
        release = _release_from_json(cast(dict[str, object], raw_item))
        if (
            _compare_versions(release.version, previous) > 0
            and _compare_versions(release.version, current) <= 0
        ):
            body = release.body.strip() or "No release notes provided."
            sections.append(f"## {release.title}\n\n{body}")
    return "\n\n".join(sections) if sections else None


def _get_json(url: str) -> object:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "mother-update-check",
        },
    )
    with urllib.request.urlopen(request, timeout=5) as response:  # pyright: ignore[reportAny]
        payload = response.read()  # pyright: ignore[reportAny]
    text = cast(bytes, payload).decode("utf-8")
    return cast(object, json.loads(text))


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
