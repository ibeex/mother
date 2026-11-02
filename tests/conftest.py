"""Shared pytest fixtures for the Mother test suite."""

from pathlib import Path

import pytest

import mother.history as history_module
import mother.theme as theme_module


@pytest.fixture(autouse=True)
def isolate_theme_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep theme auto-detection deterministic in tests."""

    def _detect_dark() -> theme_module.Appearance:
        return "dark"

    monkeypatch.setattr(theme_module, "_detect_colorfgbg", lambda: None)
    monkeypatch.setattr(theme_module, "_detect_terminal_background", lambda: None)
    monkeypatch.setattr(theme_module, "_detect_os_appearance", _detect_dark)


@pytest.fixture(autouse=True)
def isolate_default_prompt_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent tests from reading or writing the developer's real prompt history."""
    history_path = tmp_path / ".mother" / "prompt_history.jsonl"
    monkeypatch.setattr(history_module, "DEFAULT_PROMPT_HISTORY_FILE", history_path)
