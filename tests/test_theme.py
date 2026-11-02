# pyright: reportPrivateUsage=false
"""Tests for system and terminal theme detection."""

import pytest

import mother.theme as theme_module
from mother.config import AUTO_THEME, DEFAULT_DARK_THEME, DEFAULT_LIGHT_THEME, MotherConfig
from mother.theme import (
    _detect_colorfgbg,
    _is_dark_rgb,
    _parse_hex_component,
    _parse_osc11_response,
    resolve_theme,
)


def test_resolve_theme_uses_explicit_theme(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected_detection() -> theme_module.Appearance:
        raise AssertionError("detection should not run for an explicit theme")

    monkeypatch.setattr(theme_module, "detect_appearance", _unexpected_detection)

    config = MotherConfig(theme="textual-light")
    assert resolve_theme(config) == "textual-light"


def test_resolve_theme_auto_dark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(theme_module, "detect_appearance", lambda: "dark")

    config = MotherConfig(theme=AUTO_THEME)
    assert resolve_theme(config) == DEFAULT_DARK_THEME


def test_resolve_theme_auto_light(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(theme_module, "detect_appearance", lambda: "light")

    config = MotherConfig(theme=AUTO_THEME)
    assert resolve_theme(config) == DEFAULT_LIGHT_THEME


def test_resolve_theme_auto_uses_custom_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(theme_module, "detect_appearance", lambda: "light")

    config = MotherConfig(theme=AUTO_THEME, theme_dark="nord", theme_light="textual-light")
    assert resolve_theme(config) == "textual-light"


def test_resolve_theme_auto_falls_back_to_dark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(theme_module, "detect_appearance", lambda: None)

    config = MotherConfig(theme=AUTO_THEME)
    assert resolve_theme(config) == DEFAULT_DARK_THEME


def test_resolve_theme_rejects_unknown_theme() -> None:
    config = MotherConfig(theme="not-a-theme")

    try:
        _ = resolve_theme(config)
    except ValueError as exc:
        assert "not-a-theme" in str(exc)
        assert "catppuccin-mocha" in str(exc)
    else:
        raise AssertionError("Expected an unknown theme to raise ValueError")


def test_detect_colorfgbg_dark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLORFGBG", "15;0")
    assert _detect_colorfgbg() == "dark"


def test_detect_colorfgbg_light(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLORFGBG", "0;15")
    assert _detect_colorfgbg() == "light"


def test_detect_colorfgbg_ignores_missing_or_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLORFGBG", raising=False)
    assert _detect_colorfgbg() is None

    monkeypatch.setenv("COLORFGBG", "default")
    assert _detect_colorfgbg() is None


def test_parse_osc11_short_components() -> None:
    response = b"\x1b]11;rgb:1e/1e/1e\x1b\\"
    assert _parse_osc11_response(response) == (30, 30, 30)


def test_parse_osc11_long_components() -> None:
    response = b"\x1b]11;rgb:ffff/ffff/ffff\x07"
    assert _parse_osc11_response(response) == (255, 255, 255)


def test_parse_osc11_rgba_components() -> None:
    response = b"\x1b]11;rgba:0000/0000/0000/ffff\x1b\\"
    assert _parse_osc11_response(response) == (0, 0, 0)


def test_parse_osc11_ignores_unrelated_input() -> None:
    assert _parse_osc11_response(b"\x1b[?1;2c") is None
    assert _parse_osc11_response(b"\x1b]11;unknown\x1b\\") is None


def test_parse_hex_component_scales_by_width() -> None:
    assert _parse_hex_component("ff") == 255
    assert _parse_hex_component("ffff") == 255
    assert _parse_hex_component("0") == 0
    assert _parse_hex_component("zz") is None
    assert _parse_hex_component("") is None


def test_is_dark_rgb() -> None:
    assert _is_dark_rgb((0, 0, 0)) is True
    assert _is_dark_rgb((24, 24, 37)) is True
    assert _is_dark_rgb((255, 255, 255)) is False
    assert _is_dark_rgb((239, 241, 245)) is False


def test_detect_appearance_prefers_cheap_colorfgbg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLORFGBG", "15;0")
    monkeypatch.setattr(theme_module, "_detect_colorfgbg", _detect_colorfgbg)
    monkeypatch.setattr(theme_module, "_detect_terminal_background", lambda: "light")
    monkeypatch.setattr(theme_module, "_detect_os_appearance", lambda: "light")

    assert theme_module.detect_appearance() == "dark"


def test_detect_appearance_falls_through_to_os(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLORFGBG", raising=False)
    monkeypatch.setattr(theme_module, "_detect_terminal_background", lambda: None)
    monkeypatch.setattr(theme_module, "_detect_os_appearance", lambda: "light")

    assert theme_module.detect_appearance() == "light"
