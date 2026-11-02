"""Detect the user's light/dark preference and resolve the Textual theme."""

from __future__ import annotations

import math
import os
import select
import shutil
import subprocess
import sys
import time
from typing import Literal, cast

from textual.theme import BUILTIN_THEMES

from mother.config import AUTO_THEME, MotherConfig

Appearance = Literal["dark", "light"]

_OSC11_QUERY = "\x1b]11;?\x1b\\"
_OSC11_PREFIX = "\x1b]11;"
_OSC11_TIMEOUT_SECONDS = 0.15
_COMMAND_TIMEOUT_SECONDS = 1.0
_DARK_LUMINANCE_THRESHOLD = 0.5


def resolve_theme(config: MotherConfig) -> str:
    """Return the theme name to use for a configuration.

    An explicit theme always wins. ``"auto"`` resolves through the detected
    appearance and falls back to the configured dark theme when detection fails.
    """
    if config.theme != AUTO_THEME:
        return _validate_theme(config.theme)

    appearance = detect_appearance()
    theme = config.theme_light if appearance == "light" else config.theme_dark
    return _validate_theme(theme)


def detect_appearance() -> Appearance | None:
    """Detect whether the terminal or operating system prefers dark or light.

    The terminal's actual background is preferred over the OS setting because a
    light terminal can run inside a dark desktop, and vice versa.
    """
    return _detect_colorfgbg() or _detect_terminal_background() or _detect_os_appearance()


def _validate_theme(theme: str) -> str:
    if theme not in BUILTIN_THEMES:
        available = ", ".join(sorted(BUILTIN_THEMES))
        raise ValueError(f"Unknown theme {theme!r}. Available themes: {available}")
    return theme


def _detect_colorfgbg() -> Appearance | None:
    """Read the legacy ``COLORFGBG`` hint, e.g. ``15;0`` for light-on-dark."""
    value = os.environ.get("COLORFGBG", "").strip()
    if not value:
        return None

    background = value.rsplit(";", 1)[-1].strip()
    if not background.isdigit():
        return None

    background_index = int(background)
    if background_index < 0 or background_index > 15:
        return None

    return "dark" if background_index in {0, 1, 2, 3, 4, 5, 6, 8} else "light"


def _detect_terminal_background() -> Appearance | None:
    """Query the terminal background color with OSC 11, when possible."""
    if sys.platform == "win32":
        return None

    try:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return None
    except (AttributeError, ValueError):
        return None

    rgb = _query_osc11_rgb()
    if rgb is None:
        return None
    return "dark" if _is_dark_rgb(rgb) else "light"


def _query_osc11_rgb() -> tuple[int, int, int] | None:
    try:
        import termios
        import tty
    except ImportError:
        return None

    try:
        fd = sys.stdin.fileno()
        previous = termios.tcgetattr(fd)
    except (AttributeError, OSError, ValueError, termios.error):
        return None

    response = b""
    try:
        _ = tty.setcbreak(fd)
        _ = sys.stdout.write(_OSC11_QUERY)
        _ = sys.stdout.flush()

        deadline = time.monotonic() + _OSC11_TIMEOUT_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            try:
                ready, _, _ = select.select([fd], [], [], remaining)
            except OSError:
                break
            if not ready:
                break

            chunk = os.read(fd, 256)
            if not chunk:
                break
            response += chunk

            if _osc11_response_complete(response):
                break
    except (OSError, termios.error):
        return None
    finally:
        try:
            _ = termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        except termios.error:
            pass

    return _parse_osc11_response(response)


def _osc11_response_complete(data: bytes) -> bool:
    return _OSC11_PREFIX.encode() in data and (b"\x1b\\" in data or b"\x07" in data)


def _parse_osc11_response(data: bytes) -> tuple[int, int, int] | None:
    text = data.decode("ascii", errors="ignore")
    marker_index = text.rfind(_OSC11_PREFIX)
    if marker_index == -1:
        return None

    payload = text[marker_index + len(_OSC11_PREFIX) :]
    for terminator in ("\x1b", "\x07"):
        terminator_index = payload.find(terminator)
        if terminator_index != -1:
            payload = payload[:terminator_index]

    marker = "rgba:" if "rgba:" in payload else "rgb:"
    marker_index = payload.find(marker)
    if marker_index == -1:
        return None

    components = payload[marker_index + len(marker) :].split("/")
    if len(components) < 3:
        return None

    channels: list[int] = []
    for component in components[:3]:
        parsed = _parse_hex_component(component)
        if parsed is None:
            return None
        channels.append(parsed)

    return channels[0], channels[1], channels[2]


def _parse_hex_component(value: str) -> int | None:
    value = value.strip().lower()
    if not value or len(value) > 4:
        return None

    try:
        raw = int(value, 16)
    except ValueError:
        return None

    maximum = (1 << (4 * len(value))) - 1
    if maximum == 0:
        return None
    return round(raw / maximum * 255)


def _is_dark_rgb(rgb: tuple[int, int, int]) -> bool:
    red, green, blue = (_linearize(channel / 255) for channel in rgb)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return luminance < _DARK_LUMINANCE_THRESHOLD


def _linearize(channel: float) -> float:
    if channel <= 0.04045:
        return channel / 12.92
    return math.pow((channel + 0.055) / 1.055, 2.4)


def _detect_os_appearance() -> Appearance | None:
    if sys.platform == "darwin":
        return _detect_macos_appearance()
    if sys.platform == "win32":
        return _detect_windows_appearance()
    return _detect_linux_appearance()


def _detect_macos_appearance() -> Appearance | None:
    defaults = shutil.which("defaults")
    if defaults is None:
        return None

    try:
        completed = subprocess.run(
            [defaults, "read", "-g", "AppleInterfaceStyle"],
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode == 0:
        return "dark" if "dark" in completed.stdout.lower() else None

    # The key does not exist when the system is in light mode.
    if "does not exist" in completed.stderr.lower():
        return "light"
    return None


def _detect_windows_appearance() -> Appearance | None:
    try:
        import winreg
    except ImportError:
        return None

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        ) as key:
            value = cast(int, winreg.QueryValueEx(key, "AppsUseLightTheme")[0])
    except OSError:
        return None

    return "light" if value else "dark"


def _detect_linux_appearance() -> Appearance | None:
    color_scheme = _run_command(["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"])
    if color_scheme is not None:
        normalized = color_scheme.lower()
        if "dark" in normalized:
            return "dark"
        if "light" in normalized or "default" in normalized:
            return "light"

    gtk_theme = _run_command(["gsettings", "get", "org.gnome.desktop.interface", "gtk-theme"])
    if gtk_theme is not None:
        return "dark" if "dark" in gtk_theme.lower() else "light"

    for command in ("kreadconfig6", "kreadconfig5"):
        color_scheme = _run_command(
            [command, "--file", "kdeglobals", "--group", "General", "--key", "ColorScheme"]
        )
        if color_scheme is None:
            continue
        normalized = color_scheme.lower()
        if "dark" in normalized:
            return "dark"
        if "light" in normalized:
            return "light"

    gtk_theme_env = os.environ.get("GTK_THEME", "").lower()
    if "dark" in gtk_theme_env:
        return "dark"
    if "light" in gtk_theme_env:
        return "light"

    return None


def _run_command(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
