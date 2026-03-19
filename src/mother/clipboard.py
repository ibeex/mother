"""Clipboard helpers for prompt image attachments."""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

from PIL import ImageGrab
from PIL.Image import Image


class ClipboardImageError(RuntimeError):
    """Raised when Mother cannot read an image from the system clipboard."""


_IMAGE_FORMATS: dict[str, tuple[str, str]] = {
    "BMP": (".bmp", "BMP"),
    "GIF": (".gif", "GIF"),
    "JPEG": (".jpg", "JPEG"),
    "JPG": (".jpg", "JPEG"),
    "PNG": (".png", "PNG"),
    "WEBP": (".webp", "WEBP"),
}


def _target_image_format(image: Image) -> tuple[str, str]:
    """Return the preferred file suffix and Pillow format for a clipboard image."""
    format_name = (image.format or "PNG").upper()
    return _IMAGE_FORMATS.get(format_name, (".png", "PNG"))


def _prepare_image_for_save(image: Image, format_name: str) -> Image:
    """Return an image object that can be written using the target format."""
    if format_name != "JPEG" or image.mode in {"L", "RGB"}:
        return image
    return image.convert("RGB")


def save_clipboard_image(temp_dir: Path | None = None) -> Path | None:
    """Save the current clipboard image to a temp file and return its path."""
    try:
        clipboard_data = ImageGrab.grabclipboard()
    except NotImplementedError as exc:
        raise ClipboardImageError(
            "Clipboard image paste is not supported on this platform."
        ) from exc
    except OSError as exc:
        raise ClipboardImageError(f"Failed to read clipboard image: {exc}") from exc
    except Exception as exc:
        raise ClipboardImageError(f"Unexpected clipboard image error: {exc}") from exc

    if not isinstance(clipboard_data, Image):
        return None

    target_dir = (temp_dir or Path(tempfile.gettempdir()) / "mother").expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    suffix, format_name = _target_image_format(clipboard_data)
    image_to_save = _prepare_image_for_save(clipboard_data, format_name)
    path = target_dir / f"mother-clipboard-{uuid4().hex[:12]}{suffix}"
    image_to_save.save(path, format=format_name)
    return path
