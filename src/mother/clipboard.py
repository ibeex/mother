"""Clipboard helpers for prompt image attachments."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image as ImageModule
from PIL import ImageGrab, ImageOps
from PIL.Image import Image

MAX_IMAGE_DIMENSION = 2000
MAX_IMAGE_BYTES = 4_500_000
JPEG_QUALITIES: tuple[int, ...] = (85, 70, 55, 40)
DIMENSION_SCALES: tuple[float, ...] = (1.0, 0.75, 0.5, 0.35, 0.25)


class ClipboardImageError(RuntimeError):
    """Raised when Mother cannot read an image from the system clipboard."""


@dataclass(frozen=True)
class EncodedClipboardImage:
    """An encoded clipboard image candidate."""

    suffix: str
    format_name: str
    content: bytes

    @property
    def size(self) -> int:
        """Return the encoded image size in bytes."""
        return len(self.content)


def _normalize_clipboard_image(image: Image) -> Image:
    """Apply orientation fixes and detach the image from clipboard internals."""
    normalized = ImageOps.exif_transpose(image)
    _ = normalized.load()
    return normalized.copy()


def _resize_to_fit(image: Image, max_dimension: int) -> Image:
    """Resize an image so its longest edge is at most ``max_dimension``."""
    width, height = image.size
    longest_edge = max(width, height)
    if longest_edge <= max_dimension:
        return image

    scale = max_dimension / longest_edge
    size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(size, resample=ImageModule.Resampling.LANCZOS)  # pyright: ignore[reportUnknownMemberType]


def _scale_image(image: Image, scale: float) -> Image:
    """Return a resized copy for iterative size reduction."""
    if scale >= 1.0:
        return image
    width, height = image.size
    size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(size, resample=ImageModule.Resampling.LANCZOS)  # pyright: ignore[reportUnknownMemberType]


def _image_has_alpha(image: Image) -> bool:
    """Return whether the image carries transparency information."""
    if image.mode in {"RGBA", "LA"}:
        return True
    return image.mode == "P" and "transparency" in image.info


def _jpeg_source_image(image: Image) -> Image:
    """Convert an image into a JPEG-safe RGB image."""
    if _image_has_alpha(image):
        alpha_image = image.convert("RGBA")
        background = ImageModule.new("RGBA", alpha_image.size, (255, 255, 255, 255))
        composited = ImageModule.alpha_composite(background, alpha_image)
        return composited.convert("RGB")
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def _encode_png(image: Image) -> EncodedClipboardImage:
    """Encode an image as PNG."""
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return EncodedClipboardImage(suffix=".png", format_name="PNG", content=buffer.getvalue())


def _encode_jpeg(image: Image, quality: int) -> EncodedClipboardImage:
    """Encode an image as JPEG at a given quality."""
    buffer = BytesIO()
    _jpeg_source_image(image).save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
    )
    return EncodedClipboardImage(suffix=".jpg", format_name="JPEG", content=buffer.getvalue())


def _encoded_candidates(image: Image) -> list[EncodedClipboardImage]:
    """Return ordered encoded candidates for one image size."""
    png_candidate = _encode_png(image)
    jpeg_candidates = [_encode_jpeg(image, quality) for quality in JPEG_QUALITIES]
    primary = min((png_candidate, jpeg_candidates[0]), key=lambda candidate: candidate.size)
    return [primary, *jpeg_candidates[1:]]


def _optimize_image(image: Image) -> EncodedClipboardImage:
    """Optimize an image only when it exceeds size or dimension limits."""
    normalized_image = _normalize_clipboard_image(image)
    if max(normalized_image.size) <= MAX_IMAGE_DIMENSION:
        default_candidate = _encode_png(normalized_image)
        if default_candidate.size <= MAX_IMAGE_BYTES:
            return default_candidate

    base_image = _resize_to_fit(normalized_image, MAX_IMAGE_DIMENSION)
    smallest_candidate: EncodedClipboardImage | None = None

    for scale in DIMENSION_SCALES:
        scaled_image = _scale_image(base_image, scale)
        for candidate in _encoded_candidates(scaled_image):
            if smallest_candidate is None or candidate.size < smallest_candidate.size:
                smallest_candidate = candidate
            if candidate.size <= MAX_IMAGE_BYTES:
                return candidate

    details = ""
    if smallest_candidate is not None:
        details = f" Smallest result was {smallest_candidate.size / 1_000_000:.2f}MB."
    raise ClipboardImageError(
        f"Clipboard image is too large even after optimization. Limit is {MAX_IMAGE_BYTES / 1_000_000:.1f}MB.{details}"
    )


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

    optimized_image = _optimize_image(clipboard_data)
    path = target_dir / f"mother-clipboard-{uuid4().hex[:12]}{optimized_image.suffix}"
    _ = path.write_bytes(optimized_image.content)
    return path
