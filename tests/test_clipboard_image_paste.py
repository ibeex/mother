"""Tests for clipboard image paste support."""

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import PropertyMock, patch

from llm.models import Attachment
from PIL import Image

from mother import MotherApp
from mother import clipboard as clipboard_module
from mother.clipboard import MAX_IMAGE_DIMENSION, save_clipboard_image
from mother.widgets import PromptTextArea


def test_save_clipboard_image_writes_temp_image(tmp_path: Path) -> None:
    image = Image.new("RGBA", (2, 2), color=(255, 0, 0, 255))

    with patch("mother.clipboard.ImageGrab.grabclipboard", return_value=image):
        output_path = save_clipboard_image(tmp_path)

    assert output_path is not None
    assert output_path.exists()
    assert output_path.suffix == ".png"


def test_save_clipboard_image_skips_optimization_when_under_limits(tmp_path: Path) -> None:
    image = Image.new("RGBA", (32, 32), color=(255, 0, 0, 255))

    with (
        patch("mother.clipboard.ImageGrab.grabclipboard", return_value=image),
        patch("mother.clipboard._encoded_candidates", side_effect=AssertionError),
    ):
        output_path = save_clipboard_image(tmp_path)

    assert output_path is not None
    assert output_path.exists()


def test_save_clipboard_image_resizes_large_images(tmp_path: Path) -> None:
    image = Image.new("RGB", (4000, 1200), color="navy")

    with patch("mother.clipboard.ImageGrab.grabclipboard", return_value=image):
        output_path = save_clipboard_image(tmp_path)

    assert output_path is not None
    with Image.open(output_path) as saved_image:
        assert max(saved_image.size) == MAX_IMAGE_DIMENSION


def test_save_clipboard_image_corrects_exif_orientation(tmp_path: Path) -> None:
    image = Image.new("RGB", (40, 20), color="green")
    exif = Image.Exif()
    _ = exif.__setitem__(274, 6)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    _ = buffer.seek(0)

    with Image.open(buffer) as clipboard_image:
        with patch("mother.clipboard.ImageGrab.grabclipboard", return_value=clipboard_image):
            output_path = save_clipboard_image(tmp_path)

    assert output_path is not None
    with Image.open(output_path) as saved_image:
        assert saved_image.size == (20, 40)


def test_save_clipboard_image_respects_size_limit(tmp_path: Path) -> None:
    image = Image.effect_noise((1200, 1200), 100).convert("RGB")

    with (
        patch.object(clipboard_module, "MAX_IMAGE_BYTES", 50_000),
        patch("mother.clipboard.ImageGrab.grabclipboard", return_value=image),
    ):
        output_path = save_clipboard_image(tmp_path)

    assert output_path is not None
    assert output_path.stat().st_size <= 50_000


def test_prompt_text_area_action_paste_prefers_clipboard_image() -> None:
    text_area = PromptTextArea()
    fake_app = SimpleNamespace(
        clipboard="ignored text",
        capture_clipboard_image=lambda: "/tmp/pasted-image.png",
    )

    with patch.object(PromptTextArea, "app", new_callable=PropertyMock, return_value=fake_app):
        text_area.action_paste()

    assert text_area.text == "/tmp/pasted-image.png"


def test_prompt_text_area_action_paste_reads_system_text_clipboard() -> None:
    text_area = PromptTextArea()
    fake_app = SimpleNamespace(
        clipboard="ignored internal clipboard",
        capture_clipboard_image=lambda: None,
    )

    with (
        patch.object(PromptTextArea, "app", new_callable=PropertyMock, return_value=fake_app),
        patch("mother.widgets.read_clipboard_text", return_value="plain text"),
    ):
        text_area.action_paste()

    assert text_area.text == "plain text"


def test_prompt_text_area_action_paste_falls_back_to_internal_text_clipboard() -> None:
    text_area = PromptTextArea()
    fake_app = SimpleNamespace(
        clipboard="plain text",
        capture_clipboard_image=lambda: None,
    )

    with (
        patch.object(PromptTextArea, "app", new_callable=PropertyMock, return_value=fake_app),
        patch("mother.widgets.read_clipboard_text", return_value=None),
    ):
        text_area.action_paste()

    assert text_area.text == "plain text"


def test_capture_clipboard_image_registers_attachment(tmp_path: Path) -> None:
    app = MotherApp()
    image_path = tmp_path / "pasted.png"

    with (
        patch("mother.mother.save_clipboard_image", return_value=image_path),
        patch.object(app, "notify") as notify,
    ):
        result = app.capture_clipboard_image()

    assert result == str(image_path)
    assert app._pending_image_attachments[str(image_path)].path == str(image_path)  # pyright: ignore[reportPrivateUsage]
    notify.assert_called_once_with("Attached image: pasted.png", title="Clipboard")


def test_consume_attachments_for_text_only_returns_referenced_paths() -> None:
    app = MotherApp()
    first = Attachment(path="/tmp/first.png")
    second = Attachment(path="/tmp/second.png")
    app._pending_image_attachments = {  # pyright: ignore[reportPrivateUsage]
        cast(str, first.path): first,
        cast(str, second.path): second,
    }

    attachments = app._consume_attachments_for_text(  # pyright: ignore[reportPrivateUsage]
        "Please inspect /tmp/second.png"
    )

    assert attachments == [second]
    assert app._pending_image_attachments == {cast(str, first.path): first}  # pyright: ignore[reportPrivateUsage]
