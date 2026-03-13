"""Tests for bash output capture, truncation, and sanitization."""

from mother.tools.bash_capture import (
    BashResult,
    OutputCapture,
    TruncationResult,
    format_truncation_notice,
    sanitize_output,
    truncate_tail,
)


def test_bash_result_defaults():
    r = BashResult()
    assert r.output == ""
    assert r.exit_code is None
    assert r.cancelled is False
    assert r.truncated is False
    assert r.full_output_path is None


def test_output_capture_small_output():
    cap = OutputCapture()
    cap.add_bytes(b"hello\nworld\n")
    trunc, path = cap.finalize()
    assert "hello" in trunc.content
    assert "world" in trunc.content
    assert trunc.truncated is False
    assert path is None


def test_output_capture_line_truncation():
    cap = OutputCapture()
    # Generate 2500 lines of small data
    lines = "\n".join(f"line {i}" for i in range(2500)) + "\n"
    cap.add_bytes(lines.encode("utf-8"))
    trunc, _ = cap.finalize()
    assert trunc.truncated is True
    # Should keep last 2000 lines
    assert "line 2499" in trunc.content
    assert "line 499" not in trunc.content


def test_output_capture_byte_truncation():
    cap = OutputCapture()
    # 60KB of data, single chunk
    data = ("x" * 60 + "\n") * 1000
    cap.add_bytes(data.encode("utf-8"))
    trunc, _ = cap.finalize()
    assert trunc.truncated is True
    assert (
        len(trunc.content.encode("utf-8")) <= 50 * 1024 + 100
    )  # allow small overage for line boundary


def test_output_capture_temp_file_created():
    cap = OutputCapture()
    # Send >50KB to trigger temp file
    chunk = b"a" * (51 * 1024)
    cap.add_bytes(chunk)
    _, path = cap.finalize()
    assert path is not None
    from pathlib import Path

    assert Path(path).exists()


def test_output_capture_rolling_buffer_evicts():
    cap = OutputCapture()
    # Send 150KB in chunks — rolling buffer should stay ~100KB
    chunk = b"b" * (10 * 1024)  # 10KB per chunk
    for _ in range(15):
        cap.add_bytes(chunk)
    # Buffer should be trimmed
    chunks_bytes: int = cap._chunks_bytes  # pyright: ignore[reportPrivateUsage]
    assert chunks_bytes <= OutputCapture.ROLLING_BYTES + 10 * 1024  # allow one chunk overshoot


def test_output_capture_single_long_line():
    cap = OutputCapture()
    # Single line > 50KB
    cap.add_bytes(b"z" * (60 * 1024))
    trunc, _ = cap.finalize()
    assert trunc.truncated is True
    assert len(trunc.content.encode("utf-8")) <= 50 * 1024


def test_sanitize_ansi_stripped():
    raw = b"\x1b[31mred text\x1b[0m"
    result = sanitize_output(raw)
    assert "\x1b" not in result
    assert "red text" in result


def test_sanitize_line_endings():
    raw = b"line1\r\nline2\rline3\n"
    result = sanitize_output(raw)
    assert "\r" not in result
    assert result.count("\n") == 3


def test_sanitize_utf8_boundaries():
    # Create text with multi-byte UTF-8 chars around the truncation boundary
    text = "hello " + "é" * 1000  # 'é' is 2 bytes in UTF-8
    raw = text.encode("utf-8")
    # Truncate at a byte boundary that might split é
    result = sanitize_output(raw[:7])  # 7 bytes, may split é
    # Should decode without error
    assert isinstance(result, str)


def test_truncation_notice_format():
    trunc = TruncationResult(
        content="...",
        truncated=True,
        original_lines=5000,
        kept_from=3001,
        kept_to=5000,
    )
    notice = format_truncation_notice(trunc, "/tmp/mother_bash_abc.txt")
    assert "3001" in notice
    assert "5000" in notice
    assert "/tmp/mother_bash_abc.txt" in notice


def test_truncate_tail_no_truncation():
    text = "line1\nline2\nline3\n"
    result = truncate_tail(text)
    assert result.truncated is False
    assert result.content == text


def test_truncate_tail_line_limit():
    lines = [f"line {i}\n" for i in range(3000)]
    text = "".join(lines)
    result = truncate_tail(text, max_lines=2000)
    assert result.truncated is True
    assert result.original_lines == 3000
    assert "line 2999" in result.content
    assert "line 999" not in result.content
