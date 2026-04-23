"""Tests for visible tool execution trace formatting."""

from mother.tool_trace import format_tool_arguments, format_tool_event, format_tool_output


def test_format_tool_arguments_renders_shell_command_as_plain_text():
    text = format_tool_arguments({"command": "ls -la"})
    assert "Command:" in text
    assert "ls -la" in text


def test_format_tool_arguments_renders_extra_arguments_as_json():
    text = format_tool_arguments({"command": "ls -la", "timeout": 12.0})
    assert "Command:" in text
    assert '"timeout": 12.0' in text
    assert "Arguments:\n{" not in text
    assert not text.rstrip().endswith("}")


def test_format_tool_arguments_omits_empty_values():
    text = format_tool_arguments(
        {
            "url": "https://example.com/docs",
            "headers_json": "{}",
            "body": "",
            "extra": None,
        }
    )
    assert '"url": "https://example.com/docs"' in text
    assert '"headers_json"' not in text
    assert '"body"' not in text
    assert '"extra"' not in text


def test_format_tool_event_renders_started_status():
    text = format_tool_event("bash", {"command": "pwd"}, status="started")
    assert "Tool: bash" in text
    assert "Status: started" in text
    assert "pwd" in text


def test_format_tool_event_renders_finished_output():
    text = format_tool_event("bash", {"command": "pwd"}, status="finished", output="/tmp\n")
    assert "Status: finished" in text
    assert "Output:" in text
    assert "/tmp" in text


def test_format_tool_output_preserves_full_output():
    output = "\n".join(["1", "2", "3", "4", "5", "6", "7"])
    text = format_tool_output(output)
    assert text == output


def test_format_tool_event_preserves_full_output_for_long_output():
    output = "\n".join(["a", "b", "c", "d", "e", "f"])
    text = format_tool_event("bash", {"command": "seq"}, status="finished", output=output)
    assert "a\nb\nc\nd\ne\nf" in text
    assert "... (1 more lines)" not in text
