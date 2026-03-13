"""Tests for visible tool execution trace formatting."""

from mother.tool_trace import format_tool_arguments, format_tool_event, format_tool_output_preview


def test_format_tool_arguments_renders_shell_command_as_plain_text():
    text = format_tool_arguments({"command": "ls -la"})
    assert "Command:" in text
    assert "ls -la" in text


def test_format_tool_arguments_renders_extra_arguments_as_json():
    text = format_tool_arguments({"command": "ls -la", "timeout": 30.0})
    assert "Command:" in text
    assert '"timeout": 30.0' in text


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


def test_format_tool_output_preview_limits_to_five_lines():
    output = "\n".join(["1", "2", "3", "4", "5", "6", "7"])
    text = format_tool_output_preview(output)
    assert text == "1\n2\n3\n4\n5\n... (2 more lines)"


def test_format_tool_event_uses_truncated_preview_for_long_output():
    output = "\n".join(["a", "b", "c", "d", "e", "f"])
    text = format_tool_event("bash", {"command": "seq"}, status="finished", output=output)
    assert "a\nb\nc\nd\ne" in text
    assert "... (1 more lines)" in text
    assert "\nf\n" not in text
