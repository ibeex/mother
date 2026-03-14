"""Tests for runtime system prompt construction."""

from pathlib import Path

from mother.system_prompt import DEFAULT_BASE_SYSTEM, build_system_prompt


def test_build_system_prompt_chat_mode_includes_runtime_context():
    prompt = build_system_prompt(
        DEFAULT_BASE_SYSTEM,
        agent_mode=False,
        cwd=Path("/tmp/project"),
        current_date="2026-03-14",
        os_name="Linux 6.8",
        shell_name="zsh",
    )

    assert "You are Mother" in prompt
    assert "# Runtime Context" in prompt
    assert "- Current date: 2026-03-14" in prompt
    assert "- OS: Linux 6.8" in prompt
    assert "- Shell: zsh" in prompt
    assert "- Current working directory: /tmp/project" in prompt
    assert "- Mode: chat" in prompt
    assert "- Available tools:" not in prompt
    assert "- (none)" not in prompt
    assert "Do not work autonomously in a loop" not in prompt


def test_build_system_prompt_agent_mode_includes_tool_rules_and_tools():
    prompt = build_system_prompt(
        DEFAULT_BASE_SYSTEM,
        agent_mode=True,
        cwd=Path("/workspace"),
        tool_names=["bash", "web_search", "bash"],
        current_date="2026-03-14",
        os_name="macOS 15",
        shell_name="fish",
    )

    assert "In agent mode, you may use tools" in prompt
    assert "Do not work autonomously in a loop until the task is complete." in prompt
    assert "Use at most one tool call per turn" in prompt
    assert (
        "Ask before risky, destructive, privilege-requiring, or state-changing commands." in prompt
    )
    assert "- Mode: agent" in prompt
    assert "- bash: Execute shell commands on the local machine" in prompt
    assert "- web_search: Search the web for public information" in prompt
    assert prompt.count("- bash: Execute shell commands on the local machine") == 1


def test_build_system_prompt_preserves_custom_base_prompt():
    prompt = build_system_prompt(
        "Be terse.",
        agent_mode=False,
        cwd=Path("/tmp/project"),
        current_date="2026-03-14",
        os_name="Linux",
        shell_name="bash",
    )

    assert prompt.startswith("Be terse.")
    assert "- Mode: chat" in prompt
