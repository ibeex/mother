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

    assert "In agent mode" in prompt
    assert "one tool call per turn" in prompt
    assert "Ask before risky" in prompt
    assert "- Mode: agent" in prompt
    assert "- bash: Execute shell commands on the local machine" in prompt
    assert "- web_search: Search the web for public information" in prompt
    assert prompt.count("- bash: Execute shell commands on the local machine") == 1


def test_build_system_prompt_deep_research_mode_includes_research_rules_and_tools():
    prompt = build_system_prompt(
        DEFAULT_BASE_SYSTEM,
        mode="deep_research",
        cwd=Path("/research"),
        tool_names=["web_search", "web_fetch"],
        current_date="2026-03-14",
        os_name="Linux 6.8",
        shell_name="zsh",
    )

    assert "In deep research mode" in prompt
    assert "concise research plan" in prompt
    assert "confirm or adjust the plan" in prompt
    assert "execute the research autonomously" in prompt
    assert "use only web_search and web_fetch" in prompt
    assert "- Mode: deep research" in prompt
    assert "- web_search: Search the web for public information" in prompt
    assert "- web_fetch: Fetch web pages or HTTP endpoints" in prompt
    assert "- bash: Execute shell commands on the local machine" not in prompt


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
