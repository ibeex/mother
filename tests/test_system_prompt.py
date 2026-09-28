"""Tests for runtime system prompt construction.

These tests focus on the two things that actually matter and stay stable as the
prose is reworded:

* the builder wires the right mode section and runtime context together, and
* the mode guidance still expresses its core behavioural contract.

Intent checks use tolerant regular expressions instead of exact sentences so a
wording tweak does not require editing the test, but a missing rule still fails.
"""

from __future__ import annotations

import re
from pathlib import Path

from mother.system_prompt import DEFAULT_BASE_SYSTEM, build_system_prompt


def _assert_matches(prompt: str, pattern: str) -> None:
    if re.search(pattern, prompt, flags=re.IGNORECASE) is None:
        raise AssertionError(f"prompt did not match /{pattern}/i:\n{prompt}")


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

    # Agent and research guidance must not leak into plain chat mode.
    assert "In agent mode" not in prompt
    assert "In deep research mode" not in prompt


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
    assert "- Mode: agent" in prompt
    assert "- Available tools:" in prompt
    assert "- bash: Execute shell commands on the local machine" in prompt
    assert "- web_search: Search the web for public information" in prompt
    assert prompt.count("- bash: Execute shell commands on the local machine") == 1
    assert "In deep research mode" not in prompt


def test_agent_guidance_keeps_single_tool_step_and_safety_contract():
    prompt = build_system_prompt(DEFAULT_BASE_SYSTEM, agent_mode=True)

    _assert_matches(prompt, r"one tool call per (?:turn|user message)")
    _assert_matches(prompt, r"do not work autonomously")
    _assert_matches(prompt, r"ask before .*risky")
    _assert_matches(prompt, r"wait for the user")


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

    assert "- Mode: deep research" in prompt
    assert "- web_search: Search the web for public information" in prompt
    assert "- web_fetch: Fetch web pages or HTTP endpoints" in prompt

    _assert_matches(prompt, r"concise research plan")
    _assert_matches(prompt, r"confirm or adjust the plan")
    _assert_matches(prompt, r"execute the research autonomously")
    _assert_matches(prompt, r"use only web_search and web_fetch")


def test_build_system_prompt_tool_list_dedupes_and_falls_back_to_name():
    prompt = build_system_prompt(
        DEFAULT_BASE_SYSTEM,
        agent_mode=True,
        tool_names=["custom_tool", "custom_tool"],
    )

    assert "- custom_tool: custom_tool" in prompt
    assert prompt.count("- custom_tool: custom_tool") == 1

    prompt_without_tools = build_system_prompt(DEFAULT_BASE_SYSTEM, agent_mode=True)
    assert "- (none)" in prompt_without_tools


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
