"""Tests for config loading and CLI override handling."""

from pathlib import Path

from mother import DEFAULT_MODEL, DEFAULT_SYSTEM, MotherConfig, load_config
from mother.config import CouncilConfig, apply_cli_overrides


def test_load_config_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config = load_config(config_file)
    assert config.model == DEFAULT_MODEL
    assert config.theme == "catppuccin-mocha"
    assert config.system_prompt == DEFAULT_SYSTEM
    assert config.reasoning_effort == "medium"
    assert config.openai_reasoning_summary == "auto"
    assert config.tools_enabled is False
    assert config.ca_bundle_path == ""
    assert config.council == CouncilConfig()
    assert config_file.exists()


def test_load_config_from_file(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text(
        "\n".join(
            [
                'model = "gpt-4o"',
                'theme = "textual-dark"',
                'reasoning_effort = "high"',
                'openai_reasoning_summary = "detailed"',
                "tools_enabled = true",
                'ca_bundle_path = "/etc/ssl/certs/ib_cert.pem"',
                "",
                "[council]",
                'members = ["gpt-5", "g3", "opus"]',
                'judge = "opus"',
                "max_context_turns = 4",
                "max_context_chars = 6000",
                "",
            ]
        )
    )
    config = load_config(config_file)
    assert config.model == "gpt-4o"
    assert config.theme == "textual-dark"
    assert config.reasoning_effort == "high"
    assert config.openai_reasoning_summary == "detailed"
    assert config.tools_enabled is True
    assert config.ca_bundle_path == "/etc/ssl/certs/ib_cert.pem"
    assert config.system_prompt == DEFAULT_SYSTEM
    assert config.council == CouncilConfig(
        members=("gpt-5", "g3", "opus"),
        judge="opus",
        max_context_turns=4,
        max_context_chars=6000,
    )


def test_load_config_rejects_invalid_reasoning_effort(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text('reasoning_effort = "turbo"\n')

    try:
        _ = load_config(config_file)
    except ValueError as exc:
        assert "reasoning_effort" in str(exc)
    else:
        raise AssertionError("Expected invalid reasoning_effort to raise ValueError")


def test_load_config_rejects_invalid_openai_reasoning_summary(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text('openai_reasoning_summary = "verbose"\n')

    try:
        _ = load_config(config_file)
    except ValueError as exc:
        assert "openai_reasoning_summary" in str(exc)
    else:
        raise AssertionError("Expected invalid openai_reasoning_summary to raise ValueError")


def test_apply_cli_overrides() -> None:
    base = MotherConfig(
        model="gpt-5",
        theme="catppuccin-mocha",
        system_prompt="Original.",
        reasoning_effort="high",
        openai_reasoning_summary="detailed",
        ca_bundle_path="/etc/ssl/certs/ib_cert.pem",
        council=CouncilConfig(members=("gpt-5",), judge="gpt-5"),
    )
    result = apply_cli_overrides(base, model="gpt-4o-mini", system=None)
    assert result.model == "gpt-4o-mini"
    assert result.theme == "catppuccin-mocha"
    assert result.system_prompt == "Original."
    assert result.reasoning_effort == "high"
    assert result.openai_reasoning_summary == "detailed"
    assert result.ca_bundle_path == "/etc/ssl/certs/ib_cert.pem"
    assert result.council == CouncilConfig(members=("gpt-5",), judge="gpt-5")


def test_apply_cli_overrides_none() -> None:
    base = MotherConfig(
        model="gpt-5",
        theme="catppuccin-mocha",
        system_prompt="Original.",
        reasoning_effort="auto",
        openai_reasoning_summary="concise",
        ca_bundle_path="/etc/ssl/certs/ib_cert.pem",
        council=CouncilConfig(members=("opus",), judge="opus"),
    )
    result = apply_cli_overrides(base, model=None, system=None)
    assert result.model == "gpt-5"
    assert result.theme == "catppuccin-mocha"
    assert result.system_prompt == "Original."
    assert result.reasoning_effort == "auto"
    assert result.openai_reasoning_summary == "concise"
    assert result.ca_bundle_path == "/etc/ssl/certs/ib_cert.pem"
    assert result.council == CouncilConfig(members=("opus",), judge="opus")


def test_config_allowlist_from_toml(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text('allowlist = ["ls", "cat", "grep"]\n')
    config = load_config(config_file)
    assert config.allowlist == frozenset({"ls", "cat", "grep"})
