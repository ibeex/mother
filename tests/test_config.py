"""Tests for config loading and CLI override handling."""

from pathlib import Path

from mother import DEFAULT_MODEL, DEFAULT_SYSTEM, MotherConfig, load_config
from mother.config import apply_cli_overrides


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
    assert config_file.exists()


def test_load_config_from_file(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text(
        'model = "gpt-4o"\ntheme = "textual-dark"\nreasoning_effort = "high"\nopenai_reasoning_summary = "detailed"\ntools_enabled = true\nca_bundle_path = "/etc/ssl/certs/ib_cert.pem"\n'
    )
    config = load_config(config_file)
    assert config.model == "gpt-4o"
    assert config.theme == "textual-dark"
    assert config.reasoning_effort == "high"
    assert config.openai_reasoning_summary == "detailed"
    assert config.tools_enabled is True
    assert config.ca_bundle_path == "/etc/ssl/certs/ib_cert.pem"
    assert config.system_prompt == DEFAULT_SYSTEM


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
    )
    result = apply_cli_overrides(base, model="gpt-4o-mini", system=None)
    assert result.model == "gpt-4o-mini"
    assert result.theme == "catppuccin-mocha"
    assert result.system_prompt == "Original."
    assert result.reasoning_effort == "high"
    assert result.openai_reasoning_summary == "detailed"
    assert result.ca_bundle_path == "/etc/ssl/certs/ib_cert.pem"


def test_apply_cli_overrides_none() -> None:
    base = MotherConfig(
        model="gpt-5",
        theme="catppuccin-mocha",
        system_prompt="Original.",
        reasoning_effort="auto",
        openai_reasoning_summary="concise",
        ca_bundle_path="/etc/ssl/certs/ib_cert.pem",
    )
    result = apply_cli_overrides(base, model=None, system=None)
    assert result.model == "gpt-5"
    assert result.theme == "catppuccin-mocha"
    assert result.system_prompt == "Original."
    assert result.reasoning_effort == "auto"
    assert result.openai_reasoning_summary == "concise"
    assert result.ca_bundle_path == "/etc/ssl/certs/ib_cert.pem"


def test_config_allowlist_from_toml(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text('allowlist = ["ls", "cat", "grep"]\n')
    config = load_config(config_file)
    assert config.allowlist == frozenset({"ls", "cat", "grep"})
