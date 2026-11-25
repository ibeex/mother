"""Tests for config loading and CLI override handling."""

from pathlib import Path

from mother import DEFAULT_MODEL, DEFAULT_SYSTEM, MotherConfig, load_config
from mother.config import (
    DEFAULT_DARK_THEME,
    DEFAULT_LIGHT_THEME,
    CouncilConfig,
    apply_cli_overrides,
)


def test_load_config_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config = load_config(config_file)
    assert config.model == DEFAULT_MODEL
    assert config.theme == "auto"
    assert config.theme_dark == DEFAULT_DARK_THEME
    assert config.theme_light == DEFAULT_LIGHT_THEME
    assert config.system_prompt == DEFAULT_SYSTEM
    assert config.reasoning_effort == "medium"
    assert config.openai_reasoning_summary == "auto"
    assert config.tools_enabled is False
    assert config.ca_bundle_path == ""
    assert config.submit_key == "enter"
    assert config.newline_key == "shift+enter"
    assert config.keybindings == {}
    assert config.council == CouncilConfig()
    assert config.config_path == str(config_file)
    assert config_file.exists()


def test_load_config_from_file(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text(
        "\n".join(
            [
                'model = "gpt-4o"',
                'theme = "textual-dark"',
                'theme_dark = "nord"',
                'theme_light = "textual-light"',
                'reasoning_effort = "high"',
                'openai_reasoning_summary = "detailed"',
                "tools_enabled = true",
                'ca_bundle_path = "/etc/ssl/certs/ib_cert.pem"',
                'submit_key = "Enter"',
                'newline_key = "Shift+Enter"',
                "",
                "[[models]]",
                'id = "openrouter"',
                'name = "openrouter/free"',
                'api_type = "openai-chat"',
                "response_model_name = true",
                "model_settings = { temperature = 0, top_p = 1 }",
                'reasoning_effort_map = { off = "", low = "low", medium = "high" }',
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
    assert config.theme_dark == "nord"
    assert config.theme_light == "textual-light"
    assert config.reasoning_effort == "high"
    assert config.openai_reasoning_summary == "detailed"
    assert config.tools_enabled is True
    assert config.ca_bundle_path == "/etc/ssl/certs/ib_cert.pem"
    assert config.submit_key == "enter"
    assert config.newline_key == "shift+enter"
    assert len(config.models) == 1
    assert config.models[0].response_model_name is True
    assert config.models[0].model_settings == {"temperature": 0, "top_p": 1}
    assert config.models[0].reasoning_effort_map == {"none": "", "low": "low", "medium": "high"}
    assert config.system_prompt == DEFAULT_SYSTEM
    assert config.council == CouncilConfig(
        members=("gpt-5", "g3", "opus"),
        judge="opus",
        max_context_turns=4,
        max_context_chars=6000,
    )
    assert config.config_path == str(config_file)


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


def test_load_config_rejects_invalid_reasoning_effort_map(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text(
        "\n".join(
            [
                "[[models]]",
                'id = "deepseek"',
                'name = "deepseek"',
                'api_type = "openai-chat"',
                'reasoning_effort_map = { turbo = "high" }',
                "",
            ]
        )
    )

    try:
        _ = load_config(config_file)
    except ValueError as exc:
        assert "reasoning_effort_map" in str(exc)
        assert "turbo" in str(exc)
    else:
        raise AssertionError("Expected invalid reasoning_effort_map key to raise ValueError")


def test_load_config_rejects_non_table_reasoning_effort_map(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text(
        "\n".join(
            [
                "[[models]]",
                'id = "deepseek"',
                'name = "deepseek"',
                'api_type = "openai-chat"',
                'reasoning_effort_map = "high"',
                "",
            ]
        )
    )

    try:
        _ = load_config(config_file)
    except ValueError as exc:
        assert "reasoning_effort_map" in str(exc)
    else:
        raise AssertionError("Expected non-table reasoning_effort_map to raise ValueError")


def test_apply_cli_overrides() -> None:
    base = MotherConfig(
        model="gpt-5",
        theme="catppuccin-mocha",
        system_prompt="Original.",
        reasoning_effort="high",
        openai_reasoning_summary="detailed",
        ca_bundle_path="/etc/ssl/certs/ib_cert.pem",
        keybindings={"toggle_auto_scroll": "ctrl+t"},
        council=CouncilConfig(members=("gpt-5",), judge="gpt-5"),
    )
    result = apply_cli_overrides(base, model="gpt-4o-mini", system=None)
    assert result.model == "gpt-4o-mini"
    assert result.theme == "catppuccin-mocha"
    assert result.system_prompt == "Original."
    assert result.reasoning_effort == "high"
    assert result.openai_reasoning_summary == "detailed"
    assert result.ca_bundle_path == "/etc/ssl/certs/ib_cert.pem"
    assert result.submit_key == "enter"
    assert result.newline_key == "shift+enter"
    assert result.keybindings == {"toggle_auto_scroll": "ctrl+t"}
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


def test_load_config_rejects_same_submit_and_newline_keys(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text('submit_key = "enter"\nnewline_key = "Enter"\n')

    try:
        _ = load_config(config_file)
    except ValueError as exc:
        assert "submit_key" in str(exc)
        assert "newline_key" in str(exc)
    else:
        raise AssertionError("Expected duplicate prompt keys to raise ValueError")


def test_load_config_keybinding_overrides(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text(
        "\n".join(
            [
                "[keybindings]",
                'submit = "Ctrl+Enter"',
                'newline = "Enter"',
                'toggle_auto_scroll = "Ctrl+T"',
                'markdown_cursor_down = "n,down"',
                "",
            ]
        )
    )

    config = load_config(config_file)

    assert config.submit_key == "ctrl+enter"
    assert config.newline_key == "enter"
    assert config.keybindings["toggle_auto_scroll"] == "ctrl+t"
    assert config.keybindings["markdown_cursor_down"] == "n,down"


def test_keybindings_take_precedence_over_legacy_prompt_keys(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text(
        "\n".join(
            [
                'submit_key = "ctrl+x"',
                'newline_key = "ctrl+y"',
                "[keybindings]",
                'submit = "ctrl+enter"',
                'newline = "enter"',
                "",
            ]
        )
    )

    config = load_config(config_file)

    assert config.submit_key == "ctrl+enter"
    assert config.newline_key == "enter"


def test_load_config_rejects_unknown_keybinding_id(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text('[keybindings]\nnot_a_binding = "ctrl+t"\n')

    try:
        _ = load_config(config_file)
    except ValueError as exc:
        assert "keybindings.not_a_binding" in str(exc)
        assert "toggle_auto_scroll" in str(exc)
    else:
        raise AssertionError("Expected unknown keybinding id to raise ValueError")


def test_load_config_rejects_same_keybinding_submit_and_newline(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text('[keybindings]\nsubmit = "enter"\nnewline = "Enter"\n')

    try:
        _ = load_config(config_file)
    except ValueError as exc:
        assert "keybindings.submit" in str(exc)
        assert "keybindings.newline" in str(exc)
    else:
        raise AssertionError("Expected duplicate prompt keybindings to raise ValueError")


def test_config_allowlist_from_toml(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text('allowlist = ["ls", "cat", "grep"]\n')
    config = load_config(config_file)
    assert config.allowlist == frozenset({"ls", "cat", "grep"})
