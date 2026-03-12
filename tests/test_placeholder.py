"""Tests for the mother package."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from mother import DEFAULT_MODEL, DEFAULT_SYSTEM, MotherApp, MotherConfig, cli, load_config
from mother.config import apply_cli_overrides


def test_default_model_constant():
    assert DEFAULT_MODEL == "gpt-5"


def test_default_system_constant():
    assert "Mother" in DEFAULT_SYSTEM
    assert len(DEFAULT_SYSTEM) > 0


def test_mother_app_defaults():
    app = MotherApp()
    assert app.config.model == DEFAULT_MODEL
    assert app.config.system_prompt == DEFAULT_SYSTEM


def test_mother_app_custom_args():
    app = MotherApp(model_name="gpt-4o-mini", system="Custom system prompt.")
    assert app.config.model == "gpt-4o-mini"
    assert app.config.system_prompt == "Custom system prompt."


def test_mother_app_config_kwarg():
    config = MotherConfig(model="gpt-4o", system_prompt="Be brief.")
    app = MotherApp(config=config)
    assert app.config.model == "gpt-4o"
    assert app.config.system_prompt == "Be brief."


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--system" in result.output


def test_cli_default_wiring():
    runner = CliRunner()
    with (
        patch("mother.mother.load_config") as mock_load,
        patch("mother.mother.MotherApp") as mock_app_cls,
    ):
        mock_load.return_value = MotherConfig()
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        runner.invoke(cli, [])
        mock_app_cls.assert_called_once()
        mock_app.run.assert_called_once()


def test_cli_custom_model():
    runner = CliRunner()
    with (
        patch("mother.mother.load_config") as mock_load,
        patch("mother.mother.MotherApp") as mock_app_cls,
    ):
        mock_load.return_value = MotherConfig()
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        runner.invoke(cli, ["-m", "gpt-4o-mini"])
        call_kwargs = mock_app_cls.call_args
        passed_config: MotherConfig = call_kwargs.kwargs["config"]
        assert passed_config.model == "gpt-4o-mini"


def test_cli_custom_system():
    runner = CliRunner()
    with (
        patch("mother.mother.load_config") as mock_load,
        patch("mother.mother.MotherApp") as mock_app_cls,
    ):
        mock_load.return_value = MotherConfig()
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        runner.invoke(cli, ["-s", "Be a pirate."])
        call_kwargs = mock_app_cls.call_args
        passed_config: MotherConfig = call_kwargs.kwargs["config"]
        assert passed_config.system_prompt == "Be a pirate."


# Config tests


def test_load_config_defaults(tmp_path: pytest.FixtureRequest):
    config_file = tmp_path / "config.toml"  # type: ignore[operator]
    config = load_config(config_file)  # type: ignore[arg-type]
    assert config.model == DEFAULT_MODEL
    assert config.system_prompt == DEFAULT_SYSTEM
    assert config.tools_enabled is False
    assert config_file.exists()  # type: ignore[union-attr]


def test_load_config_from_file(tmp_path: pytest.FixtureRequest):
    config_file = tmp_path / "config.toml"  # type: ignore[operator]
    config_file.write_text('model = "gpt-4o"\ntools_enabled = true\n')  # type: ignore[union-attr]
    config = load_config(config_file)  # type: ignore[arg-type]
    assert config.model == "gpt-4o"
    assert config.tools_enabled is True
    assert config.system_prompt == DEFAULT_SYSTEM


def test_apply_cli_overrides():
    base = MotherConfig(model="gpt-5", system_prompt="Original.")
    result = apply_cli_overrides(base, model="gpt-4o-mini", system=None)
    assert result.model == "gpt-4o-mini"
    assert result.system_prompt == "Original."


def test_apply_cli_overrides_none():
    base = MotherConfig(model="gpt-5", system_prompt="Original.")
    result = apply_cli_overrides(base, model=None, system=None)
    assert result.model == "gpt-5"
    assert result.system_prompt == "Original."


# Tools tests


def test_tool_registry_empty():
    from mother.tools import get_default_tools

    registry = get_default_tools()
    assert registry.is_empty()
    assert registry.tools() == []


def test_tool_registry_register():
    from mother.tools import ToolRegistry

    registry = ToolRegistry()
    assert registry.is_empty()

    def my_tool() -> str:
        """A dummy tool."""
        return "result"

    registry.register(my_tool)
    assert not registry.is_empty()
    assert len(registry.tools()) == 1
