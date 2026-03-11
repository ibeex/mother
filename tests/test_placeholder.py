"""Tests for the mother package."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from mother import DEFAULT_MODEL, DEFAULT_SYSTEM, MotherApp, cli


def test_default_model_constant():
    assert DEFAULT_MODEL == "4-claude"


def test_default_system_constant():
    assert "Mother" in DEFAULT_SYSTEM
    assert len(DEFAULT_SYSTEM) > 0


def test_mother_app_defaults():
    app = MotherApp()
    assert app.model_name == DEFAULT_MODEL
    assert app.system_prompt == DEFAULT_SYSTEM


def test_mother_app_custom_args():
    app = MotherApp(model_name="gpt-4o-mini", system="Custom system prompt.")
    assert app.model_name == "gpt-4o-mini"
    assert app.system_prompt == "Custom system prompt."


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--system" in result.output


def test_cli_default_wiring():
    runner = CliRunner()
    with patch("mother.mother.MotherApp") as mock_app_cls:
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        runner.invoke(cli, [])
        mock_app_cls.assert_called_once_with(model_name=DEFAULT_MODEL, system=DEFAULT_SYSTEM)
        mock_app.run.assert_called_once()


def test_cli_custom_model():
    runner = CliRunner()
    with patch("mother.mother.MotherApp") as mock_app_cls:
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        runner.invoke(cli, ["-m", "gpt-4o-mini"])
        mock_app_cls.assert_called_once_with(model_name="gpt-4o-mini", system=DEFAULT_SYSTEM)
        mock_app.run.assert_called_once()


def test_cli_custom_system():
    runner = CliRunner()
    with patch("mother.mother.MotherApp") as mock_app_cls:
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        runner.invoke(cli, ["-s", "Be a pirate."])
        mock_app_cls.assert_called_once_with(model_name=DEFAULT_MODEL, system="Be a pirate.")
        mock_app.run.assert_called_once()
