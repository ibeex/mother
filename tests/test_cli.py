"""Tests for Mother's CLI entrypoint."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from mother import MotherConfig, cli
from mother.models import ModelEntry


def _test_model_entry(model_id: str = "gpt-5") -> ModelEntry:
    return ModelEntry(
        id=model_id,
        name=model_id,
        api_type="openai-responses",
    )


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--system" in result.output


def test_cli_default_wiring() -> None:
    runner = CliRunner()
    config = MotherConfig(model="gpt-5", models=[_test_model_entry()])
    with (
        patch("mother.mother.load_config") as mock_load,
        patch("mother.mother.MotherApp") as mock_app_cls,
    ):
        mock_load.return_value = config
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        _ = runner.invoke(cli, [])
        mock_app_cls.assert_called_once()
        mock_app.run.assert_called_once()  # pyright: ignore[reportAny]


def test_cli_custom_model() -> None:
    runner = CliRunner()
    with (
        patch("mother.mother.load_config") as mock_load,
        patch("mother.mother.MotherApp") as mock_app_cls,
    ):
        mock_load.return_value = MotherConfig(models=[_test_model_entry("gpt-4o-mini")])
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        _ = runner.invoke(cli, ["-m", "gpt-4o-mini"])
        call_kwargs = mock_app_cls.call_args
        passed_config: MotherConfig = call_kwargs.kwargs["config"]  # pyright: ignore[reportAny]
        assert passed_config.model == "gpt-4o-mini"


def test_cli_custom_system() -> None:
    runner = CliRunner()
    with (
        patch("mother.mother.load_config") as mock_load,
        patch("mother.mother.MotherApp") as mock_app_cls,
    ):
        mock_load.return_value = MotherConfig(model="gpt-5", models=[_test_model_entry()])
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        _ = runner.invoke(cli, ["-s", "Be a pirate."])
        call_kwargs = mock_app_cls.call_args
        passed_config: MotherConfig = call_kwargs.kwargs["config"]  # pyright: ignore[reportAny]
        assert passed_config.system_prompt == "Be a pirate."


def test_cli_exits_before_tui_when_no_models_are_configured() -> None:
    runner = CliRunner()
    with (
        patch("mother.mother.load_config", return_value=MotherConfig()),
        patch("mother.mother.MotherApp") as mock_app_cls,
    ):
        result = runner.invoke(cli, [])
    assert result.exit_code == 0
    assert "No models configured" in result.output
    mock_app_cls.assert_not_called()


def test_cli_exits_before_tui_when_default_model_is_not_set() -> None:
    runner = CliRunner()
    with (
        patch(
            "mother.mother.load_config",
            return_value=MotherConfig(models=[_test_model_entry()]),
        ),
        patch("mother.mother.MotherApp") as mock_app_cls,
    ):
        result = runner.invoke(cli, [])
    assert result.exit_code == 0
    assert "No default model selected" in result.output
    mock_app_cls.assert_not_called()
