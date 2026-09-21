"""Tests for Mother's CLI entrypoint."""

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from mother import MotherConfig, cli
from mother.models import ModelEntry
from mother.session import SessionManager


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
    assert "--init-config" in result.output
    assert "--print-config-path" in result.output


def test_cli_init_config_creates_default_template(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "config.toml"
    legacy_path = tmp_path / "legacy-config.toml"

    with (
        patch("mother.mother.CONFIG_FILE", config_path),
        patch("mother.mother.LEGACY_CONFIG_FILE", legacy_path),
    ):
        result = runner.invoke(cli, ["--init-config"])

    assert result.exit_code == 0
    assert f"Created config: {config_path}" in result.output
    assert config_path.exists()
    assert "[[models]]" in config_path.read_text(encoding="utf-8")


def test_cli_init_config_respects_existing_legacy_config(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "config.toml"
    legacy_path = tmp_path / "legacy-config.toml"
    _ = legacy_path.write_text('model = "legacy"\n', encoding="utf-8")

    with (
        patch("mother.mother.CONFIG_FILE", config_path),
        patch("mother.mother.LEGACY_CONFIG_FILE", legacy_path),
    ):
        result = runner.invoke(cli, ["--init-config"])

    assert result.exit_code == 0
    assert f"Legacy config already exists: {legacy_path}" in result.output
    assert not config_path.exists()


def test_cli_print_config_path_prefers_existing_legacy_config(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "config.toml"
    legacy_path = tmp_path / "legacy-config.toml"
    _ = legacy_path.write_text('model = "legacy"\n', encoding="utf-8")

    with (
        patch("mother.mother.CONFIG_FILE", config_path),
        patch("mother.mother.LEGACY_CONFIG_FILE", legacy_path),
    ):
        result = runner.invoke(cli, ["--print-config-path"])

    assert result.exit_code == 0
    assert result.output.strip() == str(legacy_path)


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


def test_cli_resumes_session_by_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    config = MotherConfig(model="gpt-5", models=[_test_model_entry()])
    sessions_dir = tmp_path / "sessions"
    monkeypatch.chdir(tmp_path)
    source = SessionManager.create(sessions_dir=sessions_dir, cwd=Path.cwd(), model_name="gpt-5")
    source.append("user", "resume me")
    session_id = source.header.get("id")
    assert isinstance(session_id, str)
    with (
        patch("mother.mother.load_config", return_value=config),
        patch("mother.mother.MotherApp") as mock_app_cls,
        patch("mother.session.DEFAULT_SESSIONS_DIR", sessions_dir),
    ):
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        result = runner.invoke(cli, ["--session", session_id])

    assert result.exit_code == 0
    passed_manager = cast(SessionManager, mock_app_cls.call_args.kwargs["session_manager"])
    assert passed_manager.path == source.path
    assert mock_app_cls.call_args.kwargs["loaded_session"] is True
    mock_app.run.assert_called_once()  # pyright: ignore[reportAny]


def test_cli_forks_session_by_id_into_a_distinct_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    config = MotherConfig(model="gpt-5", models=[_test_model_entry()])
    sessions_dir = tmp_path / "sessions"
    monkeypatch.chdir(tmp_path)
    source = SessionManager.create(sessions_dir=sessions_dir, cwd=Path.cwd(), model_name="gpt-5")
    source.append("user", "fork me")
    source.append("assistant", "forked")
    session_id = source.header.get("id")
    assert isinstance(session_id, str)
    with (
        patch("mother.mother.load_config", return_value=config),
        patch("mother.mother.MotherApp") as mock_app_cls,
        patch("mother.session.DEFAULT_SESSIONS_DIR", sessions_dir),
    ):
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        result = runner.invoke(cli, ["--fork", session_id])

    assert result.exit_code == 0
    fork = cast(SessionManager, mock_app_cls.call_args.kwargs["session_manager"])
    assert fork.path != source.path
    assert [
        (entry["role"], entry["content"])
        for entry in fork.load_entries()
        if entry["type"] == "message"
    ] == [("user", "fork me"), ("assistant", "forked")]
    fork_event = next(entry for entry in fork.load_entries() if entry["type"] == "event")
    assert fork_event["name"] == "forked_from"
    assert fork_event["details"]["id"] == session_id


def test_cli_reports_an_unknown_session_id() -> None:
    runner = CliRunner()
    config = MotherConfig(model="gpt-5", models=[_test_model_entry()])
    with (
        patch("mother.mother.load_config", return_value=config),
        patch("mother.mother.MotherApp") as mock_app_cls,
    ):
        result = runner.invoke(cli, ["--session", "does-not-exist"])

    assert result.exit_code == 0
    assert "was not found as a path or ID" in result.output
    mock_app_cls.assert_not_called()


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
