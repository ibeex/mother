"""Integration tests: config allowlist, model switch, exports."""

from pathlib import Path

from mother import MotherApp
from mother.config import load_config
from mother.tools.bash_tool import DEFAULT_ALLOWLIST


def test_config_allowlist_from_toml(tmp_path: Path):
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text('allowlist = ["ls", "cat", "grep"]\n')
    config = load_config(config_file)
    assert config.allowlist == frozenset({"ls", "cat", "grep"})


def test_config_default_allowlist(tmp_path: Path):
    config_file = tmp_path / "config.toml"
    config = load_config(config_file)
    assert config.allowlist == DEFAULT_ALLOWLIST


def test_config_ca_bundle_path_from_toml(tmp_path: Path):
    config_file = tmp_path / "config.toml"
    _ = config_file.write_text('ca_bundle_path = "/etc/ssl/certs/ib_cert.pem"\n')
    config = load_config(config_file)
    assert config.ca_bundle_path == "/etc/ssl/certs/ib_cert.pem"


def test_model_switch_preserves_agent_mode():
    app = MotherApp()
    app.agent_mode = True
    app.action_switch_model("gpt-4o-mini")
    assert app.agent_mode is True


def test_model_switch_syncs_tools_enabled_to_runtime_state():
    """config.tools_enabled should reflect the live agent_mode after a switch."""
    app = MotherApp()
    app.agent_mode = True
    app.action_switch_model("gpt-4o-mini")
    assert app.config.tools_enabled is True

    app.agent_mode = False
    app.action_switch_model("gpt-5")
    assert app.config.tools_enabled is False


def test_exports_updated():
    from mother import (
        BashExecution,
        ModelsCommand,
        NormalPrompt,
        SaveSessionCommand,
        SessionManager,
        ShellCommand,
    )

    assert BashExecution is not None
    assert ModelsCommand is not None
    assert NormalPrompt is not None
    assert SaveSessionCommand is not None
    assert SessionManager is not None
    assert ShellCommand is not None
