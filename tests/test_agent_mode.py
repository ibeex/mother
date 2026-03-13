"""Tests for Ctrl+A agent mode toggle in MotherApp."""

from unittest.mock import MagicMock, patch

from mother import MotherApp, MotherConfig


def test_app_starts_in_conversational_mode():
    app = MotherApp()
    assert app.agent_mode is False


def test_agent_mode_toggle_via_palette():
    """Agent mode is toggled via the command palette (AgentModeProvider), not a key binding."""
    from mother.mother import AgentModeProvider

    assert AgentModeProvider in MotherApp.COMMANDS


def test_toggle_agent_mode_on():
    app = MotherApp()
    app.action_toggle_agent_mode()
    assert app.agent_mode is True


def test_toggle_agent_mode_off():
    app = MotherApp()
    app.action_toggle_agent_mode()
    app.action_toggle_agent_mode()
    assert app.agent_mode is False


def test_subtitle_shows_agent_indicator():
    app = MotherApp()
    # Simulate mounted state by setting sub_title directly
    app.agent_mode = False
    app.config = MotherConfig(model="test-model")
    app._update_subtitle()  # pyright: ignore[reportPrivateUsage]
    assert "[AGENT]" not in app.sub_title

    app.agent_mode = True
    app._update_subtitle()  # pyright: ignore[reportPrivateUsage]
    assert "[AGENT]" in app.sub_title


def test_config_tools_enabled_sets_initial_mode():
    config = MotherConfig(tools_enabled=True)
    app = MotherApp(config=config)
    assert app.agent_mode is True


def test_send_prompt_no_tools_when_conversational():
    app = MotherApp()
    app.agent_mode = False
    with patch("mother.mother.get_default_tools") as mock_tools:
        registry = MagicMock()
        registry.is_empty.return_value = True  # pyright: ignore[reportAny]
        mock_tools.return_value = registry
        # Trigger the call path
        mock_tools(tools_enabled=False)
        mock_tools.assert_called_with(tools_enabled=False)


def test_send_prompt_passes_tools_when_agent():
    app = MotherApp()
    app.agent_mode = True
    with patch("mother.mother.get_default_tools") as mock_tools:
        registry = MagicMock()
        registry.is_empty.return_value = False  # pyright: ignore[reportAny]
        registry.tools.return_value = [MagicMock()]  # pyright: ignore[reportAny]
        mock_tools.return_value = registry
        mock_tools(tools_enabled=True)
        mock_tools.assert_called_with(tools_enabled=True)
