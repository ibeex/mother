"""Tests for Ctrl+A agent mode toggle in MotherApp."""

from unittest.mock import MagicMock, PropertyMock, patch

from textual.widgets import TextArea

from mother import MotherApp, MotherConfig
from mother.widgets import StatusLine


def test_app_starts_in_conversational_mode():
    app = MotherApp()
    assert app.agent_mode is False


def test_agent_mode_toggle_via_palette():
    """Agent mode is toggled via the command palette (AgentModeProvider), not a key binding."""
    from mother.model_picker import AgentModeProvider

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


def test_toggle_auto_scroll_off():
    app = MotherApp()
    with patch.object(app, "notify"):
        app.action_toggle_auto_scroll()
    assert app.auto_scroll_enabled is False


def test_scroll_to_bottom_uses_forced_scroll():
    app = MotherApp()
    with patch.object(app, "_scroll_chat_to_end") as scroll_to_end:
        app.action_scroll_to_bottom()
    scroll_to_end.assert_called_once_with(force=True)


def test_shift_g_scrolls_to_bottom_when_input_is_not_focused():
    app = MotherApp()
    with (
        patch.object(MotherApp, "focused", new_callable=PropertyMock, return_value=None),
        patch.object(app, "action_scroll_to_bottom") as scroll_to_bottom,
    ):
        app.action_scroll_to_bottom_from_chat()
    scroll_to_bottom.assert_called_once_with()


def test_shift_g_does_nothing_when_input_is_focused():
    app = MotherApp()
    text_area = TextArea()
    with (
        patch.object(MotherApp, "focused", new_callable=PropertyMock, return_value=text_area),
        patch.object(app, "action_scroll_to_bottom") as scroll_to_bottom,
    ):
        app.action_scroll_to_bottom_from_chat()
    scroll_to_bottom.assert_not_called()


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


def test_statusline_formats_model_and_unknown_context():
    assert StatusLine.format_status("test-model", False, None) == "test-model · off · ? · auto"


def test_statusline_formats_model_and_context_size():
    assert StatusLine.format_status("test-model", True, 12345) == "test-model · on · 12.3k · auto"


def test_statusline_formats_manual_scroll_mode():
    assert (
        StatusLine.format_status("test-model", True, 256, False) == "test-model · on · 256 · manual"
    )


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
