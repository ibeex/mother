"""Tests for MotherApp initialization defaults and constructor wiring."""

from mother import DEFAULT_MODEL, DEFAULT_SYSTEM, MotherApp, MotherConfig


def test_mother_app_defaults() -> None:
    app = MotherApp()
    assert app.config.model == DEFAULT_MODEL
    assert app.config.theme == "catppuccin-mocha"
    assert app.theme == "catppuccin-mocha"
    assert app.config.system_prompt == DEFAULT_SYSTEM
    assert app.config.reasoning_effort == "medium"


def test_mother_app_custom_args() -> None:
    app = MotherApp(model_name="gpt-4o-mini", system="Custom system prompt.")
    assert app.config.model == "gpt-4o-mini"
    assert app.config.system_prompt == "Custom system prompt."


def test_mother_app_config_kwarg() -> None:
    config = MotherConfig(model="gpt-4o", theme="textual-dark", system_prompt="Be brief.")
    app = MotherApp(config=config)
    assert app.config.model == "gpt-4o"
    assert app.config.theme == "textual-dark"
    assert app.theme == "textual-dark"
    assert app.config.system_prompt == "Be brief."
    assert app.config.reasoning_effort == "medium"
