"""Tests for the versioned User-Agent Mother identifies itself with."""

from importlib.metadata import PackageNotFoundError

import pytest

from mother import user_agent


def test_user_agent_carries_the_package_version() -> None:
    agent = user_agent.user_agent()

    assert agent == f"mother/{user_agent.installed_version()}"
    assert agent.startswith("mother/")


def test_unknown_version_falls_back_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_name: str) -> str:
        raise PackageNotFoundError("mother")

    monkeypatch.setattr(user_agent, "version", _raise)
    user_agent.installed_version.cache_clear()
    user_agent.user_agent.cache_clear()
    try:
        assert user_agent.user_agent() == "mother/0.0.0"
    finally:
        user_agent.installed_version.cache_clear()
        user_agent.user_agent.cache_clear()
