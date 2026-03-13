"""Tests for the LLM-based bash guard."""

from typing import final
from unittest.mock import patch

from mother.tools.bash_guard import classify_command, parse_label


@final
class _FakeResponse:
    def __init__(self, text: str) -> None:
        self._text: str = text

    def text(self) -> str:
        return self._text


@final
class _FakeModel:
    def __init__(self, text: str) -> None:
        self._text: str = text

    def prompt(
        self,
        prompt: str | None = None,
        *,
        system: str | None = None,
        stream: bool = True,
        temperature: float = 0.0,
    ) -> _FakeResponse:
        _ = prompt
        _ = system
        _ = stream
        _ = temperature
        return _FakeResponse(self._text)


def test_parse_label_prefers_final_label_line():
    label, canonical = parse_label("thinking\nLABEL: Warning\nLABEL: OK\n")
    assert label == "OK"
    assert canonical is True


def test_parse_label_supports_common_local_model_typo():
    label, canonical = parse_label("LABEL: Warrning")
    assert label == "Warning"
    assert canonical is False


def test_classify_command_returns_ok_label():
    with patch(
        "mother.tools.bash_guard.llm.get_model",
        return_value=_FakeModel("LABEL: OK"),
    ):
        decision = classify_command("ls -al", model_name="local_guard_ok")
    assert decision.label == "OK"
    assert decision.should_run is True
    assert decision.error is None


def test_classify_command_fails_closed_when_output_is_unparsed():
    with patch(
        "mother.tools.bash_guard.llm.get_model",
        return_value=_FakeModel("I refuse to answer"),
    ):
        decision = classify_command("ls -al", model_name="local_guard_unparsed")
    assert decision.label == "Fatal"
    assert decision.should_run is False
    assert decision.error is not None
    assert "Could not parse bash guard label" in decision.error


def test_classify_command_fails_closed_when_model_load_fails():
    with patch(
        "mother.tools.bash_guard.llm.get_model",
        side_effect=RuntimeError("missing model local_guard_missing"),
    ):
        decision = classify_command("ls -al", model_name="local_guard_missing")
    assert decision.label == "Fatal"
    assert decision.should_run is False
    assert decision.error is not None
    assert "Failed to load bash guard model" in decision.error
