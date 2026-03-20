"""Tests for the LLM-based bash guard."""

from typing import final
from unittest.mock import patch

from mother.tools.bash_guard import classify_command, parse_label


@final
class _FakeResult:
    def __init__(self, output: object) -> None:
        self.output = output


@final
class _FakeAgent:
    def __init__(self, output: object) -> None:
        self.output = output

    def run_sync(
        self,
        user_prompt: str | None = None,
        *,
        instructions: object = None,
        model_settings: object = None,
    ) -> _FakeResult:
        _ = user_prompt
        _ = instructions
        _ = model_settings
        return _FakeResult(self.output)


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
        "mother.tools.bash_guard._get_guard_agent",
        return_value=_FakeAgent("LABEL: OK"),
    ):
        decision = classify_command("ls -al", model_name="local_guard_ok")
    assert decision.label == "OK"
    assert decision.should_run is True
    assert decision.error is None


def test_classify_command_fails_closed_when_output_is_unparsed():
    with patch(
        "mother.tools.bash_guard._get_guard_agent",
        return_value=_FakeAgent("I refuse to answer"),
    ):
        decision = classify_command("ls -al", model_name="local_guard_unparsed")
    assert decision.label == "Fatal"
    assert decision.should_run is False
    assert decision.error is not None
    assert "Could not parse bash guard label" in decision.error


def test_classify_command_fails_closed_when_model_load_fails():
    with patch(
        "mother.tools.bash_guard._get_guard_agent",
        side_effect=RuntimeError("missing model local_guard_missing"),
    ):
        decision = classify_command("ls -al", model_name="local_guard_missing")
    assert decision.label == "Fatal"
    assert decision.should_run is False
    assert decision.error is not None
    assert "Failed to load bash guard model" in decision.error
