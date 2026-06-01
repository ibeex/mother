"""Tests for Mother's deep-research workflow helpers."""

from mother.deep_research import aggregate_turn_usage, is_research_approval
from mother.stats import TurnUsage


def test_research_approval_accepts_short_confirmation() -> None:
    assert is_research_approval("yes")
    assert is_research_approval("Go ahead, please")
    assert is_research_approval("I approve this")
    assert is_research_approval("looks good, proceed")
    assert not is_research_approval("add more primary sources")
    assert not is_research_approval("I do not approve this yet")


def test_aggregate_turn_usage_combines_internal_research_calls() -> None:
    usage = aggregate_turn_usage(
        [
            TurnUsage(
                request_tokens=10, response_tokens=4, tool_calls_started=2, duration_seconds=1.5
            ),
            TurnUsage(
                request_tokens=5, response_tokens=6, tool_calls_finished=2, duration_seconds=2.0
            ),
        ]
    )

    assert usage.request_tokens == 15
    assert usage.response_tokens == 10
    assert usage.tool_calls_started == 2
    assert usage.tool_calls_finished == 2
    assert usage.duration_seconds == 3.5
