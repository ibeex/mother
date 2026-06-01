"""Tests for Mother's deep-research workflow helpers."""

from mother.deep_research import (
    DeepResearchRunner,
    DeepResearchStats,
    aggregate_turn_usage,
    is_research_approval,
    normalize_query,
    normalize_url,
    parse_research_queries,
)
from mother.models import ModelEntry
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


def test_parse_research_queries_deduplicates_numbered_and_bulleted_lines() -> None:
    queries = parse_research_queries(
        """
        Queries:
        1. best primary sources for sodium ion battery cost 2026
        - sodium ion battery cost primary sources 2026
        Query 3: sodium ion battery cycle life commercial deployments
        * LFP vs sodium ion storage tradeoffs
        """,
        max_queries=3,
    )

    assert queries == [
        "best primary sources for sodium ion battery cost 2026",
        "sodium ion battery cost primary sources 2026",
        "sodium ion battery cycle life commercial deployments",
    ]


def test_query_and_url_normalization_for_research_state() -> None:
    assert normalize_query("  LFP   vs Sodium-Ion  ") == "lfp vs sodium-ion"
    assert normalize_url("HTTPS://Example.com/Path/?b=1#frag") == "https://example.com/Path?b=1"
    assert normalize_url("https://example.com/path/.") == "https://example.com/path"


def test_deep_research_stats_event_and_markdown_include_metadata() -> None:
    stats = DeepResearchStats(
        rounds=2,
        queries=7,
        urls=4,
        searches=7,
        fetches=5,
        tool_errors=1,
        category="technical",
        partial=True,
    )

    assert stats.to_event_details()["category"] == "technical"
    assert stats.to_event_details()["partial"] is True
    markdown = stats.format_markdown()
    assert "Rounds: 2" in markdown
    assert "Category: technical" in markdown
    assert "partial report" in markdown


def test_round_tool_limit_has_headroom_above_fetch_and_query_budget() -> None:
    runner = DeepResearchRunner(
        ModelEntry(id="test", name="Test", api_type="openai-chat"),
        base_system_prompt="test",
        max_queries_per_round=5,
        fetches_per_round=10,
    )

    assert runner.round_tool_call_limit(5) == 40
    assert runner.round_tool_call_limit(12) == 56
