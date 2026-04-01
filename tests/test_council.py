"""Tests for anonymous council helpers."""

import asyncio
from pathlib import Path
from typing import override
from unittest.mock import patch

import mother.council as council_module
from mother.council import (
    CouncilAggregateRanking,
    CouncilCandidateResponse,
    CouncilPeerReview,
    CouncilProgressUpdate,
    CouncilResult,
    CouncilRunner,
    build_stage3_prompt,
    calculate_aggregate_rankings,
    parse_ranking_from_text,
)
from mother.models import ModelEntry


def test_parse_ranking_from_text_prefers_final_ranking_section() -> None:
    ranking = parse_ranking_from_text(
        """
Response A is decent.
Response B is best.

FINAL RANKING:
1. Response B
2. Response A
        """.strip()
    )

    assert ranking == ("Response B", "Response A")


def test_parse_ranking_from_text_falls_back_to_any_response_labels() -> None:
    ranking = parse_ranking_from_text("Best to worst: Response C, Response A, Response B")

    assert ranking == ("Response C", "Response A", "Response B")


def test_calculate_aggregate_rankings_ignores_unknown_and_duplicate_labels() -> None:
    reviews = (
        CouncilPeerReview(
            reviewer_model_id="gpt-5",
            text="review one",
            parsed_ranking=("Response B", "Response A", "Response B"),
        ),
        CouncilPeerReview(
            reviewer_model_id="opus",
            text="review two",
            parsed_ranking=("Response A", "Response Z", "Response B"),
        ),
    )

    aggregate = calculate_aggregate_rankings(
        reviews,
        valid_labels={"Response A", "Response B"},
    )

    assert aggregate == (
        CouncilAggregateRanking(label="Response A", average_rank=1.5, rankings_count=2),
        CouncilAggregateRanking(label="Response B", average_rank=2.0, rankings_count=2),
    )


def test_build_stage3_prompt_keeps_model_ids_out_of_judge_prompt() -> None:
    prompt = build_stage3_prompt(
        user_question="How should we ship this feature?",
        conversation_context="User: We need a plan.\n\nAssistant: We should keep it simple.",
        supplemental_context="Shell command: git status",
        stage1_results=(
            CouncilCandidateResponse(
                label="Response A",
                model_id="gpt-5",
                text="Ship it in two phases.",
            ),
            CouncilCandidateResponse(
                label="Response B",
                model_id="opus",
                text="Ship it behind a flag first.",
            ),
        ),
        stage2_reviews=(
            CouncilPeerReview(
                reviewer_model_id="g3",
                text="Response B is safer.\n\nFINAL RANKING:\n1. Response B\n2. Response A",
                parsed_ranking=("Response B", "Response A"),
            ),
        ),
        aggregate_rankings=(
            CouncilAggregateRanking(label="Response B", average_rank=1.0, rankings_count=1),
        ),
    )

    assert "gpt-5" not in prompt
    assert "opus" not in prompt
    assert "g3" not in prompt
    assert "Response A" in prompt
    assert "Response B" in prompt


def test_council_progress_status_text_reports_stage_and_counts() -> None:
    assert CouncilProgressUpdate.stage1(0, 3).status_text() == (
        "Council stage 1/3 · collecting answers · 0/3 complete"
    )
    assert CouncilProgressUpdate.stage2(2, 4).status_text() == (
        "Council stage 2/3 · running peer review · 2/4 complete"
    )
    assert CouncilProgressUpdate.stage3().status_text() == "Council stage 3/3 · finalizing answer"


def test_council_runner_reports_progress_and_preserves_member_order() -> None:
    members = (
        ModelEntry(id="slow", name="slow", api_type="openai-responses"),
        ModelEntry(id="fast", name="fast", api_type="openai-responses"),
        ModelEntry(id="mid", name="mid", api_type="openai-responses"),
    )
    updates: list[str] = []

    class _ProgressRunner(CouncilRunner):
        @override
        async def _query_model(
            self,
            model: ModelEntry,
            *,
            prompt_text: str,
            system_prompt: str,
        ) -> council_module._CouncilModelReply | None:  # pyright: ignore[reportPrivateUsage]
            if "answer-generation step" in system_prompt:
                await asyncio.sleep({"slow": 0.03, "fast": 0.0, "mid": 0.01}[model.id])
                return council_module._CouncilModelReply(  # pyright: ignore[reportPrivateUsage]
                    model_id=model.id,
                    text=f"answer {model.id}",
                )
            if "careful reviewer comparing anonymous candidate answers" in system_prompt:
                await asyncio.sleep({"slow": 0.02, "fast": 0.0, "mid": 0.01}[model.id])
                return council_module._CouncilModelReply(  # pyright: ignore[reportPrivateUsage]
                    model_id=model.id,
                    text="Response B is best.\n\nFINAL RANKING:\n1. Response B\n2. Response A\n3. Response C",
                )
            if "internal council synthesis step" in system_prompt:
                return council_module._CouncilModelReply(  # pyright: ignore[reportPrivateUsage]
                    model_id=model.id,
                    text="final answer",
                )
            raise AssertionError(f"Unexpected system prompt: {system_prompt}")

    runner = _ProgressRunner(
        members=members,
        judge=ModelEntry(id="judge", name="judge", api_type="anthropic"),
        base_system_prompt="You are helpful.",
        reasoning_effort="high",
        openai_reasoning_summary="auto",
        cwd=Path("/tmp"),
        on_progress=lambda update: updates.append(update.status_text()),
    )

    result = asyncio.run(runner.run(user_question="How should we ship this?"))

    assert [candidate.model_id for candidate in result.stage1] == ["slow", "fast", "mid"]
    assert updates == [
        "Council stage 1/3 · collecting answers · 0/3 complete",
        "Council stage 1/3 · collecting answers · 1/3 complete",
        "Council stage 1/3 · collecting answers · 2/3 complete",
        "Council stage 1/3 · collecting answers · 3/3 complete",
        "Council stage 2/3 · running peer review · 0/3 complete",
        "Council stage 2/3 · running peer review · 1/3 complete",
        "Council stage 2/3 · running peer review · 2/3 complete",
        "Council stage 2/3 · running peer review · 3/3 complete",
        "Council stage 3/3 · finalizing answer",
    ]


def test_council_query_model_logs_failures_with_context() -> None:
    runner = CouncilRunner(
        members=(),
        judge=ModelEntry(id="judge", name="judge", api_type="anthropic"),
        base_system_prompt="You are helpful.",
        reasoning_effort="high",
        openai_reasoning_summary="auto",
        cwd=Path("/tmp"),
    )
    model = ModelEntry(id="gpt-5", name="gpt-5", api_type="openai-responses")

    class _FailingRuntime:
        async def run_stream(self, **_: object) -> object:
            raise RuntimeError("boom")

    with (
        patch("mother.council.ChatRuntime", return_value=_FailingRuntime()),
        patch("mother.council.logger.exception") as log_exception,
    ):
        result = asyncio.run(
            runner._query_model(  # pyright: ignore[reportPrivateUsage]
                model,
                prompt_text="Question",
                system_prompt="System",
            )
        )

    assert result is None
    log_exception.assert_called_once()


def test_council_result_trace_sections_include_stage_details_and_model_mapping() -> None:
    result = CouncilResult(
        final_text="Ship it behind a flag.",
        judge_model_id="opus",
        stage1=(
            CouncilCandidateResponse(
                label="Response A",
                model_id="gpt-5",
                text="Two-phase rollout.",
            ),
            CouncilCandidateResponse(
                label="Response B",
                model_id="g3",
                text="Ship behind a flag.",
            ),
        ),
        stage2=(
            CouncilPeerReview(
                reviewer_model_id="opus",
                text="Response B is safer.\n\nFINAL RANKING:\n1. Response B\n2. Response A",
                parsed_ranking=("Response B", "Response A"),
            ),
        ),
        aggregate_rankings=(
            CouncilAggregateRanking(label="Response B", average_rank=1.0, rankings_count=1),
        ),
        label_to_model={"Response A": "gpt-5", "Response B": "g3"},
        duration_seconds=12.5,
    )

    sections = result.trace_sections()
    assert sections[0].title == "Council · Stage 1 · Response A · gpt-5"
    assert sections[0].text == "Two-phase rollout."
    assert sections[1].title == "Council · Stage 1 · Response B · g3"
    assert "Parsed ranking: Response B · g3 > Response A · gpt-5" in sections[2].text
    assert sections[3].title == "Council · Stage 2 · Aggregate rankings"
    assert "Response B · g3" in sections[3].text
    assert sections[4].title == "Council · Stage 3 · Judge metadata"
    assert "Judge: opus" in sections[4].text
    assert "Duration: 12.50s" in sections[4].text

    event_details = result.to_event_details()
    trace_sections = event_details.get("trace_sections")
    assert isinstance(trace_sections, list)
    assert trace_sections[0] == {
        "title": "Council · Stage 1 · Response A · gpt-5",
        "text": "Two-phase rollout.",
    }
