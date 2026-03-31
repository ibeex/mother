"""Tests for anonymous council helpers."""

from mother.council import (
    CouncilAggregateRanking,
    CouncilCandidateResponse,
    CouncilPeerReview,
    CouncilResult,
    build_stage3_prompt,
    calculate_aggregate_rankings,
    parse_ranking_from_text,
)


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
