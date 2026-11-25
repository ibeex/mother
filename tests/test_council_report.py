"""Compare recorded ballots and labeled feedback, independently of live models."""

from dataclasses import replace

from mother.council import (
    CouncilAssessment,
    CouncilCandidateResponse,
    CouncilPeerReview,
    CouncilResult,
    calculate_aggregate_rankings,
)
from mother.council_report import comparison_markdown, comparison_sources_markdown


def _comparison(*reviews: CouncilPeerReview, fallback: bool = False) -> str:
    candidates = tuple(
        CouncilCandidateResponse(f"Response {label}", model, "draft")
        for label, model in zip("ABC", ("opus", "astra", "gemini-4-neon"), strict=True)
    )
    return comparison_markdown(
        CouncilResult(
            final_text="answer",
            judge_model_id="judge",
            stage1=candidates,
            stage2=reviews,
            aggregate_rankings=calculate_aggregate_rankings(reviews),
            label_to_model={item.label: item.model_id for item in candidates},
            used_fallback=fallback,
        )
    )


def _ballot(order: str, model: str, comments: str) -> CouncilPeerReview:
    return CouncilPeerReview(model, comments, tuple(f"Response {label}" for label in order))


def test_comparison_sources_group_by_set_not_order() -> None:
    result = CouncilResult(
        final_text="answer",
        judge_model_id="judge",
        stage1=(),
        stage2=tuple(_ballot("AB", name, "") for name in ("one", "two", "three")),
        aggregate_rankings=(),
        label_to_model={"Response A": "opus"},
    )

    def sources(a: tuple[int, ...], b: tuple[int, ...]) -> str:
        notes = (
            CouncilAssessment("Response A", "note", a),
            CouncilAssessment("Disagreement", "note", b),
        )
        return comparison_sources_markdown(replace(result, assessments=notes))

    assert comparison_sources_markdown(result) == ""
    assert (
        sources((1, 2, 3), (3, 1, 2)) == "The judge cites all 3 reviews for every comparison note."
    )
    assert sources((1, 3), (3, 1)) == "The judge cites Reviews 1, 3 for every comparison note."
    differing = sources((1,), (2, 3))
    assert "- opus: Review 1" in differing
    assert "- Disagreement: Reviews 2, 3" in differing


def test_split_vote_shows_favorite_and_dissenting_reason() -> None:
    report = _comparison(
        _ballot(
            "CAB",
            "opus",
            "**Response C**\nGood balance of brevity and substance. More detail follows.",
        ),
        _ballot("BCA", "astra", "**Response B:** Best plain-language explanation of an ISA."),
        _ballot("CAB", "gemini-4-neon", "FINAL RANKING:\n1. Response C"),
    )
    assert "**gemini-4-neon** ranked highest — 2 of 3 first-place votes." in report
    assert "Response C" not in report
    assert "opus: “Good balance of brevity and substance.”" in report
    assert "astra: “Best plain-language explanation of an ISA.”" in report
    assert "Average rank · lower is better" in report
    assert "More detail follows" not in report
    assert report.count(" · 2.33") == 2
    assert "FINAL RANKING" not in report


def test_partial_tied_ballots_and_unlabeled_prose() -> None:
    report = _comparison(
        _ballot("AB", "one", "Response A and B take different approaches."),
        _ballot("BA", "two", "No per-answer comment."),
    )
    assert "**opus, astra** tied for best average rank" in report
    assert "peer favorite" not in report
    assert "gemini-4-neon** · not ranked" in report
    assert "different approaches" not in report


def test_equal_first_places_and_named_comparisons() -> None:
    report = _comparison(
        _ballot("ACB", "opus", ""),
        _ballot("BCA", "astra", "Response B: More concise than A, like Response C."),
        _ballot("CAB", "gemini-4-neon", "Response C: A C compiler is useful."),
    )
    assert "Each answer received one first-place vote." in report
    assert "More concise than opus, like gemini-4-neon." in report
    assert "A C compiler is useful." in report


def test_fallback_without_reviews_does_not_imply_consensus() -> None:
    report = _comparison(fallback=True)
    assert "candidate answer" in report
    assert "No peer ranking available" in report
    assert "peer favorite" not in report
