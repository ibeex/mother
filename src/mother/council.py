"""Anonymous multi-model council orchestration."""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

from mother.interrupts import UserInterruptedError
from mother.models import ModelEntry
from mother.reasoning import build_reasoning_options
from mother.runtime import ChatRuntime
from mother.system_prompt import build_system_prompt


@dataclass(frozen=True, slots=True)
class CouncilCandidateResponse:
    """A stage-1 candidate answer labeled for anonymous review."""

    label: str
    model_id: str
    text: str


@dataclass(frozen=True, slots=True)
class CouncilPeerReview:
    """A stage-2 anonymous review and parsed ranking from one reviewer."""

    reviewer_model_id: str
    text: str
    parsed_ranking: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CouncilAggregateRanking:
    """Average peer-review ranking for one anonymous response label."""

    label: str
    average_rank: float
    rankings_count: int


@dataclass(frozen=True, slots=True)
class CouncilTraceSection:
    """A human-readable council trace section for UI and markdown export."""

    title: str
    text: str


@dataclass(frozen=True, slots=True)
class CouncilResult:
    """Final result and lightweight metadata from a council run."""

    final_text: str
    judge_model_id: str
    stage1: tuple[CouncilCandidateResponse, ...]
    stage2: tuple[CouncilPeerReview, ...]
    aggregate_rankings: tuple[CouncilAggregateRanking, ...]
    label_to_model: dict[str, str]
    used_fallback: bool = False
    fallback_reason: str | None = None
    duration_seconds: float | None = None

    def trace_sections(self) -> tuple[CouncilTraceSection, ...]:
        """Return human-readable trace sections for inspecting council internals."""
        sections: list[CouncilTraceSection] = []

        for candidate in self.stage1:
            sections.append(
                CouncilTraceSection(
                    title=(
                        "Council · Stage 1 · "
                        + _label_with_model(candidate.label, self.label_to_model)
                    ),
                    text=candidate.text,
                )
            )

        for index, review in enumerate(self.stage2, start=1):
            lines = [
                f"Reviewer: {review.reviewer_model_id}",
                f"Parsed ranking: {_format_parsed_ranking(review.parsed_ranking, self.label_to_model)}",
                "",
                review.text,
            ]
            sections.append(
                CouncilTraceSection(
                    title=f"Council · Stage 2 · Review {index}",
                    text="\n".join(lines),
                )
            )

        aggregate_text = _format_aggregate_rankings(
            self.aggregate_rankings,
            self.label_to_model,
        )
        if aggregate_text:
            sections.append(
                CouncilTraceSection(
                    title="Council · Stage 2 · Aggregate rankings",
                    text=aggregate_text,
                )
            )

        metadata_lines = [f"Judge: {self.judge_model_id}"]
        if self.duration_seconds is not None:
            metadata_lines.append(f"Duration: {self.duration_seconds:.2f}s")
        metadata_lines.append(
            "Fallback: "
            + (f"yes ({self.fallback_reason or 'unknown'})" if self.used_fallback else "no")
        )
        metadata_lines.extend(["", "The final synthesized answer is shown in the assistant reply."])
        sections.append(
            CouncilTraceSection(
                title="Council · Stage 3 · Judge metadata",
                text="\n".join(metadata_lines),
            )
        )

        return tuple(sections)

    def to_event_details(self) -> dict[str, object]:
        """Return a JSON-friendly council summary for session logging."""
        return {
            "judge_model": self.judge_model_id,
            "label_to_model": dict(self.label_to_model),
            "stage1_count": len(self.stage1),
            "stage2_count": len(self.stage2),
            "aggregate_rankings": [
                {
                    "label": ranking.label,
                    "average_rank": ranking.average_rank,
                    "rankings_count": ranking.rankings_count,
                }
                for ranking in self.aggregate_rankings
            ],
            "trace_sections": [
                {
                    "title": section.title,
                    "text": section.text,
                }
                for section in self.trace_sections()
            ],
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason or "",
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True, slots=True)
class _CouncilModelReply:
    """A single successful raw model reply during council orchestration."""

    model_id: str
    text: str


def parse_ranking_from_text(ranking_text: str) -> tuple[str, ...]:
    """Parse a ``FINAL RANKING`` section from a stage-2 review."""
    final_section_match = re.search(
        r"FINAL RANKING:\s*(.*)",
        ranking_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    ranking_section = ranking_text if final_section_match is None else final_section_match.group(1)

    numbered_matches = cast(
        list[str],
        re.findall(
            r"\d+\.\s*Response\s+[A-Z]+",
            ranking_section,
            flags=re.IGNORECASE,
        ),
    )
    if numbered_matches:
        return tuple(_normalize_response_label(match) for match in numbered_matches)

    matches = cast(
        list[str],
        re.findall(r"Response\s+[A-Z]+", ranking_section, flags=re.IGNORECASE),
    )
    return tuple(_normalize_response_label(match) for match in matches)


def calculate_aggregate_rankings(
    stage2_reviews: tuple[CouncilPeerReview, ...],
    *,
    valid_labels: set[str] | None = None,
) -> tuple[CouncilAggregateRanking, ...]:
    """Calculate average peer-review positions for each anonymous response label."""
    label_positions: defaultdict[str, list[int]] = defaultdict(list)

    for review in stage2_reviews:
        seen_in_review: set[str] = set()
        for position, label in enumerate(review.parsed_ranking, start=1):
            if label in seen_in_review:
                continue
            if valid_labels is not None and label not in valid_labels:
                continue
            seen_in_review.add(label)
            label_positions[label].append(position)

    aggregate = [
        CouncilAggregateRanking(
            label=label,
            average_rank=round(sum(positions) / len(positions), 2),
            rankings_count=len(positions),
        )
        for label, positions in label_positions.items()
        if positions
    ]
    aggregate.sort(key=lambda item: (item.average_rank, item.label))
    return tuple(aggregate)


def build_stage3_prompt(
    *,
    user_question: str,
    conversation_context: str,
    supplemental_context: str,
    stage1_results: tuple[CouncilCandidateResponse, ...],
    stage2_reviews: tuple[CouncilPeerReview, ...],
    aggregate_rankings: tuple[CouncilAggregateRanking, ...],
) -> str:
    """Build the final anonymized synthesis prompt for the judge model."""
    candidate_block = "\n\n".join(
        f"{candidate.label}:\n{candidate.text}" for candidate in stage1_results
    )
    ranking_summary = (
        _format_aggregate_rankings(aggregate_rankings) or "(no aggregate rankings available)"
    )
    review_block = _format_peer_reviews(stage2_reviews)

    return "\n".join(
        [
            "You are synthesizing anonymous candidate answers into one final reply for the user.",
            "Do not mention the council, peer review, response labels, or model identities.",
            "Use the supplied context when it is relevant.",
            "",
            "Conversation context:",
            _format_optional_block(conversation_context),
            "",
            "Supplemental local context:",
            _format_optional_block(supplemental_context),
            "",
            "User question:",
            user_question,
            "",
            "Anonymous candidate responses:",
            candidate_block,
            "",
            "Peer ranking summary:",
            ranking_summary,
            "",
            "Anonymous peer reviews:",
            review_block,
            "",
            "Write the single best final answer to the user.",
            "Resolve disagreements when possible and say plainly when something is uncertain.",
        ]
    ).strip()


class CouncilRunner:
    """Run a three-stage anonymous council using configured Mother models."""

    members: tuple[ModelEntry, ...]
    judge: ModelEntry
    base_system_prompt: str
    reasoning_effort: str
    openai_reasoning_summary: str
    cwd: Path

    def __init__(
        self,
        *,
        members: tuple[ModelEntry, ...],
        judge: ModelEntry,
        base_system_prompt: str,
        reasoning_effort: str,
        openai_reasoning_summary: str,
        cwd: Path | None = None,
    ) -> None:
        self.members = members
        self.judge = judge
        self.base_system_prompt = base_system_prompt
        self.reasoning_effort = reasoning_effort
        self.openai_reasoning_summary = openai_reasoning_summary
        self.cwd = cwd or Path.cwd()

    async def run(
        self,
        *,
        user_question: str,
        conversation_context: str = "",
        supplemental_context: str = "",
    ) -> CouncilResult:
        """Run stage 1 answers, stage 2 anonymous peer review, and stage 3 synthesis."""
        started_at = perf_counter()
        stage1_prompt = self._build_stage1_prompt(
            user_question=user_question,
            conversation_context=conversation_context,
            supplemental_context=supplemental_context,
        )
        stage1_replies = await self._query_models_parallel(
            self.members,
            prompt_text=stage1_prompt,
            system_prompt=self._member_system_prompt(),
        )
        if not stage1_replies:
            raise RuntimeError("All council members failed to respond.")

        stage1_results = tuple(
            CouncilCandidateResponse(
                label=_response_label(index),
                model_id=reply.model_id,
                text=reply.text,
            )
            for index, reply in enumerate(stage1_replies)
        )
        label_to_model = {candidate.label: candidate.model_id for candidate in stage1_results}

        stage2_results = await self._run_stage2(
            user_question=user_question,
            conversation_context=conversation_context,
            supplemental_context=supplemental_context,
            stage1_results=stage1_results,
        )
        aggregate_rankings = calculate_aggregate_rankings(
            stage2_results,
            valid_labels=set(label_to_model),
        )

        stage3_prompt = build_stage3_prompt(
            user_question=user_question,
            conversation_context=conversation_context,
            supplemental_context=supplemental_context,
            stage1_results=stage1_results,
            stage2_reviews=stage2_results,
            aggregate_rankings=aggregate_rankings,
        )
        judge_reply = await self._query_model(
            self.judge,
            prompt_text=stage3_prompt,
            system_prompt=self._judge_system_prompt(),
        )
        if judge_reply is not None:
            return CouncilResult(
                final_text=judge_reply.text,
                judge_model_id=self.judge.id,
                stage1=stage1_results,
                stage2=stage2_results,
                aggregate_rankings=aggregate_rankings,
                label_to_model=label_to_model,
                duration_seconds=perf_counter() - started_at,
            )

        fallback_text, fallback_reason = _fallback_result_text(stage1_results, aggregate_rankings)
        return CouncilResult(
            final_text=fallback_text,
            judge_model_id=self.judge.id,
            stage1=stage1_results,
            stage2=stage2_results,
            aggregate_rankings=aggregate_rankings,
            label_to_model=label_to_model,
            used_fallback=True,
            fallback_reason=fallback_reason,
            duration_seconds=perf_counter() - started_at,
        )

    async def _run_stage2(
        self,
        *,
        user_question: str,
        conversation_context: str,
        supplemental_context: str,
        stage1_results: tuple[CouncilCandidateResponse, ...],
    ) -> tuple[CouncilPeerReview, ...]:
        if len(stage1_results) < 2:
            return ()

        reviewer_prompt = self._build_stage2_prompt(
            user_question=user_question,
            conversation_context=conversation_context,
            supplemental_context=supplemental_context,
            stage1_results=stage1_results,
        )
        reviewer_replies = await self._query_models_parallel(
            self.members,
            prompt_text=reviewer_prompt,
            system_prompt=self._reviewer_system_prompt(),
        )
        return tuple(
            CouncilPeerReview(
                reviewer_model_id=reply.model_id,
                text=reply.text,
                parsed_ranking=parse_ranking_from_text(reply.text),
            )
            for reply in reviewer_replies
        )

    async def _query_models_parallel(
        self,
        models: tuple[ModelEntry, ...],
        *,
        prompt_text: str,
        system_prompt: str,
    ) -> tuple[_CouncilModelReply, ...]:
        tasks = [
            asyncio.create_task(
                self._query_model(
                    model,
                    prompt_text=prompt_text,
                    system_prompt=system_prompt,
                )
            )
            for model in models
        ]
        results = await asyncio.gather(*tasks)
        return tuple(result for result in results if result is not None)

    async def _query_model(
        self,
        model: ModelEntry,
        *,
        prompt_text: str,
        system_prompt: str,
    ) -> _CouncilModelReply | None:
        runtime = ChatRuntime(model)
        try:
            response = await runtime.run_stream(
                prompt_text=prompt_text,
                system_prompt=system_prompt,
                message_history=[],
                attachments=[],
                tools=[],
                model_settings=build_reasoning_options(
                    model,
                    self.reasoning_effort,
                    self.openai_reasoning_summary,
                ),
                allow_tool_fallback=False,
            )
        except UserInterruptedError:
            raise
        except Exception:
            return None

        text = response.text.strip()
        if not text:
            return None
        return _CouncilModelReply(model_id=model.id, text=text)

    def _member_system_prompt(self) -> str:
        prompt = build_system_prompt(
            self.base_system_prompt,
            mode="chat",
            agent_mode=False,
            cwd=self.cwd,
        )
        return "\n\n".join(
            [
                prompt,
                "This is an internal council answer-generation step.",
                "Answer the user's question directly and do not mention the council workflow.",
            ]
        )

    def _reviewer_system_prompt(self) -> str:
        return build_system_prompt(
            "You are a careful reviewer comparing anonymous candidate answers.",
            mode="chat",
            agent_mode=False,
            cwd=self.cwd,
        )

    def _judge_system_prompt(self) -> str:
        prompt = build_system_prompt(
            self.base_system_prompt,
            mode="chat",
            agent_mode=False,
            cwd=self.cwd,
        )
        return "\n\n".join(
            [
                prompt,
                "This is an internal council synthesis step.",
                "You are reviewing anonymous candidate answers and must not mention labels,",
                "peer review, or model identities in the final user-facing response.",
            ]
        )

    @staticmethod
    def _build_stage1_prompt(
        *,
        user_question: str,
        conversation_context: str,
        supplemental_context: str,
    ) -> str:
        return "\n".join(
            [
                "Use the available context when it is relevant.",
                "Do not mention that you are part of a council or that other models exist.",
                "",
                "Conversation context:",
                _format_optional_block(conversation_context),
                "",
                "Supplemental local context:",
                _format_optional_block(supplemental_context),
                "",
                "User question:",
                user_question,
                "",
                "Write the best answer you can for the user.",
            ]
        ).strip()

    @staticmethod
    def _build_stage2_prompt(
        *,
        user_question: str,
        conversation_context: str,
        supplemental_context: str,
        stage1_results: tuple[CouncilCandidateResponse, ...],
    ) -> str:
        candidate_block = "\n\n".join(
            f"{candidate.label}:\n{candidate.text}" for candidate in stage1_results
        )
        return "\n".join(
            [
                "You are evaluating anonymous candidate answers to a user's question.",
                "Focus on accuracy, completeness, usefulness, and clear reasoning.",
                "Do not guess or mention model identities.",
                "",
                "Conversation context:",
                _format_optional_block(conversation_context),
                "",
                "Supplemental local context:",
                _format_optional_block(supplemental_context),
                "",
                "User question:",
                user_question,
                "",
                "Candidate responses:",
                candidate_block,
                "",
                "Your task:",
                "1. Briefly evaluate each response individually.",
                "2. Then provide a final ranking at the end.",
                "",
                "IMPORTANT: Your final ranking must be formatted exactly like this:",
                "FINAL RANKING:",
                "1. Response A",
                "2. Response B",
                "3. Response C",
                "",
                "Do not add any text after the ranking section.",
            ]
        ).strip()


def _normalize_response_label(raw_text: str) -> str:
    match = re.search(r"Response\s+([A-Z]+)", raw_text, flags=re.IGNORECASE)
    if match is None:
        return raw_text.strip()
    return f"Response {match.group(1).upper()}"


def _response_label(index: int) -> str:
    current = index
    parts: list[str] = []
    while True:
        current, remainder = divmod(current, 26)
        parts.append(chr(65 + remainder))
        if current == 0:
            break
        current -= 1
    suffix = "".join(reversed(parts))
    return f"Response {suffix}"


def _format_optional_block(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "(none)"
    return stripped


def _format_peer_reviews(stage2_reviews: tuple[CouncilPeerReview, ...]) -> str:
    if not stage2_reviews:
        return "(no peer reviews available)"
    return "\n\n".join(
        f"Review {index}:\n{review.text}" for index, review in enumerate(stage2_reviews, start=1)
    )


def _label_with_model(label: str, label_to_model: dict[str, str] | None = None) -> str:
    if label_to_model is None:
        return label
    model_id = label_to_model.get(label)
    if not model_id:
        return label
    return f"{label} · {model_id}"


def _format_parsed_ranking(
    ranking: tuple[str, ...],
    label_to_model: dict[str, str] | None = None,
) -> str:
    if not ranking:
        return "(none parsed)"
    return " > ".join(_label_with_model(label, label_to_model) for label in ranking)


def _format_aggregate_rankings(
    rankings: tuple[CouncilAggregateRanking, ...],
    label_to_model: dict[str, str] | None = None,
) -> str:
    if not rankings:
        return ""
    return "\n".join(
        (
            f"- {_label_with_model(ranking.label, label_to_model)}: "
            f"average rank {ranking.average_rank:.2f} "
            f"across {ranking.rankings_count} review(s)"
        )
        for ranking in rankings
    )


def _fallback_result_text(
    stage1_results: tuple[CouncilCandidateResponse, ...],
    aggregate_rankings: tuple[CouncilAggregateRanking, ...],
) -> tuple[str, str]:
    best_label = aggregate_rankings[0].label if aggregate_rankings else stage1_results[0].label
    best_candidate = next(
        candidate for candidate in stage1_results if candidate.label == best_label
    )
    notice = "_Council judge unavailable; returning the best available council response._"
    return f"{notice}\n\n{best_candidate.text}", "judge_failed"
