"""Small first-class deep-research workflow for Mother.

This intentionally keeps orchestration simple: plan first, wait for approval, then run
bounded research rounds and a final synthesis. The standard agent path does not use
this module.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from pydantic_ai import Tool
from pydantic_ai.messages import ModelMessage

from mother.models import ModelEntry
from mother.runtime import ChatRuntime, RuntimeResponse, RuntimeToolEvent
from mother.stats import TurnUsage

_APPROVAL_WORDS: frozenset[str] = frozenset(
    {
        "approve",
        "approved",
        "continue",
        "do it",
        "go ahead",
        "ok",
        "okay",
        "proceed",
        "run it",
        "start research",
        "yes",
        "yep",
    }
)
_NEGATED_APPROVAL_PATTERNS: tuple[str, ...] = (
    "do not approve",
    "dont approve",
    "don't approve",
    "not approved",
    "not yet",
)
_RESEARCH_FEEDBACK_HINTS: tuple[str, ...] = (
    "add ",
    "change ",
    "focus ",
    "instead ",
    "narrow ",
    "remove ",
    "revise ",
    "update ",
    "wider ",
)


@dataclass(frozen=True, slots=True)
class PendingDeepResearch:
    """A plan waiting for the user to approve or revise it."""

    question: str
    plan: str


@dataclass(frozen=True, slots=True)
class DeepResearchResult:
    """Final user-visible text and aggregate usage for a deep-research workflow."""

    text: str
    usage: TurnUsage


def is_research_approval(text: str) -> bool:
    """Return whether a reply looks like approval for the pending research plan."""
    normalized = " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())
    if not normalized:
        return False
    if any(pattern in normalized for pattern in _NEGATED_APPROVAL_PATTERNS):
        return False
    if normalized in _APPROVAL_WORDS:
        return True

    approval_detected = any(
        normalized == word
        or normalized.startswith(f"{word} ")
        or normalized.endswith(f" {word}")
        or f" {word} " in normalized
        for word in _APPROVAL_WORDS
    )
    if not approval_detected:
        return False
    return not any(normalized.startswith(prefix) for prefix in _RESEARCH_FEEDBACK_HINTS)


def aggregate_turn_usage(usages: list[TurnUsage]) -> TurnUsage:
    """Merge usage from internal deep-research model calls into one visible turn."""
    if not usages:
        return TurnUsage()

    def add_optional(values: list[int | None]) -> int | None:
        total: int | None = None
        for value in values:
            if value is None:
                continue
            total = value if total is None else total + value
        return total

    last = usages[-1]
    duration_values = [
        usage.duration_seconds for usage in usages if usage.duration_seconds is not None
    ]
    duration = sum(duration_values) if duration_values else None
    return TurnUsage(
        request_tokens=add_optional([usage.request_tokens for usage in usages]),
        response_tokens=add_optional([usage.response_tokens for usage in usages]),
        total_tokens=add_optional([usage.total_tokens for usage in usages]),
        cache_read_tokens=add_optional([usage.cache_read_tokens for usage in usages]),
        cache_write_tokens=add_optional([usage.cache_write_tokens for usage in usages]),
        tool_calls_started=sum(usage.tool_calls_started for usage in usages),
        tool_calls_finished=sum(usage.tool_calls_finished for usage in usages),
        tool_call_errors=sum(usage.tool_call_errors for usage in usages),
        image_count=sum(usage.image_count for usage in usages),
        duration_seconds=duration,
        provider=last.provider,
        model_id=last.model_id,
        response_model_name=last.response_model_name,
    )


class DeepResearchRunner:
    """Orchestrate Mother's deep-research mode outside the normal agent loop."""

    def __init__(
        self,
        model_entry: ModelEntry,
        *,
        base_system_prompt: str,
        ca_bundle_path: str = "",
        model_settings: dict[str, object] | None = None,
        max_rounds: int = 2,
    ) -> None:
        self.model_entry: ModelEntry = model_entry
        self.base_system_prompt: str = base_system_prompt.strip()
        self.ca_bundle_path: str = ca_bundle_path
        self.model_settings: dict[str, object] = dict(model_settings or {})
        self.max_rounds: int = max(1, max_rounds)
        self.runtime: ChatRuntime = ChatRuntime(model_entry, ca_bundle_path=ca_bundle_path)

    def _system(self, role: str) -> str:
        return "\n\n".join(
            [
                self.base_system_prompt,
                "You are running Mother's deep-research workflow.",
                role,
            ]
        )

    async def create_plan(
        self,
        question: str,
        *,
        message_history: list[ModelMessage],
        on_text_update: Callable[[str], None] | None = None,
    ) -> RuntimeResponse:
        """Create a user-facing plan without web access."""
        prompt = "\n".join(
            [
                "Create a concise deep-research plan for this request.",
                "Do not search yet.",
                "Include:",
                "- the question and decision this research should support",
                "- 1 to 5 focused search queries",
                "- source types to prioritize",
                "- comparison criteria, trade-offs, and gaps to verify",
                "End by asking the user to approve or adjust the plan.",
                "",
                f"Request: {question}",
            ]
        )
        return await self.runtime.run_stream(
            prompt_text=prompt,
            system_prompt=self._system("Plan only. Do not use tools."),
            message_history=message_history,
            attachments=[],
            tools=[],
            model_settings=self.model_settings,
            on_text_update=on_text_update,
        )

    async def classify_plan_reply(
        self,
        pending: PendingDeepResearch,
        reply: str,
    ) -> RuntimeResponse:
        """Ask the model whether the user's reply approves or revises the plan."""
        prompt = "\n".join(
            [
                "Classify the user's reply to this pending deep-research plan.",
                "Return exactly one word:",
                "APPROVE - if the user is approving, accepting, saying go ahead, or asking to start.",
                "REVISE - if the user is changing scope, asking questions, adding/removing criteria, or not clearly approving.",
                "When uncertain, return REVISE.",
                "",
                f"Research question: {pending.question}",
                "",
                "Pending plan:",
                pending.plan,
                "",
                f"User reply: {reply}",
            ]
        )
        return await self.runtime.run_stream(
            prompt_text=prompt,
            system_prompt=self._system("Classify plan approval only. Do not use tools."),
            message_history=[],
            attachments=[],
            tools=[],
            model_settings=self.model_settings,
        )

    async def revise_plan(
        self,
        pending: PendingDeepResearch,
        feedback: str,
        *,
        message_history: list[ModelMessage],
        on_text_update: Callable[[str], None] | None = None,
    ) -> RuntimeResponse:
        """Revise a pending plan from user feedback, still without web access."""
        prompt = "\n".join(
            [
                "Revise the pending deep-research plan using the user's feedback.",
                "Do not search yet. End by asking for approval or more adjustments.",
                "",
                f"Original request: {pending.question}",
                "",
                "Current plan:",
                pending.plan,
                "",
                f"User feedback: {feedback}",
            ]
        )
        return await self.runtime.run_stream(
            prompt_text=prompt,
            system_prompt=self._system("Plan revision only. Do not use tools."),
            message_history=message_history,
            attachments=[],
            tools=[],
            model_settings=self.model_settings,
            on_text_update=on_text_update,
        )

    async def run_research(
        self,
        pending: PendingDeepResearch,
        *,
        tools: list[Tool[None]],
        on_text_update: Callable[[str], None] | None = None,
        on_tool_event: Callable[[RuntimeToolEvent], None] | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> DeepResearchResult:
        """Run bounded research rounds followed by a no-tool synthesis."""
        usages: list[TurnUsage] = []
        findings = ""
        for round_number in range(1, self.max_rounds + 1):
            if on_progress is not None:
                on_progress(
                    f"Deep research · round {round_number}/{self.max_rounds} · searching and reading"
                )
            round_prompt = "\n".join(
                [
                    "Execute this deep-research round using web_search and web_fetch.",
                    "Search focused queries from the plan, fetch the best primary/reputable sources,",
                    "extract concrete findings, and compare conflicting claims.",
                    "Fetch about 1 to 3 promising results per useful query. Avoid duplicate URLs.",
                    "At the end, write either STATUS: ENOUGH or STATUS: NEED_MORE with a short reason.",
                    "Do not write the final report yet; write research notes with source URLs.",
                    "",
                    f"Question: {pending.question}",
                    "",
                    "Approved plan:",
                    pending.plan,
                    "",
                    "Previous findings:",
                    findings or "(none yet)",
                ]
            )
            response = await self.runtime.run_stream(
                prompt_text=round_prompt,
                system_prompt=self._system("Research round. Use only the provided web tools."),
                message_history=[],
                attachments=[],
                tools=tools,
                model_settings=self.model_settings,
                tool_call_limit=None,
                on_tool_event=on_tool_event,
            )
            usages.append(response.usage)
            findings = "\n\n".join(part for part in [findings, response.text] if part.strip())
            if "STATUS: ENOUGH" in response.text.upper():
                break

        if on_progress is not None:
            on_progress("Deep research · synthesizing final report")
        synthesis_prompt = "\n".join(
            [
                "Write the final deep-research report from these findings.",
                "Be concise but complete. Include:",
                "- direct answer / recommendation",
                "- key evidence with source links",
                "- trade-offs, disagreements, uncertainty, and gaps",
                "- practical next steps when useful",
                "Do not invent sources beyond the findings.",
                "",
                f"Question: {pending.question}",
                "",
                "Approved plan:",
                pending.plan,
                "",
                "Findings:",
                findings or "No findings were gathered.",
            ]
        )
        synthesis = await self.runtime.run_stream(
            prompt_text=synthesis_prompt,
            system_prompt=self._system("Final synthesis. Do not use tools."),
            message_history=[],
            attachments=[],
            tools=[],
            model_settings=self.model_settings,
            on_text_update=on_text_update,
        )
        usages.append(synthesis.usage)
        return DeepResearchResult(text=synthesis.text, usage=aggregate_turn_usage(usages))
