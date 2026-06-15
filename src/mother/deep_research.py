"""First-class deep-research workflow for Mother.

Mother keeps deep research integrated with the normal agent/runtime stack, but this
module adds a dedicated research state machine: plan first, wait for approval,
generate explicit per-round queries, gather evidence with bounded web tools,
check whether enough has been learned, and synthesize a final report with
research metadata.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import wraps
from inspect import isawaitable
from typing import cast
from urllib.parse import urlsplit, urlunsplit

from pydantic_ai import Tool
from pydantic_ai.messages import ModelMessage

from mother.models import ModelEntry
from mother.runtime import (
    TUI_STREAM_UPDATE_INTERVAL_SECONDS,
    ChatRuntime,
    RuntimePartialRunError,
    RuntimeResponse,
    RuntimeToolEvent,
)
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
_CATEGORY_WORDS: frozenset[str] = frozenset(
    {"comparison", "product", "how-to", "fact-check", "technical", "market", "general"}
)
_QUERY_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)]|query\s*\d*\s*[:.-])\s*", re.I)
_URL_TRAILING_PUNCTUATION = ").,;:'\"]}>"
_MIN_FINAL_WORDS = 400


@dataclass(frozen=True, slots=True)
class PendingDeepResearch:
    """A plan waiting for the user to approve or revise it."""

    question: str
    plan: str
    category: str = ""


@dataclass(frozen=True, slots=True)
class DeepResearchStats:
    """User/debug visible metadata for a completed research run."""

    rounds: int = 0
    queries: int = 0
    urls: int = 0
    searches: int = 0
    fetches: int = 0
    tool_errors: int = 0
    category: str = ""
    partial: bool = False

    def to_event_details(self) -> dict[str, object]:
        details: dict[str, object] = {
            "rounds": self.rounds,
            "queries": self.queries,
            "urls": self.urls,
            "searches": self.searches,
            "fetches": self.fetches,
            "tool_errors": self.tool_errors,
            "partial": self.partial,
        }
        if self.category:
            details["category"] = self.category
        return details

    def format_markdown(self) -> str:
        lines = [
            "---",
            "",
            "### Research stats",
            f"- Rounds: {self.rounds}",
            f"- Queries: {self.queries}",
            f"- Unique URLs fetched: {self.urls}",
            f"- Tool calls: {self.searches} searches, {self.fetches} fetches",
        ]
        if self.tool_errors:
            lines.append(f"- Tool errors: {self.tool_errors}")
        if self.category:
            lines.append(f"- Category: {self.category}")
        if self.partial:
            lines.append("- Status: partial report after recoverable research failures")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class DeepResearchResult:
    """Final user-visible text, aggregate usage, and metadata."""

    text: str
    usage: TurnUsage
    stats: DeepResearchStats = field(default_factory=DeepResearchStats)


@dataclass(slots=True)
class ResearchRound:
    """Structured state for one research round."""

    number: int
    queries: list[str]
    notes: str = ""
    searches_started: int = 0
    fetches_started: int = 0
    tool_errors: int = 0
    fetched_urls: set[str] = field(default_factory=set)


@dataclass(slots=True)
class ResearchState:
    """Accumulated structured state across a deep-research run."""

    question: str
    plan: str
    category: str = ""
    prior_report: str = ""
    rounds: list[ResearchRound] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    queries_used: set[str] = field(default_factory=set)
    urls_fetched: set[str] = field(default_factory=set)
    failures: list[str] = field(default_factory=list)

    def findings_text(self) -> str:
        parts: list[str] = []
        if self.prior_report.strip():
            parts.append("Prior report / continuation context:\n" + self.prior_report.strip())
        parts.extend(self.findings)
        return "\n\n".join(part for part in parts if part.strip())

    def failure_notices_markdown(self) -> str:
        if not self.failures:
            return ""
        lines = ["### Research notices"]
        lines.extend(f"- {failure}" for failure in self.failures)
        return "\n".join(lines)

    def stats(self) -> DeepResearchStats:
        return DeepResearchStats(
            rounds=len(self.rounds),
            queries=len(self.queries_used),
            urls=len(self.urls_fetched),
            searches=sum(round_state.searches_started for round_state in self.rounds),
            fetches=sum(round_state.fetches_started for round_state in self.rounds),
            tool_errors=sum(round_state.tool_errors for round_state in self.rounds),
            category=self.category,
            partial=bool(self.failures),
        )


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


def parse_research_queries(text: str, *, max_queries: int = 5) -> list[str]:
    """Parse model-produced query lists into a small de-duplicated list."""
    queries: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("` ")
        if not line or line.casefold().startswith(("queries", "search queries")):
            continue
        line = _QUERY_PREFIX_RE.sub("", line).strip()
        line = line.strip('"')
        if not line or len(line.split()) < 2:
            continue
        key = normalize_query(line)
        if key in seen:
            continue
        seen.add(key)
        queries.append(line)
        if len(queries) >= max_queries:
            break
    return queries


def normalize_query(query: str) -> str:
    """Normalize a search query for de-duplication."""
    return " ".join(query.casefold().split())


def normalize_url(url: str) -> str:
    """Normalize a URL enough for duplicate-fetch prevention and stats."""
    stripped = url.strip().rstrip(_URL_TRAILING_PUNCTUATION)
    if not stripped:
        return ""
    parts = urlsplit(stripped)
    if not parts.scheme or not parts.netloc:
        return stripped.casefold()
    path = parts.path.rstrip("/") or "/"
    netloc = parts.netloc.casefold()
    scheme = parts.scheme.casefold()
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def _extract_urls(text: str) -> set[str]:
    matches: list[str] = re.findall(r"https?://\S+", text)
    urls: set[str] = set()
    for match in matches:
        normalized = normalize_url(match)
        if normalized:
            urls.add(normalized)
    return urls


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


async def _maybe_await(value: object) -> object:
    if isawaitable(value):
        return await cast(Awaitable[object], value)
    return value


class DeepResearchRunner:
    """Orchestrate Mother's deep-research mode outside the normal agent loop."""

    def __init__(
        self,
        model_entry: ModelEntry,
        *,
        base_system_prompt: str,
        ca_bundle_path: str = "",
        model_settings: dict[str, object] | None = None,
        min_rounds: int = 1,
        max_rounds: int = 4,
        max_queries_per_round: int = 5,
        fetches_per_round: int = 10,
        max_empty_rounds: int = 1,
        max_tool_calls_per_round: int = 40,
    ) -> None:
        self.model_entry: ModelEntry = model_entry
        self.base_system_prompt: str = base_system_prompt.strip()
        self.ca_bundle_path: str = ca_bundle_path
        self.model_settings: dict[str, object] = dict(model_settings or {})
        self.min_rounds: int = max(1, min_rounds)
        self.max_rounds: int = max(self.min_rounds, max_rounds)
        self.max_queries_per_round: int = max(1, max_queries_per_round)
        self.fetches_per_round: int = max(1, fetches_per_round)
        self.max_empty_rounds: int = max(0, max_empty_rounds)
        self.max_tool_calls_per_round: int = max(
            self.fetches_per_round + self.max_queries_per_round,
            max_tool_calls_per_round,
        )
        self.runtime: ChatRuntime = ChatRuntime(
            model_entry,
            ca_bundle_path=ca_bundle_path,
            stream_update_interval_seconds=TUI_STREAM_UPDATE_INTERVAL_SECONDS,
        )

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
                "- 3 to 5 focused seed search queries",
                "- source types to prioritize",
                "- comparison criteria, trade-offs, and gaps to verify",
                "- the likely report style: comparison, product, how-to, fact-check, technical, market, or general",
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
                "Keep 3 to 5 seed search queries and a likely report style/category.",
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

    async def classify_category(self, pending: PendingDeepResearch) -> RuntimeResponse:
        """Classify the requested report style for category-aware synthesis."""
        prompt = "\n".join(
            [
                "Classify this deep-research request into exactly one lowercase word from:",
                "comparison, product, how-to, fact-check, technical, market, general",
                "Return only that word.",
                "",
                f"Question: {pending.question}",
                "",
                "Approved plan:",
                pending.plan,
            ]
        )
        return await self.runtime.run_stream(
            prompt_text=prompt,
            system_prompt=self._system("Classify research category only. Do not use tools."),
            message_history=[],
            attachments=[],
            tools=[],
            model_settings=self.model_settings,
        )

    async def generate_queries(
        self,
        state: ResearchState,
        round_number: int,
    ) -> RuntimeResponse:
        """Generate explicit per-round search queries before the web/tool round."""
        previous_queries = "\n".join(f"- {query}" for query in sorted(state.queries_used))
        previous_urls = "\n".join(f"- {url}" for url in sorted(state.urls_fetched))
        round_instruction = (
            "This is the first round: generate broad, diverse queries that cover the main angles."
            if round_number == 1
            else "Generate targeted follow-up queries to fill gaps, resolve conflicts, and avoid repeating prior searches."
        )
        prompt = "\n".join(
            [
                "Generate search queries for the next deep-research round.",
                round_instruction,
                f"Return {min(self.max_queries_per_round, 5)} queries as plain bullet lines only.",
                "Do not include explanations. Do not use tools.",
                "",
                f"Question: {state.question}",
                f"Category: {state.category or 'general'}",
                "",
                "Approved plan:",
                state.plan,
                "",
                "Prior findings:",
                state.findings_text() or "(none yet)",
                "",
                "Previous queries to avoid:",
                previous_queries or "(none yet)",
                "",
                "Fetched URLs to avoid:",
                previous_urls or "(none yet)",
            ]
        )
        return await self.runtime.run_stream(
            prompt_text=prompt,
            system_prompt=self._system("Query generation only. Do not use tools."),
            message_history=[],
            attachments=[],
            tools=[],
            model_settings=self.model_settings,
        )

    async def should_stop(self, state: ResearchState, round_number: int) -> RuntimeResponse:
        """Dedicated no-tool coverage check after each research round."""
        prompt = "\n".join(
            [
                "Decide whether this deep research has enough evidence for a useful final report.",
                "Return exactly one line starting with YES or NO, followed by a short reason.",
                "Say YES only if the findings answer the question, cite enough reputable sources, cover trade-offs, and identify uncertainties.",
                "Say NO if important gaps remain or the findings are too thin.",
                "",
                f"Question: {state.question}",
                f"Round: {round_number}/{self.max_rounds}",
                f"Category: {state.category or 'general'}",
                "",
                "Approved plan:",
                state.plan,
                "",
                "Findings:",
                state.findings_text() or "No findings were gathered.",
            ]
        )
        return await self.runtime.run_stream(
            prompt_text=prompt,
            system_prompt=self._system("Research stopping decision only. Do not use tools."),
            message_history=[],
            attachments=[],
            tools=[],
            model_settings=self.model_settings,
        )

    def _research_tools(
        self,
        tools: list[Tool[None]],
        state: ResearchState,
        round_state: ResearchRound,
    ) -> list[Tool[None]]:
        guarded: list[Tool[None]] = []
        for tool in tools:
            if tool.name == "web_search":
                guarded.append(self._guarded_search_tool(tool, round_state))
            elif tool.name == "web_fetch":
                guarded.append(self._guarded_fetch_tool(tool, state, round_state))
            else:
                guarded.append(tool)
        return guarded

    def _guarded_search_tool(
        self,
        tool: Tool[None],
        round_state: ResearchRound,
    ) -> Tool[None]:
        original = cast(Callable[..., object], tool.function)

        @wraps(original)
        async def web_search(query: str, timeout: float = 20.0) -> object:
            """Search the public web by query.

            Args:
                query: Search query string describing what to look for.
                timeout: Network timeout in seconds.
            """
            round_state.searches_started += 1
            return await _maybe_await(original(query, timeout=timeout))

        return Tool(
            web_search,
            name=tool.name,
            description=tool.description,
            max_retries=tool.max_retries,
            docstring_format=tool.docstring_format,
            require_parameter_descriptions=False,
            strict=tool.strict,
            sequential=tool.sequential,
            requires_approval=tool.requires_approval,
            metadata=tool.metadata,
            timeout=tool.timeout,
        )

    def _guarded_fetch_tool(
        self,
        tool: Tool[None],
        state: ResearchState,
        round_state: ResearchRound,
    ) -> Tool[None]:
        original = cast(Callable[..., object], tool.function)

        @wraps(original)
        async def web_fetch(
            url: str,
            mode: str = "auto",
            method: str = "GET",
            headers_json: str = "",
            body: str = "",
            timeout: float = 20.0,
        ) -> object:
            """Fetch a web page or HTTP endpoint.

            Args:
                url: Required HTTP or HTTPS URL.
                mode: One of "auto", "raw", or "jina".
                method: HTTP method for raw requests.
                headers_json: Optional JSON object string of request headers.
                body: Optional request body string for raw requests.
                timeout: Network timeout in seconds.
            """
            normalized = normalize_url(url)
            if normalized in state.urls_fetched:
                return f"Skipped duplicate fetch: {normalized} was already fetched in this research run."
            if round_state.fetches_started >= self.fetches_per_round:
                return f"Skipped fetch: per-round fetch budget of {self.fetches_per_round} URLs is exhausted."
            state.urls_fetched.add(normalized)
            round_state.fetched_urls.add(normalized)
            round_state.fetches_started += 1
            result = await _maybe_await(
                original(
                    url,
                    mode=mode,
                    method=method,
                    headers_json=headers_json,
                    body=body,
                    timeout=timeout,
                )
            )
            return result

        return Tool(
            web_fetch,
            name=tool.name,
            description=tool.description,
            max_retries=tool.max_retries,
            docstring_format=tool.docstring_format,
            require_parameter_descriptions=False,
            strict=tool.strict,
            sequential=tool.sequential,
            requires_approval=tool.requires_approval,
            metadata=tool.metadata,
            timeout=tool.timeout,
        )

    def round_tool_call_limit(self, query_count: int) -> int:
        """Return a generous hard cap while fetch/search budgets do softer control."""
        dynamic_limit = self.fetches_per_round + (query_count * 3) + 10
        return max(dynamic_limit, self.max_tool_calls_per_round)

    def _round_tool_event_handler(
        self,
        round_state: ResearchRound,
        on_tool_event: Callable[[RuntimeToolEvent], None] | None,
    ) -> Callable[[RuntimeToolEvent], None]:
        def handle(event: RuntimeToolEvent) -> None:
            if event.phase == "finished" and event.is_error:
                round_state.tool_errors += 1
            if on_tool_event is not None:
                on_tool_event(event)

        return handle

    async def _prepare_state(
        self, pending: PendingDeepResearch, prior_report: str
    ) -> tuple[ResearchState, list[TurnUsage]]:
        usages: list[TurnUsage] = []
        category = pending.category.strip().casefold()
        if category not in _CATEGORY_WORDS:
            category_response = await self.classify_category(pending)
            usages.append(category_response.usage)
            candidate = category_response.text.strip().casefold().split(maxsplit=1)[0]
            category = candidate if candidate in _CATEGORY_WORDS else "general"
        state = ResearchState(
            question=pending.question,
            plan=pending.plan,
            category=category,
            prior_report=prior_report,
        )
        return state, usages

    async def _run_round(
        self,
        state: ResearchState,
        round_number: int,
        tools: list[Tool[None]],
        on_tool_event: Callable[[RuntimeToolEvent], None] | None,
        on_progress: Callable[[str], None] | None,
    ) -> tuple[ResearchRound, RuntimeResponse]:
        query_response = await self.generate_queries(state, round_number)
        parsed_queries = parse_research_queries(
            query_response.text,
            max_queries=self.max_queries_per_round,
        )
        queries = [
            query for query in parsed_queries if normalize_query(query) not in state.queries_used
        ]
        if not queries:
            queries = [f"{state.question} latest evidence sources"]
        for query in queries:
            state.queries_used.add(normalize_query(query))

        if on_progress is not None:
            on_progress(
                f"Deep research · round {round_number}/{self.max_rounds} · searching {len(queries)} queries"
            )

        round_state = ResearchRound(number=round_number, queries=queries)
        state.rounds.append(round_state)
        round_prompt = "\n".join(
            [
                "Execute this deep-research round using only web_search and web_fetch.",
                "Use the exact search queries listed below before inventing any additional query.",
                "Fetch the best primary/reputable sources, extract concrete findings, and compare conflicting claims.",
                f"Fetch no more than {self.fetches_per_round} unique URLs this round; duplicate fetches will be skipped.",
                "Do not write the final report yet. Write structured research notes with source URLs.",
                "Include a short 'Gaps remaining' section at the end.",
                "",
                f"Question: {state.question}",
                f"Category: {state.category or 'general'}",
                "",
                "Approved plan:",
                state.plan,
                "",
                "Search queries for this round:",
                "\n".join(f"- {query}" for query in queries),
                "",
                "Already fetched URLs to avoid:",
                "\n".join(f"- {url}" for url in sorted(state.urls_fetched)) or "(none yet)",
                "",
                "Previous findings:",
                state.findings_text() or "(none yet)",
            ]
        )
        guarded_tools = self._research_tools(
            tools,
            state,
            round_state,
        )
        recovered_notice = ""
        try:
            response = await self.runtime.run_stream(
                prompt_text=round_prompt,
                system_prompt=self._system("Research round. Use only the provided web tools."),
                message_history=[],
                attachments=[],
                tools=guarded_tools,
                model_settings=self.model_settings,
                tool_call_limit=self.round_tool_call_limit(len(queries)),
                on_tool_event=self._round_tool_event_handler(round_state, on_tool_event),
            )
        except RuntimePartialRunError as exc:
            recovered_notice = (
                f"Round {round_number} reached the per-round tool-call limit; "
                "Mother summarized the completed tool results instead of aborting the research."
            )
            state.failures.append(recovered_notice)
            if on_progress is not None:
                on_progress("Deep research · tool limit reached; summarizing completed sources")
            recovery_prompt = "\n".join(
                [
                    "The previous deep-research round was stopped because it reached the per-round tool-call limit.",
                    "Do not call tools. Use only the completed tool results already present in the conversation.",
                    "Write structured research notes from the sources/results that were already gathered.",
                    "Start with a brief notice that this round hit the tool-call limit and that these are partial round notes.",
                    "Include source URLs that are available, concrete findings, conflicts/uncertainties, and gaps remaining.",
                    "",
                    f"Question: {state.question}",
                    f"Category: {state.category or 'general'}",
                    "",
                    "Approved plan:",
                    state.plan,
                    "",
                    "Search queries for this round:",
                    "\n".join(f"- {query}" for query in queries),
                ]
            )
            response = await self.runtime.run_stream(
                prompt_text=recovery_prompt,
                system_prompt=self._system("Tool-limit recovery summary. Do not use tools."),
                message_history=exc.partial_messages,
                attachments=[],
                tools=[],
                model_settings=self.model_settings,
            )
        notes = response.text
        if recovered_notice:
            notes = f"Notice: {recovered_notice}\n\n{notes}"
        round_state.notes = notes
        state.findings.append(f"## Round {round_number} findings\n\n{notes}")
        state.urls_fetched.update(_extract_urls(response.text))
        return round_state, response

    async def _synthesize(
        self,
        state: ResearchState,
        *,
        on_text_update: Callable[[str], None] | None,
    ) -> RuntimeResponse:
        category_guidance = {
            "comparison": "Use a comparison format with criteria, trade-offs, and a recommendation.",
            "product": "Use a buying/adoption format with requirements fit, alternatives, risks, and recommendation.",
            "how-to": "Use a practical how-to format with steps, prerequisites, pitfalls, and validation checks.",
            "fact-check": "Use a fact-check format with claim, verdict, evidence, caveats, and confidence.",
            "technical": "Use a technical brief format with architecture/details, constraints, evidence, and implementation implications.",
            "market": "Use a market brief format with landscape, trends, actors, evidence, risks, and outlook.",
            "general": "Use a clear analytical report format.",
        }.get(state.category, "Use a clear analytical report format.")
        failure_note = "\n".join(f"- {failure}" for failure in state.failures)
        synthesis_prompt = "\n".join(
            [
                "Write the final deep-research report from these findings.",
                category_guidance,
                "Be complete and evidence-based. Include:",
                "- direct answer / recommendation",
                "- key evidence with source links",
                "- trade-offs, disagreements, uncertainty, and gaps",
                "- practical next steps when useful",
                "Do not invent sources beyond the findings.",
                "If research was partial, say so clearly and explain what still needs verification.",
                "",
                f"Question: {state.question}",
                f"Category: {state.category or 'general'}",
                "",
                "Approved plan:",
                state.plan,
                "",
                "Recoverable failures:",
                failure_note or "(none)",
                "",
                "Findings:",
                state.findings_text() or "No findings were gathered.",
            ]
        )
        return await self.runtime.run_stream(
            prompt_text=synthesis_prompt,
            system_prompt=self._system("Final synthesis. Do not use tools."),
            message_history=[],
            attachments=[],
            tools=[],
            model_settings=self.model_settings,
            on_text_update=on_text_update,
        )

    async def _expand_final_report(
        self, state: ResearchState, draft: str
    ) -> RuntimeResponse | None:
        if _word_count(draft) >= _MIN_FINAL_WORDS:
            return None
        prompt = "\n".join(
            [
                "The deep-research report below is too brief. Expand it significantly while staying faithful to the findings.",
                "Add missing evidence, caveats, comparisons, and next steps. Do not invent sources.",
                "",
                f"Question: {state.question}",
                f"Category: {state.category or 'general'}",
                "",
                "Findings:",
                state.findings_text() or "No findings were gathered.",
                "",
                "Draft report:",
                draft,
            ]
        )
        return await self.runtime.run_stream(
            prompt_text=prompt,
            system_prompt=self._system("Final report expansion. Do not use tools."),
            message_history=[],
            attachments=[],
            tools=[],
            model_settings=self.model_settings,
        )

    async def run_research(
        self,
        pending: PendingDeepResearch,
        *,
        tools: list[Tool[None]],
        on_text_update: Callable[[str], None] | None = None,
        on_tool_event: Callable[[RuntimeToolEvent], None] | None = None,
        on_progress: Callable[[str], None] | None = None,
        prior_report: str = "",
    ) -> DeepResearchResult:
        """Run bounded research rounds followed by no-tool synthesis."""
        usages: list[TurnUsage] = []
        state, setup_usages = await self._prepare_state(pending, prior_report)
        usages.extend(setup_usages)
        empty_rounds = 0

        for round_number in range(1, self.max_rounds + 1):
            try:
                round_state, response = await self._run_round(
                    state,
                    round_number,
                    tools,
                    on_tool_event,
                    on_progress,
                )
                usages.append(response.usage)
                if not round_state.fetched_urls and not _extract_urls(response.text):
                    empty_rounds += 1
                else:
                    empty_rounds = 0
            except Exception as exc:
                state.failures.append(f"Round {round_number} failed: {exc}")
                if not state.findings:
                    raise
                empty_rounds += 1

            if round_number < self.min_rounds:
                continue
            if empty_rounds > self.max_empty_rounds:
                state.failures.append(
                    "Stopped early because consecutive rounds produced no new fetched sources."
                )
                break
            stop_response = await self.should_stop(state, round_number)
            usages.append(stop_response.usage)
            if stop_response.text.strip().upper().startswith("YES"):
                break

        if on_progress is not None:
            on_progress("Deep research · synthesizing final report")
        synthesis = await self._synthesize(state, on_text_update=on_text_update)
        usages.append(synthesis.usage)
        final_text = synthesis.text

        expansion = await self._expand_final_report(state, final_text)
        if expansion is not None:
            usages.append(expansion.usage)
            if _word_count(expansion.text) > _word_count(final_text):
                final_text = expansion.text

        stats = state.stats()
        final_parts = [final_text.strip()]
        notices = state.failure_notices_markdown()
        if notices:
            final_parts.append(notices)
        final_parts.append(stats.format_markdown())
        final_text = "\n\n".join(final_parts).strip()
        return DeepResearchResult(
            text=final_text,
            usage=aggregate_turn_usage(usages),
            stats=stats,
        )
