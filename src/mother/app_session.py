"""Non-UI session state and prompt/runtime helpers for MotherApp."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pydantic_ai import Tool

from mother.agent_modes import (
    DEFAULT_AGENT_PROFILE,
    AgentProfile,
    RuntimeMode,
    format_agent_status,
    resolve_runtime_mode,
)
from mother.bash_execution import BashExecution, format_for_context
from mother.config import MotherConfig
from mother.conversation import ConversationState
from mother.models import ModelEntry, find_model_entry, resolve_model_entry
from mother.prompt_expansion import expand_prompt_fetch_directives
from mother.reasoning import (
    build_reasoning_options,
    format_reasoning_effort,
    supports_openai_reasoning_summary,
    supports_reasoning_effort,
)
from mother.session import SessionManager
from mother.stats import SessionUsage, TurnUsage
from mother.system_prompt import build_system_prompt
from mother.tools import get_default_tools


class CouncilModelResolutionError(ValueError):
    """Raised when council members or judge cannot be resolved from config."""


class AppSession:
    """Own MotherApp's conversation state and non-UI prompt helpers."""

    def __init__(
        self,
        config: MotherConfig,
        *,
        session_manager: SessionManager | None = None,
    ) -> None:
        self.config: MotherConfig = config
        self.current_model_entry: ModelEntry = resolve_model_entry(config.model, config.models)
        self.conversation_state: ConversationState = ConversationState()
        self.session_manager: SessionManager | None = session_manager
        self.agent_mode: bool = config.tools_enabled
        self.agent_profile: AgentProfile = DEFAULT_AGENT_PROFILE
        self.pending_executions: list[BashExecution] = []
        self.pending_image_attachments: dict[str, Path] = {}
        self.session_usage: SessionUsage = SessionUsage()
        self.last_turn_usage: TurnUsage | None = None
        self.last_context_tokens: int | None = None
        self.session_input_tokens: int | None = None
        self.session_output_tokens: int | None = None
        self.session_cached_tokens: int | None = None
        self.last_response_time_seconds: float | None = None

    @property
    def has_history(self) -> bool:
        """Return whether the current conversation already has visible history."""
        return self.conversation_state.has_history

    def switch_model(self, model_id: str) -> None:
        """Update model selection and clear conversation-scoped runtime state."""
        self.config = replace(self.config, model=model_id, tools_enabled=self.agent_mode)
        self.current_model_entry = resolve_model_entry(model_id, self.config.models)
        self.conversation_state = ConversationState()
        self.last_context_tokens = None
        self.last_response_time_seconds = None

    def runtime_mode(self) -> RuntimeMode:
        """Return the effective runtime mode for prompts and tool limits."""
        return resolve_runtime_mode(
            agent_enabled=self.agent_mode,
            agent_profile=self.agent_profile,
        )

    def status_agent_label(self) -> str:
        """Return the compact status-line agent label."""
        return format_agent_status(self.agent_mode, self.agent_profile)

    def reasoning_options(self) -> dict[str, object]:
        """Return supported reasoning settings for the current model."""
        return build_reasoning_options(
            self.current_model_entry,
            self.config.reasoning_effort,
            self.config.openai_reasoning_summary,
        )

    def status_reasoning_effort(self) -> str | None:
        """Return the visible reasoning label for the status line."""
        if not supports_reasoning_effort(self.current_model_entry):
            return None
        label = format_reasoning_effort(self.config.reasoning_effort)
        if supports_openai_reasoning_summary(self.current_model_entry):
            summary = self.config.openai_reasoning_summary
            if summary != "auto":
                return f"{label}/{summary}"
            return label
        if self.current_model_entry.api_type == "anthropic" and label not in {"auto", "off"}:
            return f"{label}/thinking"
        return label

    def apply_turn_usage(self, usage: TurnUsage) -> None:
        """Accumulate normalized turn statistics for status rendering."""
        self.last_turn_usage = usage
        self.session_usage.add_turn(usage)
        self.last_context_tokens = self.session_usage.last_context_tokens
        self.last_response_time_seconds = self.session_usage.last_response_time_seconds
        self.session_input_tokens = self.session_usage.request_tokens
        self.session_output_tokens = self.session_usage.response_tokens
        self.session_cached_tokens = self.session_usage.cache_read_tokens

    def drain_pending_context_text(self) -> str:
        """Return queued shell context and clear the pending list."""
        context_parts: list[str] = []
        for execution in self.pending_executions:
            if not execution.exclude_from_context:
                context_parts.append(format_for_context(execution))
        self.pending_executions.clear()
        return "\n\n".join(context_parts)

    def flush_pending_context(self, value: str) -> str:
        """Prepend queued shell context to a prompt, if any."""
        pending_context = self.drain_pending_context_text()
        if pending_context:
            return pending_context + "\n\n" + value
        return value

    def expand_prompt_fetch_directives(self, prompt: str, user_text: str) -> str:
        """Expand explicit inline fetch directives in the user-visible prompt text."""
        expanded_user_prompt = expand_prompt_fetch_directives(
            user_text,
            ca_bundle_path=self.config.ca_bundle_path,
        ).prompt_text
        if prompt == user_text:
            return expanded_user_prompt
        if prompt.endswith(f"\n\n{user_text}"):
            return f"{prompt[: -len(user_text)]}{expanded_user_prompt}"
        return expanded_user_prompt

    def consume_attachments_for_text(self, text: str) -> list[Path]:
        """Return and clear pending image attachments referenced by prompt text."""
        attachment_paths = [path for path in self.pending_image_attachments if path in text]
        return [self.pending_image_attachments.pop(path) for path in attachment_paths]

    def record_session_message(self, role: str, content: str) -> None:
        """Append a user/assistant message to the persisted session, if enabled."""
        if self.session_manager is None:
            return
        self.session_manager.append(role, content)

    def record_session_event(
        self,
        name: str,
        details: dict[str, object] | None = None,
    ) -> None:
        """Append a structured lifecycle event to the persisted session."""
        if self.session_manager is None:
            return
        self.session_manager.record_event(name, details)

    def record_prompt_context(
        self,
        *,
        user_text: str,
        prompt_text: str,
        system_prompt: str,
        tool_names: list[str],
        attachment_paths: list[str],
    ) -> None:
        """Record the exact prompt context sent to the model for this turn."""
        if self.session_manager is None:
            return
        self.session_manager.record_prompt(
            user_text=user_text,
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            agent_mode=self.agent_mode,
            mode=self.runtime_mode(),
            tool_names=tool_names,
            attachment_paths=attachment_paths,
        )

    def build_council_context(self) -> str:
        """Return a bounded plain-text transcript snapshot for /council."""
        return self.conversation_state.formatted_recent_transcript(
            max_turns=self.config.council.max_context_turns,
            max_chars=self.config.council.max_context_chars,
        )

    def resolve_council_models(self) -> tuple[tuple[ModelEntry, ...], ModelEntry]:
        """Resolve configured council members and judge from the model registry."""
        if not self.config.council.members:
            raise CouncilModelResolutionError(
                "Configure council.members in ~/.config/mother/config.toml first"
            )
        if not self.config.council.judge:
            raise CouncilModelResolutionError(
                "Configure council.judge in ~/.config/mother/config.toml first"
            )

        resolved_members: list[ModelEntry] = []
        missing_members: list[str] = []
        seen_member_ids: set[str] = set()
        for model_id in self.config.council.members:
            if model_id in seen_member_ids:
                continue
            seen_member_ids.add(model_id)
            entry = find_model_entry(model_id, self.config.models)
            if entry is None:
                missing_members.append(model_id)
                continue
            resolved_members.append(entry)

        if missing_members:
            missing_text = ", ".join(missing_members)
            raise CouncilModelResolutionError(
                f"Council members not found in config: {missing_text}"
            )
        if not resolved_members:
            raise CouncilModelResolutionError("Configure at least one valid council member first")

        judge_entry = find_model_entry(self.config.council.judge, self.config.models)
        if judge_entry is None:
            raise CouncilModelResolutionError(
                "Council judge not found in config: " + self.config.council.judge
            )

        return tuple(resolved_members), judge_entry

    def start_new_session(self) -> None:
        """Rotate to a fresh transient session after a successful save."""
        self.session_manager = SessionManager.create(
            markdown_dir=Path(self.config.session_markdown_dir),
            model_name=self.config.model,
        )

    def enabled_tools(self) -> list[Tool[None]]:
        """Return the active tool definitions, if any are enabled and available."""
        tool_registry = get_default_tools(
            tools_enabled=self.agent_mode,
            ca_bundle_path=self.config.ca_bundle_path,
            agent_profile=self.agent_profile,
        )
        if tool_registry.is_empty():
            return []
        return tool_registry.tools()

    @staticmethod
    def tool_names(tools: list[Tool[None]]) -> list[str]:
        """Extract stable prompt-friendly tool names from tool definitions."""
        if not tools:
            return []

        names: list[str] = []
        seen: set[str] = set()
        for tool in tools:
            raw_name = getattr(tool, "name", None)
            if not isinstance(raw_name, str) or not raw_name:
                raw_name = getattr(tool, "__name__", tool.__class__.__name__)
            if raw_name in seen:
                continue
            seen.add(raw_name)
            names.append(raw_name)
        return names

    def build_system_prompt(
        self,
        tools: list[Tool[None]],
        *,
        agent_mode: bool | None = None,
    ) -> str:
        """Build the runtime system prompt for the current model turn."""
        effective_agent_mode = self.agent_mode if agent_mode is None else agent_mode
        effective_mode = self.runtime_mode() if effective_agent_mode else "chat"
        return build_system_prompt(
            self.config.system_prompt,
            mode=effective_mode,
            agent_mode=effective_agent_mode,
            agent_profile=self.agent_profile,
            cwd=Path.cwd(),
            tool_names=self.tool_names(tools),
        )

    def tool_call_limit(self) -> int | None:
        """Return the per-turn tool-call limit for the active runtime mode."""
        if not self.agent_mode:
            return None
        if self.runtime_mode() == "deep_research":
            return None
        return 1
