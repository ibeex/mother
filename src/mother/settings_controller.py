"""Model, agent-mode, and reasoning setting actions for MotherApp."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from mother.agent_modes import AgentProfile, format_agent_profile
from mother.app_session import AppSession
from mother.reasoning import (
    REASONING_EFFORT_HELP,
    format_reasoning_effort,
    normalize_reasoning_effort,
    supported_reasoning_efforts,
    supports_openai_reasoning_summary,
    supports_reasoning_effort,
)


@dataclass(frozen=True, slots=True)
class SettingsControllerCallbacks:
    """Explicit callbacks and state used by settings-related actions."""

    app_session: AppSession
    notify: Callable[..., None]
    update_subtitle: Callable[[], None]
    update_statusline: Callable[[], None]
    conversation_has_history: Callable[[], bool]
    push_model_switch_confirm: Callable[[str, Callable[[bool | None], None]], object]


class SettingsController:
    """Encapsulate model, reasoning, and agent-mode settings mutations."""

    def __init__(self, callbacks: SettingsControllerCallbacks) -> None:
        self.callbacks: SettingsControllerCallbacks = callbacks

    def show_reasoning_status(self) -> None:
        """Notify the user of the current reasoning setting and model support."""
        session = self.callbacks.app_session
        configured = format_reasoning_effort(session.config.reasoning_effort)
        summary_suffix = ""
        if supports_openai_reasoning_summary(session.current_model_entry):
            summary_suffix = f" · summary {session.config.openai_reasoning_summary}"
        supported = supported_reasoning_efforts(session.current_model_entry)
        if supported:
            if (
                session.config.reasoning_effort != "auto"
                and session.config.reasoning_effort not in supported
            ):
                supported_text = "|".join(format_reasoning_effort(value) for value in supported)
                self.callbacks.notify(
                    (
                        f"{session.config.model} reasoning: {configured}{summary_suffix} "
                        f"(not supported here). Supported: {supported_text}"
                    ),
                    title="Reasoning",
                    severity="warning",
                )
                return
            self.callbacks.notify(
                f"{session.config.model} reasoning: {configured}{summary_suffix}",
                title="Reasoning",
            )
            return
        self.callbacks.notify(
            (
                f"{session.config.model} does not expose reasoning control. "
                f"Configured default: {configured}"
            ),
            title="Reasoning",
            severity="warning",
        )

    def set_reasoning_effort(self, effort: str) -> None:
        """Update the configured reasoning effort for future model requests."""
        session = self.callbacks.app_session
        normalized = normalize_reasoning_effort(effort)
        if normalized is None:
            self.callbacks.notify(
                f"Use /reasoning {REASONING_EFFORT_HELP}",
                title="Reasoning",
                severity="warning",
            )
            return

        supported = supported_reasoning_efforts(session.current_model_entry)
        if supported and normalized != "auto" and normalized not in supported:
            supported_text = "|".join(format_reasoning_effort(value) for value in supported)
            self.callbacks.notify(
                f"{session.config.model} supports: {supported_text}",
                title="Reasoning",
                severity="warning",
            )
            return

        previous = session.config.reasoning_effort
        session.config = replace(
            session.config,
            reasoning_effort=normalized,
            tools_enabled=session.agent_mode,
        )
        session.record_session_event(
            "reasoning_effort_change",
            {
                "from": previous,
                "reasoning_effort": normalized,
                "model": session.config.model,
            },
        )
        self.callbacks.update_statusline()
        configured = format_reasoning_effort(normalized)
        if supports_reasoning_effort(session.current_model_entry):
            self.callbacks.notify(
                f"Reasoning set to {configured} for {session.config.model}",
                title="Reasoning",
            )
            return
        self.callbacks.notify(
            (
                f"Reasoning set to {configured}. {session.config.model} does not expose "
                "reasoning control, so this will apply when you switch models."
            ),
            title="Reasoning",
            severity="warning",
        )

    def apply_model_switch(self, model_id: str) -> None:
        """Switch to a different model and refresh conversation-scoped UI state."""
        session = self.callbacks.app_session
        previous_model = session.config.model
        session.switch_model(model_id)
        session.record_session_event(
            "model_change",
            {"from": previous_model, "model": model_id},
        )
        self.callbacks.update_subtitle()
        self.callbacks.update_statusline()
        self.callbacks.notify(f"Switched to {model_id}", title="Model changed")

    def action_switch_model(self, model_id: str) -> None:
        """Switch to a different model, asking first if that will clear context."""
        session = self.callbacks.app_session
        if model_id == session.config.model:
            return

        if not self.callbacks.conversation_has_history():
            self.apply_model_switch(model_id)
            return

        def on_confirm(confirmed: bool | None) -> None:
            if confirmed:
                self.apply_model_switch(model_id)

        _ = self.callbacks.push_model_switch_confirm(model_id, on_confirm)

    def set_agent_mode(
        self,
        *,
        enabled: bool,
        profile: AgentProfile | None = None,
    ) -> tuple[bool, AgentProfile]:
        """Update agent state, refresh UI, and record the transition."""
        session = self.callbacks.app_session
        previous_enabled = session.agent_mode
        previous_profile = session.agent_profile
        if profile is not None:
            session.agent_profile = profile
        session.agent_mode = enabled
        session.record_session_event(
            "agent_mode_change",
            {
                "enabled": session.agent_mode,
                "profile": session.agent_profile,
                "previous_enabled": previous_enabled,
                "previous_profile": previous_profile,
            },
        )
        self.callbacks.update_subtitle()
        self.callbacks.update_statusline()
        return previous_enabled, previous_profile

    def action_set_agent_profile(self, profile: AgentProfile) -> None:
        """Enable agent mode with a specific profile."""
        previous_enabled, previous_profile = self.set_agent_mode(enabled=True, profile=profile)
        if profile == "deep_research":
            message = "Deep research mode enabled"
            if previous_enabled and previous_profile != profile:
                message = "Switched to deep research mode"
        else:
            message = "Agent mode enabled"
            if previous_enabled and previous_profile != profile:
                message = f"Switched to {format_agent_profile(profile)} agent mode"
        self.callbacks.notify(message, title="Agent mode")

    def action_toggle_agent_mode(self) -> None:
        """Toggle standard agent mode on or disable the current agent mode."""
        if self.callbacks.app_session.agent_mode:
            _ = self.set_agent_mode(enabled=False)
            self.callbacks.notify("Agent mode disabled", title="Agent mode")
            return
        self.action_set_agent_profile("standard")
