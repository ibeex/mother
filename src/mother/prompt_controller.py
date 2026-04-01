"""Prompt input history and slash-completion controller for MotherApp."""

from __future__ import annotations

from typing import Protocol

from textual.widgets import OptionList, Static

from mother.agent_modes import AgentProfile, format_agent_profile
from mother.config import MotherConfig
from mother.history import PromptHistory, PromptHistoryMatch
from mother.reasoning import format_reasoning_effort
from mother.slash_commands import (
    SlashCommand,
    current_slash_argument_query,
    current_slash_query,
    get_slash_argument_spec,
)
from mother.user_commands import is_council_multiline_input
from mother.widgets import (
    PromptHistoryComplete,
    PromptTextArea,
    SlashArgumentComplete,
    SlashComplete,
)


class PromptControllerHost(Protocol):
    """Minimal MotherApp surface used by the prompt controller."""

    @property
    def prompt_input(self) -> PromptTextArea: ...

    @property
    def slash_complete(self) -> SlashComplete: ...

    @property
    def slash_argument_complete(self) -> SlashArgumentComplete: ...

    @property
    def prompt_history_help(self) -> Static: ...

    @property
    def prompt_history_complete(self) -> PromptHistoryComplete: ...

    @property
    def prompt_council_help(self) -> Static: ...

    @property
    def agent_mode(self) -> bool: ...

    @property
    def agent_profile(self) -> AgentProfile: ...

    @property
    def config(self) -> MotherConfig: ...

    def notify(self, *args: object, **kwargs: object) -> None: ...


class PromptController:
    """Encapsulate prompt-history, slash completion, and inline prompt help."""

    def __init__(
        self,
        host: PromptControllerHost,
        *,
        prompt_history: PromptHistory,
    ) -> None:
        self.host: PromptControllerHost = host
        self.prompt_history: PromptHistory = prompt_history
        self._suppress_prompt_completion_once: str | None = None
        self._suppress_prompt_history_once: str | None = None
        self._prompt_history_index: int = 0
        self._prompt_history_draft: str = ""
        self._prompt_history_search_query: str | None = None
        self._prompt_history_search_restore_text: str = ""

    @property
    def prompt_input(self) -> PromptTextArea:
        return self.host.prompt_input

    @property
    def slash_complete(self) -> SlashComplete:
        return self.host.slash_complete

    @property
    def slash_argument_complete(self) -> SlashArgumentComplete:
        return self.host.slash_argument_complete

    @property
    def prompt_history_help(self) -> Static:
        return self.host.prompt_history_help

    @property
    def prompt_history_complete(self) -> PromptHistoryComplete:
        return self.host.prompt_history_complete

    @property
    def prompt_council_help(self) -> Static:
        return self.host.prompt_council_help

    def hide_slash_complete(self) -> None:
        """Hide slash-command autocomplete and restore normal prompt keys."""
        self.slash_complete.display = False
        self.prompt_input.slash_complete_active = False

    def hide_slash_argument_complete(self) -> None:
        """Hide inline slash-argument autocomplete and restore normal prompt keys."""
        self.slash_argument_complete.display = False
        self.prompt_input.slash_argument_complete_active = False

    def hide_prompt_history_complete(self) -> None:
        """Hide prompt-history search results and restore normal prompt keys."""
        self.prompt_history_help.display = False
        self.prompt_history_complete.display = False
        self.prompt_input.history_search_active = False

    @staticmethod
    def prompt_text_end_location(text: str) -> tuple[int, int]:
        """Return the cursor location representing the end of text."""
        lines = text.split("\n")
        return (len(lines) - 1, len(lines[-1]))

    def reset_prompt_history_state(self, draft: str | None = None) -> None:
        """Reset history browsing/searching back to the current editable draft."""
        self._prompt_history_index = 0
        self._prompt_history_draft = self.prompt_input.text if draft is None else draft
        self._prompt_history_search_query = None

    def apply_prompt_history_text(self, text: str) -> None:
        """Load a history entry into the prompt without treating it as manual editing."""
        self._suppress_prompt_history_once = text
        self.prompt_input.load_text(text)
        self.prompt_input.move_cursor(self.prompt_text_end_location(text), record_width=False)
        _ = self.prompt_input.focus()

    def refresh_prompt_history_search_matches(self, query: str) -> bool:
        """Refresh the prompt-history popup for the current fuzzy query."""
        matches = self.prompt_history.search(query)
        has_matches = self.prompt_history_complete.update_matches(matches, query)
        self.prompt_history_complete.display = True
        self.prompt_input.history_search_active = True
        return has_matches

    def selected_prompt_history_match(self) -> PromptHistoryMatch | None:
        """Return the highlighted fuzzy prompt-history match, if any."""
        if not self.prompt_history_complete.display:
            return None
        return self.prompt_history_complete.highlighted_match()

    def start_prompt_history_search(self, query: str | None = None) -> None:
        """Open fuzzy prompt-history search using the prompt text as its initial query."""
        initial_query = self.prompt_input.text if query is None else query
        self._prompt_history_search_restore_text = self.prompt_input.text
        self._prompt_history_search_query = initial_query
        self.hide_slash_argument_complete()
        self.hide_slash_complete()
        self.prompt_council_help.display = False
        self.prompt_history_help.display = True
        _ = self.refresh_prompt_history_search_matches(initial_query)
        _ = self.prompt_input.focus()

    def accept_prompt_history_search(
        self,
        match: PromptHistoryMatch | None = None,
    ) -> None:
        """Accept the currently highlighted prompt-history search result."""
        selected_match = match or self.selected_prompt_history_match()
        if selected_match is None:
            return
        self._prompt_history_draft = self._prompt_history_search_restore_text
        self._prompt_history_index = selected_match.index
        self._prompt_history_search_query = None
        self.apply_prompt_history_text(selected_match.text)
        self.hide_prompt_history_complete()

    def dismiss_prompt_history_search(self) -> None:
        """Close prompt-history search and restore the pre-search draft."""
        restore_text = self._prompt_history_search_restore_text
        self._prompt_history_search_query = None
        self.hide_prompt_history_complete()
        self.apply_prompt_history_text(restore_text)
        self.reset_prompt_history_state(restore_text)

    def action_prompt_history_previous(self) -> None:
        """Recall the previous submitted prompt from persistent history."""
        text_area = self.prompt_input
        if text_area.read_only:
            return
        next_index = self._prompt_history_index + 1
        try:
            next_text = self.prompt_history.entry(next_index)
        except IndexError:
            return
        if self._prompt_history_index == 0:
            self._prompt_history_draft = text_area.text
        self._prompt_history_index = next_index
        self._prompt_history_search_query = None
        self.apply_prompt_history_text(next_text)

    def action_prompt_history_next(self) -> None:
        """Move toward newer prompt-history entries or restore the current draft."""
        text_area = self.prompt_input
        if text_area.read_only or self._prompt_history_index == 0:
            return
        next_index = self._prompt_history_index - 1
        self._prompt_history_index = next_index
        self._prompt_history_search_query = None
        if next_index == 0:
            self.apply_prompt_history_text(self._prompt_history_draft)
            return
        self.apply_prompt_history_text(self.prompt_history.entry(next_index))

    def action_prompt_history_search(self) -> None:
        """Open fuzzy prompt-history search for the current input or an empty query."""
        text_area = self.prompt_input
        if text_area.read_only or text_area.history_search_active:
            return
        if self.prompt_history.size == 0:
            self.host.notify("No prompt history yet", title="History")
            return
        self.start_prompt_history_search(text_area.text)

    def current_slash_argument_value(self, command: str) -> str | None:
        """Return the active value to highlight for a slash command argument."""
        normalized_command = command.lower()
        if normalized_command == "/agent":
            if not self.host.agent_mode:
                return None
            return format_agent_profile(self.host.agent_profile)
        if normalized_command == "/models":
            return self.host.config.model
        if normalized_command == "/reasoning":
            return format_reasoning_effort(self.host.config.reasoning_effort)
        return None

    def refresh_prompt_inline_help(self, text: str) -> None:
        """Show contextual helper text for prompt workflows that span multiple lines."""
        self.prompt_council_help.display = is_council_multiline_input(text)

    def refresh_prompt_completions(self, text: str) -> None:
        """Show, filter, or hide prompt helpers for slash commands and arguments."""
        argument_query = current_slash_argument_query(text)
        if argument_query is not None:
            spec = get_slash_argument_spec(argument_query.command)
            if spec is not None:
                self.hide_slash_complete()
                matches = spec.complete(argument_query.query)
                has_matches = self.slash_argument_complete.update_matches(
                    matches,
                    self.current_slash_argument_value(argument_query.command),
                )
                self.slash_argument_complete.display = has_matches
                self.prompt_input.slash_argument_complete_active = has_matches
                return

        self.hide_slash_argument_complete()
        query = current_slash_query(text)
        if query is None or get_slash_argument_spec(query) is not None:
            self.hide_slash_complete()
            return
        has_matches = self.slash_complete.update_query(query)
        self.slash_complete.display = has_matches
        self.prompt_input.slash_complete_active = has_matches

    def selected_slash_command(self) -> SlashCommand | None:
        """Return the highlighted slash command from the popup, if any."""
        return self.slash_complete.highlighted_command()

    def selected_slash_argument_value(self, command: str | None = None) -> str | None:
        """Return the highlighted inline slash-argument value, if any."""
        if not self.slash_argument_complete.display:
            return None
        active_query = current_slash_argument_query(self.prompt_input.text)
        if active_query is None:
            return None
        if command is not None and active_query.command.lower() != command.lower():
            return None
        return self.slash_argument_complete.highlighted_value()

    def apply_slash_completion(self, command: SlashCommand | None = None) -> None:
        """Insert the selected slash command back into the prompt input."""
        selected_command = command or self.selected_slash_command()
        if selected_command is None:
            return

        completed_text = f"{selected_command.command} "
        self.prompt_input.load_text(completed_text)
        self.prompt_input.move_cursor((0, len(completed_text)), record_width=False)
        self.hide_slash_complete()
        self.refresh_prompt_completions(completed_text)
        _ = self.prompt_input.focus()

    def apply_slash_argument_completion(self, value: str | None = None) -> None:
        """Insert the selected inline slash-argument completion into the prompt input."""
        active_query = current_slash_argument_query(self.prompt_input.text)
        if active_query is None:
            return
        selected_value = value or self.selected_slash_argument_value(active_query.command)
        if selected_value is None:
            return

        completed_text = f"{active_query.command} {selected_value}"
        self._suppress_prompt_completion_once = completed_text
        self.prompt_input.load_text(completed_text)
        self.prompt_input.move_cursor((0, len(completed_text)), record_width=False)
        self.hide_slash_argument_complete()
        _ = self.prompt_input.focus()

    def resolve_slash_argument_query(self, command: str, query: str) -> str | None:
        """Resolve a slash-command argument query to a concrete completion value."""
        highlighted_value = self.selected_slash_argument_value(command)
        if highlighted_value is not None:
            return highlighted_value

        spec = get_slash_argument_spec(command)
        if spec is None:
            return None
        return spec.resolve(query)

    def handle_text_changed(self, text: str) -> None:
        """Refresh prompt autocomplete helpers as the main prompt changes."""
        if text == self._suppress_prompt_history_once:
            self._suppress_prompt_history_once = None
            self.hide_slash_argument_complete()
            self.hide_slash_complete()
            self.refresh_prompt_inline_help(text)
            return
        self._suppress_prompt_history_once = None
        if text == self._suppress_prompt_completion_once:
            self._suppress_prompt_completion_once = None
            self.hide_slash_argument_complete()
            self.hide_slash_complete()
            self.hide_prompt_history_complete()
            self.reset_prompt_history_state(text)
            self.refresh_prompt_inline_help(text)
            return
        self._suppress_prompt_completion_once = None
        if self.prompt_input.history_search_active:
            self._prompt_history_search_query = text
            self.hide_slash_argument_complete()
            self.hide_slash_complete()
            self.prompt_council_help.display = False
            _ = self.refresh_prompt_history_search_matches(text)
            return
        self.hide_prompt_history_complete()
        self.reset_prompt_history_state(text)
        self.refresh_prompt_completions(text)
        self.refresh_prompt_inline_help(text)

    def navigate_slash(self, direction: int) -> None:
        """Move the slash-command highlight while the prompt retains focus."""
        if not self.prompt_input.slash_complete_active:
            return
        if direction < 0:
            self.slash_complete.action_cursor_up()
        else:
            self.slash_complete.action_cursor_down()

    def navigate_slash_argument(self, direction: int) -> None:
        """Move the inline slash-argument highlight while the prompt retains focus."""
        if not self.prompt_input.slash_argument_complete_active:
            return
        if direction < 0:
            self.slash_argument_complete.action_cursor_up()
        else:
            self.slash_argument_complete.action_cursor_down()

    def navigate_history_search(self, direction: int) -> None:
        """Move the prompt-history fuzzy-search highlight while the prompt retains focus."""
        if not self.prompt_input.history_search_active:
            return
        if direction < 0:
            self.prompt_history_complete.action_cursor_up()
        else:
            self.prompt_history_complete.action_cursor_down()

    def handle_option_selected(self, event: OptionList.OptionSelected) -> bool:
        """Handle prompt popup selections from slash and argument helpers."""
        if event.option_list is self.slash_complete and event.option.id is not None:
            selected = next(
                (
                    command
                    for command in self.slash_complete.matches
                    if command.command == event.option.id
                ),
                None,
            )
            self.apply_slash_completion(selected)
            return True

        if event.option_list is self.slash_argument_complete and event.option.id is not None:
            self.apply_slash_argument_completion(event.option.id)
            return True

        if event.option_list is self.prompt_history_complete and event.option.id is not None:
            self.accept_prompt_history_search()
            return True

        return False
