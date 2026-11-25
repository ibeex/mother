"""Guided setup helpers for the anonymous /council workflow."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from mother.config import MotherConfig
from mother.models import ModelEntry

DEFAULT_CONFIG_HINT = "~/.config/mother/config.toml"
MAX_SUGGESTED_MEMBERS = 3

# Heuristic ordering for picking a judge and diverse members from the models a
# user already configured. Frontier families first, then reasoning-capable
# models, then config order. This never invents model ids.
_FRONTIER_HINTS: tuple[str, ...] = (
    "opus",
    "gpt-5",
    "gpt-6",
    "o3",
    "o4",
    "sonnet",
    "gemini",
    "grok",
    "deepseek",
    "qwen",
    "kimi",
    "glm",
)


@dataclass(frozen=True, slots=True)
class CouncilSetupSuggestion:
    """A suggested member/judge pair derived from configured model entries."""

    members: tuple[ModelEntry, ...]
    judge: ModelEntry

    @property
    def member_ids(self) -> tuple[str, ...]:
        """Return the suggested member ids in council order."""
        return tuple(entry.id for entry in self.members)


def _family(entry: ModelEntry) -> str:
    """Return a coarse model family key used to encourage provider diversity."""
    token = re.split(r"[^a-z]+", entry.name.strip().lower())[0]
    if token:
        return token
    return entry.api_type


def _quality_rank(entry: ModelEntry) -> tuple[int, int]:
    """Rank models heuristically so strong, reasoning-capable models come first."""
    haystack = f"{entry.id} {entry.name}".lower()
    hint_position = next(
        (index for index, hint in enumerate(_FRONTIER_HINTS) if hint in haystack),
        len(_FRONTIER_HINTS),
    )
    return (hint_position, 0 if entry.supports_reasoning else 1)


def suggest_council_setup(
    models: Sequence[ModelEntry],
    *,
    max_members: int = MAX_SUGGESTED_MEMBERS,
) -> CouncilSetupSuggestion | None:
    """Pick a diverse council and a strong judge from the configured models."""
    if not models or max_members <= 0:
        return None

    ordered = sorted(enumerate(models), key=lambda item: (_quality_rank(item[1]), item[0]))
    selected: list[ModelEntry] = []
    seen_families: set[str] = set()
    for _, entry in ordered:
        family = _family(entry)
        if family in seen_families:
            continue
        seen_families.add(family)
        selected.append(entry)
        if len(selected) >= max_members:
            break

    if len(selected) < max_members:
        selected_ids = {entry.id for entry in selected}
        for _, entry in ordered:
            if entry.id in selected_ids:
                continue
            selected.append(entry)
            selected_ids.add(entry.id)
            if len(selected) >= max_members:
                break

    judge = ordered[0][1]
    return CouncilSetupSuggestion(members=tuple(selected), judge=judge)


def council_config_toml(member_ids: Sequence[str], judge_id: str) -> str:
    """Render a copy-pasteable ``[council]`` TOML table for the given ids."""
    members = ", ".join(json.dumps(member_id) for member_id in member_ids)
    return "\n".join(
        [
            "[council]",
            f"members = [{members}]",
            f"judge = {json.dumps(judge_id)}",
        ]
    )


def _config_location(config: MotherConfig) -> str:
    path = config.config_path.strip()
    return path or DEFAULT_CONFIG_HINT


def _model_line(entry: ModelEntry) -> str:
    details = [entry.api_type]
    if entry.supports_reasoning:
        details.append("reasoning")
    if entry.supports_tools:
        details.append("tools")
    label = entry.id if entry.name == entry.id else f"{entry.id} — {entry.name}"
    return f"- `{label}` ({', '.join(details)})"


def _known_models_block(models: Sequence[ModelEntry]) -> list[str]:
    return ["**Models currently in your registry:**", "", *(_model_line(entry) for entry in models)]


def _build_no_models_guidance(location: str) -> str:
    return "\n".join(
        [
            "### Council setup",
            "",
            "Council needs at least two models in your registry for peer review.",
            "",
            f"No `[[models]]` entries were found in `{location}`. Add two or three models first, "
            + "for example:",
            "",
            "```toml",
            "[[models]]",
            'id = "my-strong-model"',
            'name = "provider-model-name"',
            'api_type = "openai-chat"  # openai-responses | openai-chat | anthropic',
            'base_url = "https://example.com/v1"',
            'api_key_env = "MY_API_KEY"',
            "supports_reasoning = true",
            "```",
            "",
            "Then run `/council <question>` again. Mother will read the model registry, pick a "
            + "diverse set of members plus the strongest judge, and show the exact `[council]` "
            + "block to paste in. Run `/help` for the full model-configuration reference.",
        ]
    )


def build_council_setup_guidance(config: MotherConfig) -> str:
    """Return chat-ready setup guidance for an unconfigured or invalid council."""
    models = tuple(config.models)
    location = _config_location(config)
    if not models:
        return _build_no_models_guidance(location)

    suggestion = suggest_council_setup(models)
    if suggestion is None:
        return _build_no_models_guidance(location)

    known_ids = {entry.id for entry in models}
    configured_members = tuple(dict.fromkeys(config.council.members))
    configured_judge = config.council.judge.strip()
    unknown_members = tuple(member for member in configured_members if member not in known_ids)
    unknown_judge = (
        configured_judge if configured_judge and configured_judge not in known_ids else ""
    )
    known_members = tuple(member for member in configured_members if member in known_ids)

    if unknown_members or unknown_judge:
        heading = "Council needs a configuration fix."
    elif configured_members or configured_judge:
        heading = "Council is not fully configured yet."
    else:
        heading = "Council is not configured yet."

    members = list(known_members)
    if not members:
        members = list(suggestion.member_ids)
    else:
        for member_id in suggestion.member_ids:
            if len(members) >= 2:
                break
            if member_id not in members:
                members.append(member_id)
    judge_id = (
        configured_judge
        if configured_judge and configured_judge in known_ids
        else suggestion.judge.id
    )

    issues: list[str] = []
    if not configured_members:
        issues.append("`council.members` is empty.")
    if not configured_judge:
        issues.append("`council.judge` is empty.")
    if unknown_members:
        issues.append(
            "Unknown `council.members` ids: "
            + ", ".join(f"`{member}`" for member in unknown_members)
        )
    if unknown_judge:
        issues.append(f"Unknown `council.judge` id: `{unknown_judge}`")

    lines = ["### Council setup", "", heading, ""]
    if issues:
        lines.extend([*(f"- {issue}" for issue in issues), ""])
    lines.extend([*_known_models_block(models), ""])
    lines.extend(
        [
            "**Suggested setup** — diverse members with the highest-ranked model as judge:",
            "",
            "```toml",
            council_config_toml(members, judge_id),
            "```",
            "",
            f"Add this table to `{location}` (replace any existing `[council]` table), then run "
            + "`/council <question>` again.",
            "",
            "Notes:",
            "",
            "- `members` and `judge` must use ids from your `[[models]]` entries.",
            "- The judge may also be a member; it only sees anonymous response labels.",
            "- Peer review needs at least two members. With one member the council answers "
            + "directly and skips cross-checking.",
            "- `max_context_turns` (default 8) and `max_context_chars` (default 12000) are "
            + "optional.",
        ]
    )
    if len(models) < 3 or len({_family(entry) for entry in models}) < 2:
        lines.extend(
            [
                "",
                "For a stronger council, add one or two more models from different families or "
                + "providers under `[[models]]`. Mother will include them in the next suggestion.",
            ]
        )
    return "\n".join(lines)
