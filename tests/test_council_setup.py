"""Tests for guided /council setup suggestions."""

from mother.config import CouncilConfig, MotherConfig
from mother.council_setup import build_council_setup_guidance, suggest_council_setup
from mother.models import ApiType, ModelEntry


def _entry(
    model_id: str,
    *,
    name: str | None = None,
    api_type: ApiType = "openai-responses",
    supports_reasoning: bool = False,
) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        name=name or model_id,
        api_type=api_type,
        supports_reasoning=supports_reasoning,
    )


def test_suggest_council_setup_returns_none_without_models() -> None:
    assert suggest_council_setup([]) is None


def test_suggest_council_setup_prefers_reasoning_and_family_diversity() -> None:
    models = (
        _entry("fast-chat", name="fast-chat"),
        _entry("gpt-5", name="gpt-5", supports_reasoning=True),
        _entry("claude-opus", name="claude-opus", api_type="anthropic", supports_reasoning=True),
    )

    suggestion = suggest_council_setup(models)

    assert suggestion is not None
    assert suggestion.judge.id == "claude-opus"
    assert suggestion.member_ids == ("claude-opus", "gpt-5", "fast-chat")


def test_suggest_council_setup_fills_remaining_slots_after_family_pass() -> None:
    models = (
        _entry("gpt-5", name="gpt-5", supports_reasoning=True),
        _entry("gpt-5-mini", name="gpt-5-mini", supports_reasoning=True),
        _entry("claude-opus", name="claude-opus", api_type="anthropic", supports_reasoning=True),
    )

    suggestion = suggest_council_setup(models)

    assert suggestion is not None
    assert suggestion.member_ids == ("claude-opus", "gpt-5", "gpt-5-mini")


def test_guidance_without_models_explains_how_to_add_them() -> None:
    guidance = build_council_setup_guidance(MotherConfig())

    assert "No `[[models]]` entries were found" in guidance
    assert "[[models]]" in guidance
    assert 'id = "my-strong-model"' in guidance


def test_guidance_suggests_available_models_and_config_block() -> None:
    config = MotherConfig(
        models=[
            _entry("gpt-5", supports_reasoning=True),
            _entry("claude-opus", api_type="anthropic", supports_reasoning=True),
        ]
    )

    guidance = build_council_setup_guidance(config)

    assert "Council is not configured yet." in guidance
    assert "`gpt-5`" in guidance
    assert "`claude-opus`" in guidance
    assert '[council]\nmembers = ["claude-opus", "gpt-5"]\njudge = "claude-opus"' in guidance
    assert "For a stronger council, add one or two more models" in guidance


def test_guidance_keeps_valid_members_when_only_judge_is_missing() -> None:
    config = MotherConfig(
        models=[
            _entry("gpt-5", supports_reasoning=True),
            _entry("claude-opus", api_type="anthropic", supports_reasoning=True),
        ],
        council=CouncilConfig(members=("gpt-5",), judge=""),
    )

    guidance = build_council_setup_guidance(config)

    assert "Council is not fully configured yet." in guidance
    assert "`council.judge` is empty." in guidance
    assert 'members = ["gpt-5", "claude-opus"]' in guidance


def test_guidance_reports_unknown_ids_and_corrects_them() -> None:
    config = MotherConfig(
        models=[
            _entry("gpt-5", supports_reasoning=True),
            _entry("claude-opus", api_type="anthropic", supports_reasoning=True),
        ],
        council=CouncilConfig(members=("missing-model",), judge="also-missing"),
    )

    guidance = build_council_setup_guidance(config)

    assert "Council needs a configuration fix." in guidance
    assert "Unknown `council.members` ids: `missing-model`" in guidance
    assert "Unknown `council.judge` id: `also-missing`" in guidance
    assert '[council]\nmembers = ["claude-opus", "gpt-5"]\njudge = "claude-opus"' in guidance
