"""Tests for README- and keybinding-backed help prompts."""

from mother.help import build_help_prompt, read_bundled_keybindings


def test_help_prompt_includes_keybinding_reference() -> None:
    prompt = build_help_prompt("How do I change the autoscroll key?")
    keybindings = read_bundled_keybindings()

    assert "--- BEGIN MOTHER KEYBINDINGS ---" in prompt
    assert "--- END MOTHER KEYBINDINGS ---" in prompt
    assert "toggle_auto_scroll" in prompt
    assert "## Defaults" in prompt
    assert keybindings in prompt


def test_default_help_prompt_mentions_key_bindings() -> None:
    prompt = build_help_prompt()

    assert "key bindings" in prompt.lower()
    assert "toggle_auto_scroll" in prompt


def test_help_question_is_included_verbatim() -> None:
    prompt = build_help_prompt("How do I remap submit?")

    assert "Answer this user question about Mother: How do I remap submit?" in prompt
