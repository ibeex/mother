# pyright: reportPrivateUsage=false
"""Tests for update-check release note formatting."""

from mother.update_check import _format_release_body, _html_to_text


def test_format_release_body_separates_compact_github_notes() -> None:
    body = (
        "feat: Swap default prompt keys to Enter for submit "
        "breaking change Updated config defaults and example files "
        "feat: Implement /help command with concise cheatsheet "
        "Full Changelog: v0.9.4...v0.9.6"
    )

    formatted = _format_release_body(body)

    assert formatted == (
        "- feat: Swap default prompt keys to Enter for submit\n"
        "- breaking change Updated config defaults and example files\n"
        "- feat: Implement /help command with concise cheatsheet\n\n"
        "Full Changelog: v0.9.4...v0.9.6"
    )


def test_html_to_text_keeps_list_items_on_separate_markdown_lines() -> None:
    html = "<ul><li>feat: Add one</li><li>fix: Add two</li></ul>"

    formatted = _html_to_text(html)

    assert formatted == "- feat: Add one\n- fix: Add two"
