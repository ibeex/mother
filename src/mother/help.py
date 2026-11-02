"""Helpers for README-backed /help prompts."""

from __future__ import annotations

from importlib.resources import files

_HELP_PROMPT = """You are Mother's built-in help assistant.
Use the bundled README and keybinding reference below as the authoritative sources.
Answer clearly and concisely. If they do not contain the answer, say so.

{instruction}

--- BEGIN MOTHER README ---
{readme}
--- END MOTHER README ---

--- BEGIN MOTHER KEYBINDINGS ---
{keybindings}
--- END MOTHER KEYBINDINGS ---"""


def read_bundled_readme() -> str:
    """Return the README installed inside the mother package."""
    return files("mother").joinpath("README.md").read_text(encoding="utf-8")


def read_bundled_keybindings() -> str:
    """Return the keybinding reference installed inside the mother package."""
    return files("mother").joinpath("KEYBINDINGS.md").read_text(encoding="utf-8")


def build_help_prompt(question: str | None = None) -> str:
    """Build the model prompt for a README- and keybinding-backed /help request."""
    readme = read_bundled_readme()
    keybindings = read_bundled_keybindings()
    if question is None or not question.strip():
        instruction = (
            "Return a short quick-start cheatsheet for Mother. Keep it compact: "
            "one brief sentence, then at most 8 bullets covering the most useful slash "
            "commands, key bindings, and everyday workflows. End by telling the user "
            "they can now ask plain follow-up questions for details. Do not include "
            "long explanations."
        )
    else:
        instruction = f"Answer this user question about Mother: {question.strip()}"
    return _HELP_PROMPT.format(
        instruction=instruction,
        readme=readme,
        keybindings=keybindings,
    )
