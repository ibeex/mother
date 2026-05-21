"""Helpers for README-backed /help prompts."""

from __future__ import annotations

from importlib.resources import files

_HELP_PROMPT = """You are Mother's built-in help assistant.
Use the bundled README below as the authoritative source.
Answer clearly and concisely. If the README does not contain the answer, say so.

{instruction}

--- BEGIN MOTHER README ---
{readme}
--- END MOTHER README ---"""


def read_bundled_readme() -> str:
    """Return the README installed inside the mother package."""
    return files("mother").joinpath("README.md").read_text(encoding="utf-8")


def build_help_prompt(question: str | None = None) -> str:
    """Build the model prompt for a README-backed /help request."""
    readme = read_bundled_readme()
    if question is None or not question.strip():
        instruction = (
            "Summarize what Mother can do for a user. Include the most useful commands, "
            "configuration concepts, and workflow tips from the README."
        )
    else:
        instruction = f"Answer this user question about Mother: {question.strip()}"
    return _HELP_PROMPT.format(instruction=instruction, readme=readme)
