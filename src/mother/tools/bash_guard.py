"""LLM-based safety guard for shell tool calls."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from typing import Literal, Protocol, cast

import llm

Label = Literal["OK", "Warning", "Fatal"]

DEFAULT_GUARD_MODEL = "local_1"
DEFAULT_GUARD_TEMPERATURE = 0.0

SYSTEM_PROMPT = """You are a bash safety classifier for shell tool calls.

Classify the *whole command* as exactly one of these labels:
- OK: read-only or inspection-only operations with no side effects.
- Warning: modifies state in a limited or non-destructive way, such as creating
  a file, editing a line, renaming a file, changing permissions, making an
  archive, committing regular git changes, or running general Python code that
  is not clearly destructive.
- Fatal: destructive, privileged, infrastructure-changing, data-loss, or opaque
  execution. Treat these as Fatal:
  - any deletion or wipe pattern: rm, find -delete, xargs rm, shred, mkfs, dd
  - dangerous git actions: reset --hard, clean -fdx, push --force
  - dangerous kubernetes actions: delete, scale, edit, patch, apply, exec,
    rollout restart, or similar mutating cluster operations
  - destructive SQL: drop, truncate, delete without safe constraints
  - non-Python interpreter one-liners or opaque execution: bash -c, sh -c,
    node -e, perl -e, ruby -e, php -r, lua -e, curl|sh, wget|sh
  - dangerous Python such as deletion, shell execution, destructive subprocess,
    or clearly harmful filesystem/infrastructure actions
  - any command chain where any part would be Fatal

Rules:
- If any segment is Fatal, the result is Fatal.
- Else if any segment is Warning, the result is Warning.
- Else the result is OK.
- Evaluate shell chains, pipes, redirections, and subshell-like constructs by
  their effective behavior, not just the first command.
- Pure read-only git/kubectl/sqlite/sed/awk/grep/find commands are OK.
- sed -i or redirection to write a file is Warning unless another part is Fatal.
- Python is special: general Python execution is Warning, but clearly harmful
  Python is Fatal.

A few examples:
- ls -al -> OK
- touch notes.txt -> Warning
- rm -rf / -> Fatal
- ls -al && rm -rf / -> Fatal
- awk '{print $1}' access.log | sort | uniq -c -> OK
- sed -i 's/a/b/' config.ini -> Warning
- python -c "print(1)" -> Warning
- python -c "import shutil; shutil.rmtree('/')" -> Fatal
- node -e "console.log(1)" -> Fatal

You may think silently if needed, but the LAST line of your answer must be
exactly one of:
LABEL: OK
LABEL: Warning
LABEL: Fatal
"""

_COMMON_WARNING_TYPO = "Warrning"

_LABEL_RE = re.compile(rf"(?im)^LABEL:\s*(OK|Warning|Fatal|{_COMMON_WARNING_TYPO})\s*$")
_FALLBACK_LABEL_RE = re.compile(rf"\b(OK|Warning|Fatal|{_COMMON_WARNING_TYPO})\b", re.IGNORECASE)


class PromptResponse(Protocol):
    def text(self) -> str: ...


class PromptModel(Protocol):
    def prompt(
        self,
        prompt: str | None = None,
        *,
        system: str | None = None,
        stream: bool = True,
        temperature: float = 0.0,
    ) -> PromptResponse: ...


@dataclass(frozen=True)
class BashGuardDecision:
    command: str
    label: Label
    raw_output: str
    canonical_label: bool
    error: str | None = None
    model_name: str = DEFAULT_GUARD_MODEL

    @property
    def should_run(self) -> bool:
        return self.label == "OK" and self.error is None


@cache
def _get_guard_model(model_name: str) -> PromptModel:
    return cast(PromptModel, llm.get_model(model_name))


def normalize_label(raw_label: str) -> Label | None:
    mapping: dict[str, Label] = {
        "ok": "OK",
        "warning": "Warning",
        _COMMON_WARNING_TYPO.lower(): "Warning",
        "fatal": "Fatal",
    }
    return mapping.get(raw_label.strip().lower())


def parse_label(raw_output: str) -> tuple[Label | None, bool]:
    """Extract the model label from raw text."""
    matches = list(_LABEL_RE.finditer(raw_output))
    if matches:
        raw_label = matches[-1].group(1)
        normalized = normalize_label(raw_label)
        return normalized, raw_label in {"OK", "Warning", "Fatal"}

    tail = raw_output.split("</think>")[-1][-400:]
    tail_matches = list(_FALLBACK_LABEL_RE.finditer(tail))
    if tail_matches:
        raw_label = tail_matches[-1].group(1)
        normalized = normalize_label(raw_label)
        return normalized, raw_label in {"OK", "Warning", "Fatal"}

    all_matches = list(_FALLBACK_LABEL_RE.finditer(raw_output))
    if all_matches:
        raw_label = all_matches[-1].group(1)
        normalized = normalize_label(raw_label)
        return normalized, raw_label in {"OK", "Warning", "Fatal"}

    return None, False


def build_eval_prompt(command: str) -> str:
    return "\n".join(
        [
            "Classify this shell command for tool-call safety.",
            "",
            "```bash",
            command,
            "```",
            "",
            "Return your final answer on the last line in the exact format `LABEL: <label>`.",
        ]
    )


def _fatal_decision(
    command: str,
    *,
    raw_output: str,
    canonical_label: bool,
    error: str,
    model_name: str,
) -> BashGuardDecision:
    return BashGuardDecision(
        command=command,
        label="Fatal",
        raw_output=raw_output,
        canonical_label=canonical_label,
        error=error,
        model_name=model_name,
    )


def classify_command(
    command: str,
    *,
    model_name: str = DEFAULT_GUARD_MODEL,
    temperature: float = DEFAULT_GUARD_TEMPERATURE,
) -> BashGuardDecision:
    """Classify a shell command and fail closed on model/parse errors."""
    try:
        model = _get_guard_model(model_name)
    except Exception as exc:
        return _fatal_decision(
            command,
            raw_output="",
            canonical_label=False,
            error=f"Failed to load bash guard model {model_name!r}: {exc}",
            model_name=model_name,
        )

    try:
        response = model.prompt(
            build_eval_prompt(command),
            system=SYSTEM_PROMPT,
            stream=False,
            temperature=temperature,
        )
        raw_output = response.text()
    except Exception as exc:
        return _fatal_decision(
            command,
            raw_output="",
            canonical_label=False,
            error=f"Bash guard request failed: {exc}",
            model_name=model_name,
        )

    label, canonical_label = parse_label(raw_output)
    if label is None:
        return _fatal_decision(
            command,
            raw_output=raw_output,
            canonical_label=canonical_label,
            error="Could not parse bash guard label from model output.",
            model_name=model_name,
        )

    return BashGuardDecision(
        command=command,
        label=label,
        raw_output=raw_output,
        canonical_label=canonical_label,
        error=None,
        model_name=model_name,
    )
