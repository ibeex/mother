"""Manual evaluation helper for single-turn agent tool-usage discipline.

This file lives in ``tests/`` for convenience, but it is intentionally *not*
a pytest test. It has no ``test_*`` functions and only runs when invoked
explicitly.

Run it from the Mother repository root, just like the existing dev helpers.
It reuses Mother's real config base system prompt plus the normal runtime
system-prompt builder.

The target scenario is:
- user asks: ``hi, tell me about current project``
- the model should use exactly one tool call
- that tool should be ``bash``
- the bash command can be any reasonable project-inspection command
- after the tool result, the model should summarize what it found and suggest a
  next action instead of continuing with another tool call

The eval uses the real guarded bash tool, so harmful commands are still blocked
by the bash guard rather than by an eval-only restriction.

Usage:
    uv run python tests/tool_usage_eval.py
    uv run python tests/tool_usage_eval.py --model local_1
    uv run python tests/tool_usage_eval.py --model gpt-5-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Tool

from mother.config import load_config
from mother.models import get_all_entries, resolve_model_entry
from mother.runtime import ChatRuntime, RuntimePartialRunError, RuntimeToolEvent
from mother.system_prompt import build_system_prompt
from mother.tools import get_default_tools
from mother.tools.bash_tool import make_bash_tool

DEFAULT_EVAL_MODEL = "local_1"
CASE_NAME = "tell_me_about_current_project"
CASE_PROMPT = "hi, tell me about current project"

_PROJECT_HINTS = (
    "readme",
    "src",
    "tests",
    "pyproject.toml",
    "justfile",
    "agent_docs",
    "devtools",
    "installation.md",
    "development.md",
)
_NEXT_STEP_PHRASES = (
    "want me to",
    "would you like",
    "if you want",
    "i can",
    "we can",
    "should i",
    "next step",
    "next,",
    "next:",
    "next?",
)
_NEXT_STEP_VERBS = (
    "inspect",
    "open",
    "read",
    "check",
    "dig",
    "explore",
    "scan",
    "look",
    "review",
)


@dataclass(frozen=True)
class EvalCase:
    name: str
    prompt: str


@dataclass(frozen=True)
class EvalArgs:
    model: str


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str | None = None


@dataclass
class EvalResult:
    case: EvalCase
    model: str
    final_text: str
    error: str | None
    tool_events: list[RuntimeToolEvent]
    tool_limit_recovery_used: bool = False

    @property
    def started_events(self) -> list[RuntimeToolEvent]:
        return [event for event in self.tool_events if event.phase == "started"]

    @property
    def finished_events(self) -> list[RuntimeToolEvent]:
        return [event for event in self.tool_events if event.phase == "finished"]

    @property
    def first_command(self) -> str | None:
        if not self.started_events:
            return None
        command = self.started_events[0].arguments.get("command")
        if not isinstance(command, str):
            return None
        return command

    def checks(self) -> list[CheckResult]:
        command = self.first_command
        first_tool_name = self.started_events[0].tool_name if self.started_events else None
        mentions_project_shape = _mentions_project_shape(self.final_text)
        suggests_next_action = _suggests_next_action(self.final_text)
        completed_tool_call = len(self.started_events) == 1 and len(self.finished_events) == 1

        return [
            CheckResult(
                name="no runtime error",
                passed=self.error is None,
                detail=self.error,
            ),
            CheckResult(
                name=(
                    "runtime tool-limit recovery used"
                    if self.tool_limit_recovery_used
                    else "no runtime tool-limit recovery used"
                ),
                passed=not self.tool_limit_recovery_used,
                detail=None,
            ),
            CheckResult(
                name="exactly one completed tool call",
                passed=completed_tool_call,
                detail=(
                    f"started={len(self.started_events)}, finished={len(self.finished_events)}"
                    if not completed_tool_call
                    else None
                ),
            ),
            CheckResult(
                name="tool is bash",
                passed=first_tool_name == "bash",
                detail=first_tool_name,
            ),
            CheckResult(
                name="bash command is present",
                passed=command is not None and bool(command.strip()),
                detail=command,
            ),
            CheckResult(
                name="final answer mentions project files",
                passed=mentions_project_shape,
                detail=None if mentions_project_shape else self.final_text or None,
            ),
            CheckResult(
                name="final answer suggests a next action",
                passed=suggests_next_action,
                detail=None if suggests_next_action else self.final_text or None,
            ),
        ]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks())


def _require_repo_root() -> Path:
    cwd = Path.cwd()
    if (
        not (cwd / "pyproject.toml").exists()
        or not (cwd / "src" / "mother").exists()
        or not (cwd / "tests").exists()
    ):
        raise RuntimeError(
            "Run this eval from the Mother repository root (the dev dir), for example: "
            "`cd /path/to/mother && uv run python tests/tool_usage_eval.py --model local_1`."
        )
    return cwd


def build_cases() -> list[EvalCase]:
    return [EvalCase(name=CASE_NAME, prompt=CASE_PROMPT)]


def _tool_names_from_registry(cwd: Path) -> list[str]:
    registry = get_default_tools(tools_enabled=True, cwd=cwd, agent_profile="standard")
    names: list[str] = []
    seen: set[str] = set()
    for tool in registry.tools():
        tool_name = tool.name or getattr(tool.function, "__name__", tool.__class__.__name__)
        if tool_name in seen:
            continue
        seen.add(tool_name)
        names.append(tool_name)
    return names


def _build_system_prompt(cwd: Path) -> str:
    config = load_config()
    return build_system_prompt(
        config.system_prompt,
        mode="agent",
        cwd=cwd,
        tool_names=_tool_names_from_registry(cwd),
    )


def _mentions_project_shape(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in _PROJECT_HINTS)


def _suggests_next_action(text: str) -> bool:
    lowered = text.lower()
    has_phrase = any(phrase in lowered for phrase in _NEXT_STEP_PHRASES)
    has_verb = any(verb in lowered for verb in _NEXT_STEP_VERBS)
    return has_phrase and (has_verb or "?" in text)


def _build_eval_tools(cwd: Path) -> list[Tool[None]]:
    bash_tool = make_bash_tool(cwd=cwd)

    async def web_search(query: str, timeout: float = 30.0) -> str:
        """Search the web for public information.

        This tool is available to mirror normal agent mode, but it should not be
        needed for this eval.
        """
        _ = query, timeout
        return "EVAL_STUB: web_search should not be used for this question."

    async def web_fetch(url: str, timeout: float = 30.0) -> str:
        """Fetch web pages or HTTP endpoints.

        This tool is available to mirror normal agent mode, but it should not be
        needed for this eval.
        """
        _ = url, timeout
        return "EVAL_STUB: web_fetch should not be used for this question."

    return [
        Tool(bash_tool, name="bash"),
        Tool(web_search, name="web_search"),
        Tool(web_fetch, name="web_fetch"),
    ]


async def evaluate_case(model_name: str, case: EvalCase, cwd: Path) -> EvalResult:
    model_entry = resolve_model_entry(model_name, get_all_entries())
    runtime = ChatRuntime(model_entry)
    tool_events: list[RuntimeToolEvent] = []
    latest_text = ""

    def on_text_update(text: str) -> None:
        nonlocal latest_text
        latest_text = text

    def on_tool_event(event: RuntimeToolEvent) -> None:
        tool_events.append(event)

    try:
        runtime_response = await runtime.run_stream(
            prompt_text=case.prompt,
            system_prompt=_build_system_prompt(cwd),
            message_history=[],
            attachments=[],
            tools=_build_eval_tools(cwd),
            model_settings={},
            tool_call_limit=1,
            allow_tool_fallback=False,
            on_text_update=on_text_update,
            on_tool_event=on_tool_event,
        )
    except RuntimePartialRunError as exc:
        return EvalResult(
            case=case,
            model=model_name,
            final_text=latest_text,
            error=str(exc.cause),
            tool_events=tool_events,
            tool_limit_recovery_used=False,
        )
    except Exception as exc:
        return EvalResult(
            case=case,
            model=model_name,
            final_text=latest_text,
            error=str(exc),
            tool_events=tool_events,
            tool_limit_recovery_used=False,
        )

    return EvalResult(
        case=case,
        model=model_name,
        final_text=runtime_response.text,
        error=None,
        tool_events=tool_events,
        tool_limit_recovery_used=runtime_response.tool_limit_recovery_used,
    )


def parse_args() -> EvalArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--model",
        default=DEFAULT_EVAL_MODEL,
        help="Model id to evaluate.",
    )
    namespace = parser.parse_args()
    return EvalArgs(model=namespace.model)


def _print_report(result: EvalResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.case.name}")
    print(f"Model: {result.model}")
    print(f"Prompt: {result.case.prompt}")
    print()
    print("Checks:")
    for check in result.checks():
        prefix = "PASS" if check.passed else "FAIL"
        detail = f" — {check.detail}" if check.detail else ""
        print(f"- {prefix}: {check.name}{detail}")

    print()
    print(f"Runtime tool-limit recovery: {'yes' if result.tool_limit_recovery_used else 'no'}")
    print()
    print("Tool events:")
    if not result.tool_events:
        print("- (none)")
    else:
        for event in result.tool_events:
            arguments = json.dumps(event.arguments, sort_keys=True)
            suffix = " [error]" if event.is_error else ""
            print(f"- {event.phase}: {event.tool_name} {arguments}{suffix}")

    print()
    print("Final answer:")
    print(result.final_text or "(empty)")

    if result.error is not None:
        print()
        print("Runtime error:")
        print(result.error)


def main() -> int:
    args = parse_args()
    cwd = _require_repo_root()
    cases = build_cases()
    if not cases:
        print("No cases configured.", file=sys.stderr)
        return 2

    result = asyncio.run(evaluate_case(args.model, cases[0], cwd))
    _print_report(result)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
