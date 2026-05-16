"""Modal screen prompting the user to approve risky bash commands."""

from __future__ import annotations

from pathlib import Path, PurePath
from typing import ClassVar, override

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from mother.tools.bash_guard import BashGuardDecision

_CSS_DIR = Path(__file__).resolve().parent / "css"


class BashApprovalScreen(ModalScreen[bool]):
    """Ask the user whether a warning/fatal bash command should run."""

    CSS_PATH: ClassVar[str | PurePath | list[str | PurePath] | None] = (
        _CSS_DIR / "bash_approval.tcss"
    )

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "approve", "Run"),
        Binding("enter", "approve", show=False),
        Binding("n", "deny", "Copy command"),
        Binding("escape", "deny", show=False),
    ]

    def __init__(self, decision: BashGuardDecision) -> None:
        super().__init__()
        self.decision: BashGuardDecision = decision

    @override
    def compose(self) -> ComposeResult:
        title = f"{self.decision.label}: run this command anyway?"
        help_text = (
            "Press Y or Enter to run it in Mother. "
            "Press N or Escape to deny and copy it to the clipboard instead."
        )
        with Container(id="bash-approval-modal"):
            with Vertical(id="bash-approval-dialog"):
                yield Static(title, id="bash-approval-title")
                yield Static("Command", classes="bash-approval-label")
                yield Static(self.decision.command, id="bash-approval-command")
                yield Static(
                    f"Guard model: {self.decision.model_name}",
                    id="bash-approval-model",
                )
                if self.decision.error is not None:
                    yield Static(
                        f"Reason: {self.decision.error}",
                        id="bash-approval-reason",
                    )
                yield Static(help_text, id="bash-approval-help")

    def action_approve(self) -> None:
        """Approve the pending command."""
        _ = self.dismiss(True)

    def action_deny(self) -> None:
        """Deny the pending command."""
        _ = self.dismiss(False)
