"""Optional, display-only Bash formatting without executing shell code."""

import shutil
import subprocess


def format_bash_for_display(command: str) -> str:
    """Use shfmt when available, preserving the original on any formatter failure."""
    executable = shutil.which("shfmt")
    if executable is None:
        return command

    try:
        result = subprocess.run(
            [executable, "-ln", "bash", "-i", "2"],
            input=command,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        return command

    if result.returncode != 0 or not result.stdout.strip():
        return command
    return result.stdout.removesuffix("\n")
