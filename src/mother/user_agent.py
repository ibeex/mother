"""Versioned User-Agent string that identifies Mother to every outbound server.

Mother is built on the OpenAI/Anthropic SDKs, and both stamp their own
``AsyncOpenAI/Python <sdk version>`` header on every request.  Gateways that
count clients by User-Agent (LiteLLM's ``harness`` label, for example) then see
``openai-sdk`` and cannot tell Mother apart from any other SDK script, nor which
Mother release is running.

Everything Mother sends -- model traffic, update checks, web tools -- therefore
carries :func:`user_agent` (``mother/<version>``) instead.
"""

from __future__ import annotations

from functools import cache
from importlib.metadata import PackageNotFoundError, version

PACKAGE_NAME = "mother"


@cache
def installed_version() -> str:
    """Return the installed distribution version, or ``0.0.0`` from a source tree."""
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0.0.0"


@cache
def user_agent() -> str:
    """Return the identification string Mother sends, e.g. ``mother/0.9.36``."""
    return f"mother/{installed_version()}"
