"""Tests for the bash approval modal."""

import asyncio

from mother import MotherApp
from mother.bash_approval_screen import BashApprovalScreen
from mother.tools.bash_guard import BashGuardDecision


def _decision() -> BashGuardDecision:
    return BashGuardDecision(
        command="rm -rf tmpdir",
        label="Fatal",
        raw_output="LABEL: Fatal",
        canonical_label=True,
        model_name="test-guard",
    )


def test_bash_approval_screen_approves_on_enter() -> None:
    async def run() -> None:
        app = MotherApp()
        result: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        def on_result(approved: bool | None) -> None:
            result.set_result(bool(approved))

        async with app.run_test() as pilot:
            push_result: object = app.push_screen(BashApprovalScreen(_decision()), on_result)
            _ = push_result
            await pilot.pause()
            await pilot.press("enter")
            assert await result is True

    asyncio.run(run())


def test_bash_approval_screen_denies_on_escape() -> None:
    async def run() -> None:
        app = MotherApp()
        result: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        def on_result(approved: bool | None) -> None:
            result.set_result(bool(approved))

        async with app.run_test() as pilot:
            push_result: object = app.push_screen(BashApprovalScreen(_decision()), on_result)
            _ = push_result
            await pilot.pause()
            await pilot.press("escape")
            assert await result is False

    asyncio.run(run())
