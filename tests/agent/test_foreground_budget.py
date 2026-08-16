from __future__ import annotations

import math

import pytest


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_disabled_guard_preserves_existing_tool_behavior() -> None:
    from agent.foreground_budget import ForegroundBudgetGuard

    clock = FakeClock()
    guard = ForegroundBudgetGuard(0, clock=clock)
    guard.start_turn()
    clock.now = 10_000

    assert guard.before_tool("write_file", {"path": "x"}).allowed
    assert guard.before_tool("terminal", {"command": "sleep 1"}).allowed


def test_budget_blocks_new_foreground_tool_after_deadline() -> None:
    from agent.foreground_budget import ForegroundBudgetGuard

    clock = FakeClock()
    guard = ForegroundBudgetGuard(30, clock=clock)
    guard.start_turn()
    clock.now = 30

    decision = guard.before_tool("write_file", {"path": "x"})

    assert not decision.allowed
    assert decision.code == "foreground_budget_exhausted"
    assert "delegate_task" in decision.message


def test_budget_allows_exactly_one_background_terminal_handoff() -> None:
    from agent.foreground_budget import ForegroundBudgetGuard

    clock = FakeClock()
    guard = ForegroundBudgetGuard(5, clock=clock)
    guard.start_turn()
    clock.now = 6

    foreground = guard.before_tool("terminal", {"command": "work"})
    handoff = guard.before_tool(
        "terminal", {"command": "work", "background": True}
    )
    duplicate = guard.before_tool(
        "terminal", {"command": "other", "background": True}
    )

    assert not foreground.allowed
    assert handoff.allowed and handoff.is_handoff
    assert not duplicate.allowed
    assert duplicate.code == "foreground_handoff_already_started"


def test_delegate_and_cron_create_are_safe_handoffs_but_cron_reads_are_not() -> None:
    from agent.foreground_budget import ForegroundBudgetGuard

    clock = FakeClock()
    guard = ForegroundBudgetGuard(1, clock=clock)
    guard.start_turn()
    clock.now = 2

    assert not guard.before_tool("cronjob", {"action": "list"}).allowed
    assert guard.before_tool(
        "cronjob", {"action": "create", "schedule": "in 1m", "prompt": "finish"}
    ).is_handoff

    guard.start_turn()
    clock.now = 4
    assert guard.before_tool("delegate_task", {"goal": "finish"}).is_handoff


def test_new_turn_resets_deadline_and_handoff_state() -> None:
    from agent.foreground_budget import ForegroundBudgetGuard

    clock = FakeClock()
    guard = ForegroundBudgetGuard(5, clock=clock)
    guard.start_turn()
    clock.now = 6
    assert guard.before_tool("delegate_task", {"goal": "one"}).allowed

    guard.start_turn()
    assert guard.before_tool("write_file", {"path": "x"}).allowed
    clock.now = 12
    assert guard.before_tool("delegate_task", {"goal": "two"}).allowed


@pytest.mark.parametrize(
    "value",
    ["abc", "30", -1, math.nan, math.inf, -math.inf, True, None],
)
def test_guard_rejects_invalid_budget_instead_of_failing_open(value) -> None:
    from agent.foreground_budget import ForegroundBudgetGuard

    with pytest.raises(ValueError, match="finite non-negative number"):
        ForegroundBudgetGuard(value)
