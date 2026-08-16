from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.foreground_budget import ForegroundBudgetGuard
from agent.tool_guardrails import ToolGuardrailDecision


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeAgent:
    def __init__(self, guard: ForegroundBudgetGuard) -> None:
        self._foreground_budget_guard = guard
        self._interrupt_requested = False
        self.log_prefix = ""
        self.quiet_mode = True
        self.verbose_logging = False
        self.log_prefix_chars = 100
        self.tool_progress_mode = "off"
        self.tool_progress_callback = None
        self.tool_start_callback = None
        self.tool_complete_callback = None
        self.tool_delay = 0
        self._checkpoint_mgr = SimpleNamespace(enabled=False)
        self._tool_guardrails = SimpleNamespace(
            before_call=lambda *_: ToolGuardrailDecision(action="allow")
        )
        self._tool_guardrail_halt_decision = None
        self._turns_since_memory = 0
        self._iters_since_skill = 0
        self._current_tool = None
        self._current_turn_id = "turn"
        self._current_api_request_id = "api"
        self._session_db = None
        self.session_id = "session"
        self.valid_tool_names = {"write_file", "terminal"}
        self.enabled_toolsets = None
        self.disabled_toolsets = None
        self._context_engine_tool_names = set()
        self._memory_manager = None
        self._subdirectory_hints = SimpleNamespace(check_tool_call=lambda *_: "")
        self._print_fn = print
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._invoke_tool = MagicMock(return_value='{"success": true}')

    def _touch_activity(self, _description: str) -> None:
        pass

    def _vprint(self, *_args, **_kwargs) -> None:
        pass

    def _should_emit_quiet_tool_messages(self) -> bool:
        return False

    def _should_start_quiet_spinner(self) -> bool:
        return False

    def _flush_messages_to_session_db(self, _messages) -> None:
        pass

    def _apply_pending_steer_to_tool_results(self, *_args) -> None:
        pass

    def _tool_result_content_for_active_model(self, _name, result):
        return result

    def _record_file_mutation_result(self, *_args) -> None:
        pass

    def _append_guardrail_observation(self, _name, _args, result, **_kwargs):
        return result


class FakeToolCall:
    def __init__(self, name: str, args: dict, call_id: str = "call-1") -> None:
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=json.dumps(args))


def test_guard_factory_only_enables_configured_gateway_sessions() -> None:
    from agent.foreground_budget import foreground_budget_guard_from_config
    from hermes_cli.config import DEFAULT_CONFIG

    gateway = foreground_budget_guard_from_config(
        {"agent": {"foreground_budget_seconds": 15}},
        gateway_session_key="agent:main:telegram:dm:1",
    )
    cli = foreground_budget_guard_from_config(
        {"agent": {"foreground_budget_seconds": 15}},
        gateway_session_key=None,
    )
    default = foreground_budget_guard_from_config(
        {"agent": {}}, gateway_session_key="agent:main:telegram:dm:1"
    )

    assert gateway.budget_seconds == 15
    assert not cli.enabled
    assert not default.enabled
    assert DEFAULT_CONFIG["agent"]["foreground_budget_seconds"] == 0


def test_guard_factory_enables_strict_handoff_only_for_selected_platform() -> None:
    from agent.foreground_budget import foreground_budget_guard_from_config

    config = {
        "agent": {
            "strict_background_handoff_platforms": ["slack"],
            "strict_background_handoff_ack": "Checking in background.",
        }
    }

    slack = foreground_budget_guard_from_config(
        config,
        gateway_session_key="agent:main:slack:dm:D123:thread",
        platform="slack",
    )
    telegram = foreground_budget_guard_from_config(
        config,
        gateway_session_key="agent:main:telegram:dm:123",
        platform="telegram",
    )

    assert slack.strict_background_handoff is True
    assert slack.strict_handoff_ack == "Checking in background."
    assert telegram.strict_background_handoff is False


def test_strict_handoff_latches_successful_dispatch_before_deadline() -> None:
    clock = FakeClock()
    guard = ForegroundBudgetGuard(
        30,
        strict_background_handoff=True,
        strict_handoff_ack="Working in background.",
        clock=clock,
    )
    guard.start_turn()
    clock.now = 2

    assert guard.observe_tool_result(
        "delegate_task",
        {"goal": "do work"},
        '{"status":"dispatched","delegation_id":"deleg_1"}',
    )
    decision = guard.before_tool("terminal", {"command": "should-not-run"})

    assert decision.allowed is False
    assert decision.code == "foreground_handoff_already_started"
    assert guard.consume_strict_handoff_ack() == "Working in background."
    assert guard.consume_strict_handoff_ack() is None


def test_sequential_executor_stops_same_batch_after_strict_dispatch(monkeypatch) -> None:
    from agent.tool_executor import execute_tool_calls_sequential
    import run_agent

    guard = ForegroundBudgetGuard(30, strict_background_handoff=True)
    guard.start_turn()
    agent = FakeAgent(guard)
    agent.valid_tool_names.add("delegate_task")
    agent._dispatch_delegate_task = MagicMock(
        return_value='{"status":"dispatched","delegation_id":"deleg_1"}'
    )
    foreground_dispatch = MagicMock(return_value='{"success": true}')
    monkeypatch.setattr(run_agent, "handle_function_call", foreground_dispatch)
    messages = []
    assistant = SimpleNamespace(
        tool_calls=[
            FakeToolCall("delegate_task", {"goal": "background"}, "call-a"),
            FakeToolCall("write_file", {"path": "x", "content": "y"}, "call-b"),
        ]
    )

    execute_tool_calls_sequential(agent, assistant, messages, "task")

    agent._dispatch_delegate_task.assert_called_once()
    foreground_dispatch.assert_not_called()
    assert len(messages) == 2
    assert json.loads(messages[0]["content"])["status"] == "dispatched"
    assert "skipped" in messages[1]["content"].lower()
    assert guard.consume_strict_handoff_ack()


def test_sequential_executor_blocks_expired_foreground_call(monkeypatch) -> None:
    from agent.tool_executor import execute_tool_calls_sequential
    import run_agent

    clock = FakeClock()
    guard = ForegroundBudgetGuard(5, clock=clock)
    guard.start_turn()
    clock.now = 6
    agent = FakeAgent(guard)
    dispatch = MagicMock(return_value='{"success": true}')
    monkeypatch.setattr(run_agent, "handle_function_call", dispatch)
    messages = []
    assistant = SimpleNamespace(
        tool_calls=[FakeToolCall("write_file", {"path": "x", "content": "y"})]
    )

    execute_tool_calls_sequential(agent, assistant, messages, "task")

    dispatch.assert_not_called()
    result = json.loads(messages[0]["content"])
    assert result["error"] == "foreground_budget_exhausted"
    assert result["status"] == "blocked"


def test_sequential_executor_allows_background_handoff(monkeypatch) -> None:
    from agent.tool_executor import execute_tool_calls_sequential
    import run_agent

    clock = FakeClock()
    guard = ForegroundBudgetGuard(5, clock=clock)
    guard.start_turn()
    clock.now = 6
    agent = FakeAgent(guard)
    dispatch = MagicMock(return_value='{"success": true, "session_id": "bg-1"}')
    monkeypatch.setattr(run_agent, "handle_function_call", dispatch)
    messages = []
    assistant = SimpleNamespace(
        tool_calls=[
            FakeToolCall(
                "terminal", {"command": "long-job", "background": True}
            )
        ]
    )

    execute_tool_calls_sequential(agent, assistant, messages, "task")

    dispatch.assert_called_once()
    assert json.loads(messages[0]["content"])["success"] is True


def test_tool_admitted_before_deadline_finishes_but_next_tool_is_blocked(
    monkeypatch,
) -> None:
    from agent.tool_executor import execute_tool_calls_sequential
    import run_agent

    clock = FakeClock()
    guard = ForegroundBudgetGuard(5, clock=clock)
    guard.start_turn()
    clock.now = 4
    agent = FakeAgent(guard)

    def finish_after_deadline(*_args, **_kwargs):
        clock.now = 6
        return '{"success": true}'

    dispatch = MagicMock(side_effect=finish_after_deadline)
    monkeypatch.setattr(run_agent, "handle_function_call", dispatch)
    messages = []
    assistant = SimpleNamespace(
        tool_calls=[
            FakeToolCall("write_file", {"path": "a", "content": "1"}, "call-a"),
            FakeToolCall("write_file", {"path": "b", "content": "2"}, "call-b"),
        ]
    )

    execute_tool_calls_sequential(agent, assistant, messages, "task")

    dispatch.assert_called_once()
    assert json.loads(messages[0]["content"])["success"] is True
    assert json.loads(messages[1]["content"])["error"] == "foreground_budget_exhausted"


def test_concurrent_executor_blocks_every_expired_foreground_call() -> None:
    from agent.tool_executor import execute_tool_calls_concurrent

    clock = FakeClock()
    guard = ForegroundBudgetGuard(5, clock=clock)
    guard.start_turn()
    clock.now = 6
    agent = FakeAgent(guard)
    messages = []
    assistant = SimpleNamespace(
        tool_calls=[
            FakeToolCall("write_file", {"path": "a", "content": "1"}, "call-a"),
            FakeToolCall("write_file", {"path": "b", "content": "2"}, "call-b"),
        ]
    )

    execute_tool_calls_concurrent(agent, assistant, messages, "task")

    agent._invoke_tool.assert_not_called()
    assert len(messages) == 2
    assert {
        json.loads(message["content"])["error"] for message in messages
    } == {"foreground_budget_exhausted"}


def test_concurrent_executor_admits_only_one_handoff_from_same_batch() -> None:
    from agent.tool_executor import execute_tool_calls_concurrent

    clock = FakeClock()
    guard = ForegroundBudgetGuard(5, clock=clock)
    guard.start_turn()
    clock.now = 6
    agent = FakeAgent(guard)
    messages = []
    assistant = SimpleNamespace(
        tool_calls=[
            FakeToolCall(
                "terminal",
                {"command": "job-a", "background": True},
                "call-a",
            ),
            FakeToolCall(
                "terminal",
                {"command": "job-b", "background": True},
                "call-b",
            ),
        ]
    )

    execute_tool_calls_concurrent(agent, assistant, messages, "task")

    agent._invoke_tool.assert_called_once()
    results = [json.loads(message["content"]) for message in messages]
    assert sum(result.get("success") is True for result in results) == 1
    assert sum(
        result.get("error") == "foreground_handoff_already_started"
        for result in results
    ) == 1


def test_tool_search_underlying_foreground_tool_is_blocked_after_deadline(
    monkeypatch,
) -> None:
    from agent import tool_executor
    from tools import tool_search
    import run_agent

    clock = FakeClock()
    guard = ForegroundBudgetGuard(5, clock=clock)
    guard.start_turn()
    clock.now = 6
    agent = FakeAgent(guard)
    dispatch = MagicMock(return_value='{"success": true}')
    monkeypatch.setattr(run_agent, "handle_function_call", dispatch)
    monkeypatch.setattr(
        tool_search,
        "resolve_underlying_call",
        lambda _args: ("write_file", {"path": "x", "content": "y"}, None),
    )
    monkeypatch.setattr(
        tool_executor, "_tool_search_scoped_names", lambda _agent: {"write_file"}
    )
    messages = []
    assistant = SimpleNamespace(
        tool_calls=[FakeToolCall("tool_call", {"name": "write_file"})]
    )

    tool_executor.execute_tool_calls_sequential(agent, assistant, messages, "task")

    dispatch.assert_not_called()
    assert json.loads(messages[0]["content"])["error"] == "foreground_budget_exhausted"


def test_plugin_blocked_handoff_consumes_the_only_safe_slot(monkeypatch) -> None:
    from agent.tool_executor import execute_tool_calls_sequential
    from hermes_cli import plugins
    import run_agent

    clock = FakeClock()
    guard = ForegroundBudgetGuard(5, clock=clock)
    guard.start_turn()
    clock.now = 6
    agent = FakeAgent(guard)
    agent.valid_tool_names.add("delegate_task")
    dispatch = MagicMock(return_value='{"success": true}')
    monkeypatch.setattr(run_agent, "handle_function_call", dispatch)
    monkeypatch.setattr(plugins, "resolve_pre_tool_block", lambda *_a, **_kw: "denied")
    messages = []
    assistant = SimpleNamespace(
        tool_calls=[
            FakeToolCall("delegate_task", {"goal": "one"}, "call-a"),
            FakeToolCall("delegate_task", {"goal": "two"}, "call-b"),
        ]
    )

    execute_tool_calls_sequential(agent, assistant, messages, "task")

    dispatch.assert_not_called()
    assert json.loads(messages[0]["content"])["error"] == "denied"
    assert (
        json.loads(messages[1]["content"])["error"]
        == "foreground_handoff_already_started"
    )


def test_conversation_starts_a_fresh_budget_before_turn_setup(monkeypatch) -> None:
    from agent import conversation_loop

    guard = MagicMock()
    agent = SimpleNamespace(_foreground_budget_guard=guard)

    def fail_after_asserting_timer_started(*_args, **_kwargs):
        guard.start_turn.assert_called_once_with()
        raise RuntimeError("stop after prologue boundary")

    monkeypatch.setattr(conversation_loop, "build_turn_context", fail_after_asserting_timer_started)

    with pytest.raises(RuntimeError, match="prologue boundary"):
        conversation_loop.run_conversation(agent, "hello")


def test_configured_guard_fails_closed_for_opaque_codex_runtime(monkeypatch) -> None:
    from agent import conversation_loop

    guard = SimpleNamespace(enabled=True, start_turn=MagicMock())
    agent = SimpleNamespace(
        _foreground_budget_guard=guard,
        api_mode="codex_app_server",
    )
    turn_setup = MagicMock()
    monkeypatch.setattr(conversation_loop, "build_turn_context", turn_setup)

    result = conversation_loop.run_conversation(agent, "hello")

    turn_setup.assert_not_called()
    assert result["failed"] is True
    assert result["error"] == "foreground_budget_unsupported_runtime"
