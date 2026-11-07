"""Detached child must return evidence, not an orphaned executable approval."""
import contextvars
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import approval, approval_context
from agent.delegation_context import delegated_child_context
from tools.delegate_tool_child_run import _ChildRun, _build_result_entry, _SchemaOutcome
from tools.terminal_tool import _run_approval_guards, _Rejected


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setattr(approval_context, "_get_approval_config",
                        lambda: {"mode": "manual", "slack_owner_id": "U_OWNER"})
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    token = approval_context.set_current_session_key("agent:lerian:slack:C:thread-A")
    yield
    approval.unregister_gateway_notify(approval_context.get_current_session_key())
    approval_context.reset_current_session_key(token)


@pytest.mark.parametrize("parent_ended", [False, True])
def test_child_handoff_survives_parent_exit_and_preserves_accounting(parent_ended):
    session = approval_context.get_current_session_key()
    notified = Mock()
    approval.register_gateway_notify(session, notified)
    worker_context = contextvars.copy_context()
    if parent_ended:
        approval.unregister_gateway_notify(session)

    class Child:
        session_id = "child-A"
        session_prompt_tokens = 123
        session_completion_tokens = 45
        session_estimated_cost_usd = 0.3
        session_cost_status = "estimated"
        model = "test-model"

        def run_conversation(self, **kwargs):
            with pytest.raises(_Rejected) as rejected:
                _run_approval_guards("rm -rf ./fixture", "local", {}, force=False)
            payload = json.loads(rejected.value.args[0])
            assert payload["status"] == "approval_handoff"
            assert payload["approval_pending"] is False
            return {"final_response": "blocked", "completed": True, "api_calls": 2}

    child = Child()
    run = _ChildRun(child, SimpleNamespace(), 0, "task", None, None)
    result, failure, _ = worker_context.run(run.await_child)
    assert failure is None
    record = result["approval_handoffs"][0]
    assert record["command"] == "rm -rf ./fixture"
    assert record["session_key"] == session
    assert record["active_request"] is record["delivery_confirmed"] is record["executed"] is False
    assert approval.resolve_gateway_approval(session, "once") == 0
    assert not approval.has_blocking_approval(session)
    notified.assert_not_called()
    entry = _build_result_entry(child, result, 0, 1, _SchemaOutcome(None, None, [], 0))
    assert entry["tokens"] == {"input": 123, "output": 45}
    assert entry["cost_usd"] == 0.3 and entry["api_calls"] == 2
    assert entry["approval_handoffs"] == result["approval_handoffs"]
    assert "NOT executed" in entry["summary"]


def test_opt_in_does_not_change_other_profile(monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": "manual"})
    from tools.approval_handoff import handoff_if_detached
    with delegated_child_context():
        assert handoff_if_detached("command", "reason", [], "s") is None


@pytest.mark.parametrize("parent_ended", [False, True])
def test_automatic_mode_keeps_child_unprompted_after_parent_exit(monkeypatch, parent_ended):
    from tools.approval_handoff import collect_handoffs
    monkeypatch.setattr(approval_context, "_get_approval_config",
                        lambda: {"mode": "off", "slack_owner_id": "U_OWNER"})
    session = approval_context.get_current_session_key()
    notified = Mock()
    approval.register_gateway_notify(session, notified)
    if parent_ended:
        approval.unregister_gateway_notify(session)
    with delegated_child_context(), collect_handoffs() as records:
        # Exercise only the guard; no shell command is executed by this test.
        result = approval.check_all_command_guards("rm -rf ./fixture", "local")
    assert result["approved"] is True
    assert not records
    assert not approval.has_blocking_approval(session)
    notified.assert_not_called()


def test_handoff_redacts_but_retains_fingerprint():
    from tools.approval_handoff import collect_handoffs, handoff_if_detached
    import hashlib
    secret = "ghp_" + "X" * 36
    command = f"curl -H 'Authorization: token {secret}' https://example.test"
    with delegated_child_context(), collect_handoffs() as records:
        outcome = handoff_if_detached(command, "reason", ["key"], "s")
    assert secret not in json.dumps(outcome)
    assert records[0]["command_sha256"] == hashlib.sha256(command.encode()).hexdigest()
