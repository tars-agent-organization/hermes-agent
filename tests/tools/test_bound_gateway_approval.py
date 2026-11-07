"""Real gate and queue; no terminal backend, paid API or network is used."""
from concurrent.futures import ThreadPoolExecutor
import contextvars
import threading

import pytest

from tools import approval, approval_context
from tools.approval_binding import bind_request
from tools.approval_gateway_wait import _await_gateway_decision

ORIGIN = {"user_id": "OWNER", "chat_id": "C", "thread_id": "1.2", "team_id": "T"}


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 30)
    yield
    for key in ("s", "other"):
        approval.unregister_gateway_notify(key)


def request(command="command A", validate=lambda: True, deliver=True, after_notify=None):
    delivered = threading.Event()
    record = {}

    def notify(data):
        record.update(data)
        assert bind_request("s", data["request_id"], ORIGIN, validate)
        if deliver:
            assert approval.ack_gateway_approval("s", data["request_id"])
        delivered.set()
        if after_notify:
            after_notify(data)

    pool = ThreadPoolExecutor(1)
    future = pool.submit(contextvars.copy_context().run, _await_gateway_decision,
                         "s", notify, {"command": command, "pattern_keys": [command], "requires_binding": True})
    assert delivered.wait(5)
    return pool, future, record


@pytest.mark.parametrize("choice", ["once", "deny"])
def test_exact_request_only_and_duplicate_is_inert(choice):
    pool_a, a, record_a = request()
    pool_b, b, record_b = request("command B")
    try:
        assert approval.resolve_gateway_approval("s", choice, request_id=record_b["request_id"], origin=ORIGIN) == 1
        assert b.result(5)["choice"] == choice
        assert not a.done()
        assert approval.resolve_gateway_approval("s", "once", request_id=record_b["request_id"], origin=ORIGIN) == 0
        assert approval.resolve_gateway_approval("s", "deny", request_id=record_a["request_id"], origin=ORIGIN) == 1
        assert a.result(5)["choice"] == "deny"
    finally:
        approval.unregister_gateway_notify("s")
        pool_a.shutdown(); pool_b.shutdown()


@pytest.mark.parametrize("field", ["user_id", "chat_id", "thread_id", "team_id"])
def test_origin_isolation(field):
    pool, future, record = request()
    try:
        wrong = {**ORIGIN, field: "different"}
        assert approval.resolve_gateway_approval("s", "once", request_id=record["request_id"], origin=wrong) == 0
        assert approval.resolve_gateway_approval("other", "once", request_id=record["request_id"], origin=ORIGIN) == 0
        assert approval.resolve_gateway_approval("s", "once") == 0
        assert approval.resolve_gateway_approval("s", "always", request_id=record["request_id"], origin=ORIGIN) == 0
        assert approval.resolve_gateway_approval("s", "once", resolve_all=True, request_id=record["request_id"], origin=ORIGIN) == 0
    finally:
        approval.unregister_gateway_notify("s")
        assert future.result(5).get("cancelled")
        pool.shutdown()


def test_unconfirmed_delivery_cannot_authorize():
    pool, future, record = request(deliver=False)
    try:
        assert approval.resolve_gateway_approval("s", "once", request_id=record["request_id"], origin=ORIGIN) == 0
    finally:
        approval.unregister_gateway_notify("s")
        assert future.result(5).get("cancelled")
        pool.shutdown()


def test_timeout_and_late_click(monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 0)
    pool, future, record = request()
    try:
        assert future.result(5)["resolved"] is False
        assert approval.resolve_gateway_approval("s", "once", request_id=record["request_id"], origin=ORIGIN) == 0
    finally:
        pool.shutdown()


def test_authorization_expires_after_click_before_resume():
    state = {"authorized": True}
    def click_then_revoke(data):
        assert approval.resolve_gateway_approval("s", "once", request_id=data["request_id"], origin=ORIGIN) == 1
        state["authorized"] = False
    pool, future, _ = request(validate=lambda: state["authorized"], after_notify=click_then_revoke)
    try:
        outcome = future.result(5)
        assert outcome["choice"] is None
        assert "validity changed" in outcome["cancelled"]
    finally:
        pool.shutdown()


def test_notify_failure_withdraws_even_if_callback_clicked():
    def notify(data):
        raise RuntimeError("Slack unavailable")
    result = _await_gateway_decision("s", notify, {"command": "A", "requires_binding": True})
    assert result["notify_failed"]
    assert not approval.has_blocking_approval("s")


def test_expired_deadline_cannot_be_resolved_before_waiter_observes_timeout(monkeypatch):
    pool, future, record = request()
    try:
        with approval._lock:
            approval._gateway_queues["s"][0].expires_at = 0
        assert approval.resolve_gateway_approval("s", "once", request_id=record["request_id"], origin=ORIGIN) == 0
    finally:
        approval.unregister_gateway_notify("s")
        assert future.result(5).get("cancelled")
        pool.shutdown()
