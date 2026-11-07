"""Non-authorizing handoffs from detached workers to a native foreground gate.

Opt-in is profile-local: approvals.slack_owner_id. A handoff is evidence of a
blocked operation, NEVER a ticket that /approve can spend or an execution queue.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import uuid

from tools import approval_context

_records = ContextVar("approval_handoff_records", default=None)
_execution_scope = ContextVar("approval_execution_scope", default=None)


def owner_id():
    value = approval_context._get_approval_config().get("slack_owner_id")
    return value if isinstance(value, str) else ""


@contextmanager
def collect_handoffs():
    records = []
    token = _records.set(records)
    try:
        yield records
    finally:
        _records.reset(token)


@contextmanager
def execution_scope(plan, *, workdir=None, background=False, pty=False):
    token = _execution_scope.set({"cwd": plan.cwd, "backend": plan.env_type,
                                 "task_id": plan.effective_task_id, "timeout": plan.effective_timeout,
                                 "workdir": workdir, "background": background, "pty": pty})
    try:
        yield
    finally:
        _execution_scope.reset(token)


def current_execution_scope():
    return dict(_execution_scope.get() or {})


def attach_handoffs(entry, result):
    if result.get("approval_handoffs"):
        entry["approval_handoffs"] = result["approval_handoffs"]
        entry["summary"] = (
            "Approval handoff: blocked commands were NOT executed. No approval was delivered or is active. "
            "Revalidate the original authority, expiry, scope, budget, counters, evidence and idempotency; "
            "present through the native foreground approval flow in the original authenticated thread, "
            "not another delegation. Do not retry expired/denied operations without a new user request.\n"
            + json.dumps(result["approval_handoffs"], ensure_ascii=False) + "\n" + entry["summary"]
        )
    return entry


def handoff_if_detached(command, description, pattern_keys, session_key, *, unavailable=False):
    from agent.delegation_context import is_delegated_child_context
    if not owner_id() or not (is_delegated_child_context() or unavailable):
        return None
    from agent.redact import redact_sensitive_text
    from hermes_constants import get_hermes_home
    record = {
        "handoff_id": uuid.uuid4().hex,
        "session_key": session_key,
        "profile_home": str(get_hermes_home()),
        "command": redact_sensitive_text(command, force=True),
        "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "description": redact_sensitive_text(description, force=True),
        "pattern_keys": list(pattern_keys),
        "execution_scope": current_execution_scope(),
        "tool_call_id": approval_context._approval_tool_call_id.get(),
        "session_id": approval_context._approval_session_id.get(),
        "executed": False,
        "delivery_confirmed": False,
        "active_request": False,
        "revalidate": ["identity", "origin", "scope", "authorization_expiry", "budget",
                       "counters", "evidence", "idempotency", "execution_arguments"],
    }
    records = _records.get()
    if records is not None:
        records.append(record)
    return {
        "approved": False, "status": "approval_handoff", "approval_pending": False,
        "handoff": record,
        "message": (
            "NOT EXECUTED. No approval card was delivered and no request is active. "
            "Return this handoff to the parent; do not retry or switch tools. The parent must "
            "revalidate original authorization, validity, budget, counters, evidence and idempotency "
            "without resetting them, recover exact arguments from the originating tool call, "
            "and use the native terminal approval flow in that authenticated origin. "
            "If any precondition cannot be verified, stop. This handoff grants NO authority. "
            "Never replay redacted text or claim to be awaiting approval."
        ),
    }
