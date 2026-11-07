"""Request-specific, fail-closed constraints for owner-bound native approvals."""
import logging
import time

logger = logging.getLogger("tools.approval")


def bound_request_reply(session_key):
    from tools.approval import list_gateway_approvals
    if any(e.get("requires_binding") for e in list_gateway_approvals(session_key)):
        return ("Use Approve once or Deny on the specific command card in its original thread. "
                "This request cannot be approved by FIFO, session-wide or /approve all.")
    return None


def bind_request(session_key, request_id, origin, validate):
    from tools import approval
    with approval._lock:
        entry = next((e for e in approval._gateway_queues.get(session_key, [])
                      if e.data["request_id"] == request_id), None)
        if entry is None:
            return False
        entry.data["approval_origin"] = dict(origin)
        entry.data["allow_session"] = entry.data["allow_permanent"] = False
        entry.validate = validate
        return True


def valid(entry, *, origin=None, resolving=False):
    expected = entry.data.get("approval_origin")
    if expected is None:
        return not entry.data.get("requires_binding")
    if time.monotonic() >= entry.expires_at or not entry.acknowledged:
        return False
    if resolving and origin != expected:
        return False
    try:
        return entry.validate is not None and entry.validate() is True
    except Exception:
        logger.warning("Bound approval revalidation failed", exc_info=True)
        return False
