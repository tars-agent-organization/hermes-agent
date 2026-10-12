"""Privacy-safe, point-in-time work snapshot for gateway status commands.

The gateway's foreground runner map, delegated-worker registry, terminal
process registry, and asyncio housekeeping tasks are different lifecycles.
This adapter deliberately keeps them separate so ``/agents`` and ``/tasks``
cannot count one source while rendering another.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, List

_ACTIVE_DELEGATION_STATES = frozenset({"running", "finalizing"})
_SECRET_VALUE = re.compile(
    r"(?i)\b(api[_ -]?key|token|secret|password|passwd|authorization)\s*([=:])\s*\S+"
)
_URL = re.compile(r"(?i)\b(?:https?|wss?)://\S+")
_EMAIL = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
_ABSOLUTE_PATH = re.compile(r"(?<!\w)(?:~?/|[A-Za-z]:[\\/])[^\s`]*")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d(). -]{7,}\d(?!\w)")


def short_handle(value: Any) -> str:
    """Return a non-routable display handle, never a full registry id."""
    text = str(value or "")
    prefix_match = re.match(r"[A-Za-z]+", text)
    prefix = prefix_match.group(0) if prefix_match else "work"
    # A short registry id can be entirely identifying; do not reveal it just
    # to make a status row look more specific.
    suffix = text[-4:] if len(text) > 8 else ""
    return f"{prefix}…{suffix}" if suffix else prefix


def safe_goal_preview(value: Any, limit: int = 72) -> str:
    """Redact common private material before exposing a short status goal."""
    text = " ".join(str(value or "").split())
    text = _SECRET_VALUE.sub(lambda match: f"{match.group(1)}{match.group(2)}[redacted]", text)
    text = _URL.sub("[link redacted]", text)
    text = _EMAIL.sub("[email redacted]", text)
    text = _ABSOLUTE_PATH.sub("[path redacted]", text)
    text = _PHONE.sub("[number redacted]", text)
    # Keep user-controlled content from changing Telegram Markdown structure.
    text = text.replace("`", "'").replace("*", "").replace("_", "")
    if len(text) > limit:
        text = text[: max(1, limit - 1)].rstrip() + "…"
    return text or "(no safe goal preview)"


def _elapsed(now: float, started_at: Any) -> int:
    try:
        return max(0, int(now - float(started_at)))
    except (TypeError, ValueError):
        return 0


def _snapshot_delegated_agents(records: Iterable[Dict[str, Any]], now: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for record_index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("status") not in _ACTIVE_DELEGATION_STATES:
            continue
        goals = record.get("goals") if record.get("is_batch") else None
        if not isinstance(goals, list) or not goals:
            goals = [record.get("goal")]
        raw_names = record.get("display_names") if record.get("is_batch") else [record.get("display_name")]
        raw_tasks = record.get("display_tasks") if record.get("is_batch") else [record.get("display_task")]
        if not isinstance(raw_names, list):
            raw_names = []
        if not isinstance(raw_tasks, list):
            raw_tasks = []
        from tools.async_delegation import normalize_public_display_metadata
        for child_index, _goal in enumerate(goals):
            identity = (record_index, child_index)
            if identity in seen:
                continue
            seen.add(identity)
            display_name, display_task = normalize_public_display_metadata(
                raw_names[child_index] if child_index < len(raw_names) else None,
                raw_tasks[child_index] if child_index < len(raw_tasks) else None,
                goal=_goal,
                context=record.get("context"),
            )
            rows.append(
                {
                    "status": str(record.get("status") or "running"),
                    "elapsed_seconds": _elapsed(now, record.get("dispatched_at")),
                    "display_name": display_name,
                    "display_task": display_task,
                    "child_index": child_index,
                }
            )
    return sorted(rows, key=lambda row: (-row["elapsed_seconds"], row["display_name"], row["child_index"]))


def _snapshot_processes(processes: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for process in processes:
        if not isinstance(process, dict) or process.get("status") != "running":
            continue
        raw_handle = str(process.get("session_id") or process.get("id") or "process")
        if raw_handle in seen:
            continue
        seen.add(raw_handle)
        try:
            elapsed = max(0, int(process.get("uptime_seconds", 0)))
        except (TypeError, ValueError):
            elapsed = 0
        rows.append({"handle": short_handle(raw_handle), "status": "running", "elapsed_seconds": elapsed})
    return sorted(rows, key=lambda row: (-row["elapsed_seconds"], row["handle"]))


def build_activity_snapshot(runner: Any, now: float | None = None) -> Dict[str, List[Dict[str, Any]]]:
    """Take one canonical, best-effort snapshot without leaking registry payloads."""
    from tools.async_delegation import list_async_delegations
    from tools.process_registry import process_registry

    captured_at = time.time() if now is None else now
    try:
        delegations = list_async_delegations()
    except Exception:
        delegations = []
    try:
        processes = process_registry.list_sessions()
    except Exception:
        processes = []
    try:
        background_tasks = list(getattr(runner, "_background_tasks", set()) or set())
    except Exception:
        background_tasks = []

    async_jobs = []
    for task in background_tasks:
        try:
            active = hasattr(task, "done") and not task.done()
        except Exception:
            active = False
        if active:
            # Do not expose task names, coroutine reprs, or callback metadata.
            async_jobs.append({"category": "gateway task", "status": "running"})

    return {
        "delegated_agents": _snapshot_delegated_agents(delegations, captured_at),
        "processes": _snapshot_processes(processes),
        "async_jobs": async_jobs,
    }
