"""Private, per-response provenance accounting for gateway observability.

This module deliberately records only fixed source categories, validated tool
identifiers, and aggregate counts. It never retains arguments, outputs, query
text, identity values, paths, or opaque revisions.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import re
import threading
from typing import Iterator

_SOURCE_ORDER = (
    "obsidian_shared_projection",
    "persistent_memory",
    "session_history",
    "direct_user_input",
    "web",
    "file",
    "terminal",
    "other_tool",
)
_ALLOWED_SOURCES = frozenset(_SOURCE_ORDER)
_SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_PLUGIN_SOURCE = re.compile(r"^[a-z][a-z_]{0,63}$")
_MAX_DOCUMENT_COUNT = 100_000
_ALLOWED_POLICY_SCOPES = frozenset({
    "explicitly_shareable_only",
    "authorized_document_delivery",
    "read_only",
})

# Tool-to-source mappings are intentionally narrow. A plugin may register a
# source for its own tool at startup; unknown tools are coarsened to other_tool.
_TOOL_SOURCES: dict[str, str] = {
    "memory": "persistent_memory",
    "session_search": "session_history",
    "terminal": "terminal",
    "execute_code": "terminal",
    "web_search": "web",
    "web_extract": "web",
    "browser_navigate": "web",
    "browser_snapshot": "web",
    "browser_click": "web",
    "browser_type": "web",
    "browser_press": "web",
    "browser_scroll": "web",
    "browser_console": "web",
    "browser_get_images": "web",
    "browser_vision": "web",
    "read_file": "file",
    "write_file": "file",
    "patch": "file",
    "search_files": "file",
}
_TOOL_SOURCE_LOCK = threading.Lock()
_REGISTERED_PLUGIN_SOURCES: set[str] = set()
_DEFERRED_TOOL_SOURCES: set[str] = set()

_current_response_provenance: ContextVar[ResponseProvenance | None] = ContextVar(
    "current_response_provenance", default=None
)


def _safe_tool_name(tool_name: object) -> str:
    candidate = str(tool_name or "")
    return candidate if _SAFE_IDENTIFIER.fullmatch(candidate) else "other_tool"


def _safe_source(source: object) -> str:
    candidate = str(source or "")
    if candidate in _ALLOWED_SOURCES:
        return candidate
    with _TOOL_SOURCE_LOCK:
        return candidate if candidate in _REGISTERED_PLUGIN_SOURCES else "other_tool"


def _ordered(values: set[str]) -> list[str]:
    order = {name: index for index, name in enumerate(_SOURCE_ORDER)}
    return sorted(values, key=lambda value: (order.get(value, len(order)), value))


def register_tool_source(
    tool_name: object,
    source: object,
    *,
    defer_until_success: bool = False,
) -> bool:
    """Register a fixed provenance category for a plugin-provided tool.

    This is startup-time registration only. Invalid/unrecognized values are
    rejected rather than copied into a log event.
    """
    safe_tool = _safe_tool_name(tool_name)
    candidate_source = str(source or "")
    if safe_tool == "other_tool" or not _SAFE_PLUGIN_SOURCE.fullmatch(candidate_source):
        return False
    with _TOOL_SOURCE_LOCK:
        _REGISTERED_PLUGIN_SOURCES.add(candidate_source)
        _TOOL_SOURCES[safe_tool] = candidate_source
        if defer_until_success:
            _DEFERRED_TOOL_SOURCES.add(safe_tool)
        else:
            _DEFERRED_TOOL_SOURCES.discard(safe_tool)
    return True


@dataclass
class ResponseProvenance:
    """Session-local aggregate provenance with no user/model-visible payload."""

    available_context: set[str] = field(default_factory=set)
    consulted_sources: set[str] = field(default_factory=set)
    tools: set[str] = field(default_factory=set)
    document_counts: dict[str, int] = field(default_factory=dict)
    policy_scopes: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.available_context = {
            _safe_source(value)
            for value in self.available_context
            if _safe_source(value) != "other_tool"
        }

    def record_tool_call(self, tool_name: object, *, force: bool = False) -> None:
        safe_tool = _safe_tool_name(tool_name)
        with _TOOL_SOURCE_LOCK:
            source = _TOOL_SOURCES.get(safe_tool, "other_tool")
            deferred = safe_tool in _DEFERRED_TOOL_SOURCES
        if deferred and not force:
            return
        self.tools.add(safe_tool if source != "other_tool" else "other_tool")
        self.consulted_sources.add(source)

    def record_source_metadata(
        self,
        source: object,
        *,
        document_count: object | None = None,
        policy_scope: object | None = None,
    ) -> None:
        safe_source = _safe_source(source)
        self.consulted_sources.add(safe_source)
        if isinstance(document_count, int) and not isinstance(document_count, bool):
            if 0 <= document_count <= _MAX_DOCUMENT_COUNT:
                self.document_counts[safe_source] = (
                    self.document_counts.get(safe_source, 0) + document_count
                )
        if isinstance(policy_scope, str) and policy_scope in _ALLOWED_POLICY_SCOPES:
            self.policy_scopes.add(policy_scope)

    def as_log_fields(self) -> dict[str, object]:
        return {
            "consulted_sources": _ordered(self.consulted_sources),
            "tools": sorted(self.tools),
            "available_context": _ordered(self.available_context),
            "document_counts": {
                source: self.document_counts[source]
                for source in _ordered(set(self.document_counts))
            },
            "policy_scopes": sorted(self.policy_scopes),
        }

    def format_log_line(self) -> str:
        consulted_sources = _ordered(self.consulted_sources)
        tools = sorted(self.tools)
        available_context = _ordered(self.available_context)
        document_counts = ",".join(
            f"{source}:{self.document_counts[source]}"
            for source in _ordered(set(self.document_counts))
        )
        policy_scopes = sorted(self.policy_scopes)
        return (
            "response provenance: "
            f"consulted_sources=[{','.join(consulted_sources)}] "
            f"tools=[{','.join(tools)}] "
            f"available_context=[{','.join(available_context)}] "
            f"document_counts={{{document_counts}}} "
            f"policy_scopes=[{','.join(policy_scopes)}]"
        )


def current_response_provenance() -> ResponseProvenance | None:
    return _current_response_provenance.get()


def record_tool_call(tool_name: object) -> None:
    provenance = current_response_provenance()
    if provenance is not None:
        provenance.record_tool_call(tool_name)


def record_successful_tool_call(
    tool_name: object,
    *,
    provenance: ResponseProvenance | None = None,
) -> None:
    """Record a deferred tool only after its handler confirms safe submission."""
    collector = provenance if provenance is not None else current_response_provenance()
    if collector is not None:
        collector.record_tool_call(tool_name, force=True)


def record_source_metadata(
    source: object,
    *,
    document_count: object | None = None,
    policy_scope: object | None = None,
) -> None:
    provenance = current_response_provenance()
    if provenance is not None:
        provenance.record_source_metadata(
            source, document_count=document_count, policy_scope=policy_scope
        )


@contextmanager
def response_provenance_context(
    *, available_context: set[str] | None = None,
) -> Iterator[ResponseProvenance]:
    provenance = ResponseProvenance(available_context=set(available_context or ()))
    token = _current_response_provenance.set(provenance)
    try:
        yield provenance
    finally:
        _current_response_provenance.reset(token)
