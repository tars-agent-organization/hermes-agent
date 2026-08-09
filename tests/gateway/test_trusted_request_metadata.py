"""Request-local trusted gateway metadata contract."""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Mapping

import pytest

import gateway.session_context as session_context


def test_trusted_metadata_api_binds_and_clears_without_environment_fallback(monkeypatch):
    getter = getattr(session_context, "get_trusted_request_metadata", None)
    assert callable(getter), "public trusted-metadata getter is missing"

    session_context.reset_session_vars()
    monkeypatch.setenv("HERMES_TRUSTED_REQUEST_METADATA", "spoofed")
    assert getter() is None

    tokens = session_context.set_session_vars(
        platform="local",
        trusted_request_metadata={"origin": "adapter"},
    )
    assert getter() == {"origin": "adapter"}
    assert "HERMES_TRUSTED_REQUEST_METADATA" not in session_context._VAR_MAP
    assert os.environ["HERMES_TRUSTED_REQUEST_METADATA"] == "spoofed"

    session_context.clear_session_vars(tokens)
    assert getter() is None

    session_context.set_session_vars(trusted_request_metadata={"turn": 2})
    session_context.reset_session_vars()
    assert getter() is None


def test_trusted_metadata_is_a_deep_immutable_snapshot():
    original = {
        "nested": {"items": [{"value": "before"}]},
        "labels": {"alpha", "beta"},
        "coords": [1, 2],
    }

    session_context.set_session_vars(trusted_request_metadata=original)
    snapshot = session_context.get_trusted_request_metadata()

    original["nested"]["items"][0]["value"] = "after"
    original["labels"].add("gamma")
    original["coords"].append(3)

    assert isinstance(snapshot, Mapping)
    assert snapshot["nested"]["items"][0]["value"] == "before"
    assert snapshot["labels"] == frozenset({"alpha", "beta"})
    assert snapshot["coords"] == (1, 2)

    with pytest.raises(TypeError):
        snapshot["new"] = "blocked"
    with pytest.raises(TypeError):
        snapshot["nested"]["new"] = "blocked"


def test_trusted_metadata_cycles_and_incompatible_values_fail_closed():
    cyclic_mapping = {}
    cyclic_mapping["self"] = cyclic_mapping
    cyclic_list = []
    cyclic_list.append(cyclic_list)

    invalid_values = (
        cyclic_mapping,
        {"items": cyclic_list},
        {"unsupported": object()},
    )
    for invalid in invalid_values:
        session_context.set_session_vars(trusted_request_metadata={"valid": True})
        assert session_context.get_trusted_request_metadata() is not None

        session_context.set_session_vars(trusted_request_metadata=invalid)
        assert session_context.get_trusted_request_metadata() is None


@pytest.mark.asyncio
async def test_trusted_metadata_is_isolated_between_concurrent_tasks():
    ready = [asyncio.Event(), asyncio.Event()]

    async def worker(index: int, label: str):
        tokens = session_context.set_session_vars(
            trusted_request_metadata={"label": label}
        )
        ready[index].set()
        await ready[1 - index].wait()
        observed = session_context.get_trusted_request_metadata()
        session_context.clear_session_vars(tokens)
        return observed

    first, second = await asyncio.gather(
        worker(0, "first"),
        worker(1, "second"),
    )

    assert first == {"label": "first"}
    assert second == {"label": "second"}
    assert session_context.get_trusted_request_metadata() is None


@pytest.mark.asyncio
async def test_gateway_executor_copy_context_preserves_trusted_metadata():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._executor_closing = False
    tokens = session_context.set_session_vars(
        trusted_request_metadata={"executor": ["copied"]}
    )
    try:
        observed = await runner._run_in_executor_with_context(
            session_context.get_trusted_request_metadata
        )
    finally:
        runner._shutdown_executor()
        session_context.clear_session_vars(tokens)

    assert observed == {"executor": ("copied",)}
    assert session_context.get_trusted_request_metadata() is None


def test_optional_metadata_parameter_is_final_for_strict_caller_compatibility():
    parameters = list(inspect.signature(session_context.set_session_vars).parameters.values())

    assert parameters[-1].name == "trusted_request_metadata"
    assert parameters[-1].default is None


def test_synthetic_plugin_can_consume_the_public_metadata_contract():
    def synthetic_plugin_probe():
        from gateway.session_context import get_trusted_request_metadata

        metadata = get_trusted_request_metadata() or {}
        return metadata.get("adapter_signal")

    tokens = session_context.set_session_vars(
        trusted_request_metadata={"adapter_signal": {"trusted": True}}
    )
    try:
        assert synthetic_plugin_probe() == {"trusted": True}
    finally:
        session_context.clear_session_vars(tokens)
