"""Durable pre-model rotation for a changed contact-memory projection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import AsyncSessionStore, SessionSource, SessionStore


def _source(sender: str, *, chat_id: str = "family@g.us") -> SessionSource:
    return SessionSource(
        platform=Platform.WHATSAPP,
        chat_id=chat_id,
        user_id=sender,
        user_name=sender,
        chat_type="group",
    )


def _event(text: str, sender: str = "rosane") -> MessageEvent:
    return MessageEvent(text=text, message_id=f"m-{sender}", source=_source(sender))


def _runner(store: SessionStore) -> tuple[object, SimpleNamespace]:
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)},
    )
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._running_agents = {}
    runner._update_prompt_pending = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._last_resolved_model = {}
    runner._evict_cached_agent = MagicMock()
    return runner, adapter


def test_detach_is_durable_exact_lane_and_preserves_transcript(monkeypatch, tmp_path):
    """The old group-sender lineage cannot be recovered from DB or JSON mirror."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    sessions_dir = tmp_path / "sessions"
    store = SessionStore(sessions_dir=sessions_dir, config=GatewayConfig())
    rosane = _source("rosane")
    jeff = _source("jeff")
    old = store.get_or_create_session(rosane)
    unaffected = store.get_or_create_session(jeff)
    store.append_to_transcript(old.session_id, {"role": "user", "content": "pedido da Maju"})
    store.append_to_transcript(old.session_id, {"role": "assistant", "content": "negação antiga"})

    assert store.detach_session(old.session_key, reason="identity_projection_rotation") is True
    assert store.peek_session_id(old.session_key) is None
    assert store._db.get_session(old.session_id)["end_reason"] == "identity_projection_rotation"
    assert [m["content"] for m in store.load_transcript(old.session_id)] == [
        "pedido da Maju", "negação antiga",
    ]
    assert store.peek_session_id(unaffected.session_key) == unaffected.session_id

    # A restarted store must not re-import the legacy mirror or recover this boundary.
    restarted = SessionStore(sessions_dir=sessions_dir, config=GatewayConfig())
    fresh = restarted.get_or_create_session(rosane)
    assert fresh.session_id != old.session_id
    assert restarted.load_transcript(fresh.session_id) == []
    assert restarted.peek_session_id(unaffected.session_key) == unaffected.session_id


@pytest.mark.asyncio
async def test_changed_grant_rotates_before_model_commits_after_detach_and_keeps_other_sender(monkeypatch, tmp_path):
    """Real pre_gateway dispatch detaches Rosane's bound group lane before model input."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    rosane = _source("rosane")
    jeff = _source("jeff")
    old = store.get_or_create_session(rosane)
    jeff_entry = store.get_or_create_session(jeff)
    store.append_to_transcript(old.session_id, {"role": "user", "content": "pedido da Maju"})
    store.append_to_transcript(old.session_id, {"role": "assistant", "content": "negação antiga"})
    runner, adapter = _runner(store)

    state = {"pending": True, "commits": 0, "hook_calls": 0}

    def commit() -> None:
        assert state["pending"] is True
        state["pending"] = False
        state["commits"] += 1

    async def hook(name, **_kwargs):
        assert name == "pre_gateway_dispatch"
        state["hook_calls"] += 1
        if state["pending"]:
            return [{
                "action": "invalidate_identity_projection_session",
                "platform": "whatsapp",
                "scope": "sender_group",
                "commit": commit,
            }]
        return [{"action": "allow"}]

    monkeypatch.setattr("hermes_cli.plugins.ainvoke_hook", hook)
    seen: dict[str, object] = {}

    async def pre_model(event, source, session_key, _generation):
        # This is the pre-model boundary: a replacement pointer must not exist yet.
        seen.setdefault("detached_before_model", store.peek_session_id(session_key) is None)
        entry = store.get_or_create_session(source)
        history = store.load_transcript(entry.session_id)
        seen["session_id"] = entry.session_id
        seen["history"] = history
        if event.text == "e a Maju?":
            return "Precisa esclarecer a qual Maju você se refere."
        seen["tool_calls"] = ["contact_memory_lookup"]
        return "Consulta atualizada."

    runner._handle_message_with_agent = pre_model  # noqa: SLF001
    assert await runner._handle_message(_event("e a Maju?")) == "Precisa esclarecer a qual Maju você se refere."

    assert seen["detached_before_model"] is True
    assert seen["session_id"] != old.session_id
    assert seen["history"] == []
    assert state == {"pending": False, "commits": 1, "hook_calls": 1}
    assert store._db.get_session(old.session_id)["end_reason"] == "identity_projection_rotation"
    assert [m["content"] for m in store.load_transcript(old.session_id)][-1] == "negação antiga"
    assert store.peek_session_id(jeff_entry.session_key) == jeff_entry.session_id
    runner._evict_cached_agent.assert_called_once()
    adapter.send.assert_not_awaited()

    # Same revision keeps the fresh lineage; an explicit next request uses current lookup.
    fresh_id = seen["session_id"]
    assert await runner._handle_message(_event("qual é o exame atual?")) == "Consulta atualizada."
    assert seen["session_id"] == fresh_id
    assert seen["tool_calls"] == ["contact_memory_lookup"]
    assert state["hook_calls"] == 2
    runner._evict_cached_agent.assert_called_once()


@pytest.mark.asyncio
async def test_pending_revision_is_not_committed_or_dispatched_when_detach_fails(monkeypatch, tmp_path):
    """A partial durable detach fails closed, retaining the plugin's pending revision."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    store.get_or_create_session(_source("rosane"))
    runner, adapter = _runner(store)
    state = {"pending": True, "commits": 0}

    async def hook(_name, **_kwargs):
        return [{
            "action": "invalidate_identity_projection_session",
            "platform": "whatsapp",
            "scope": "sender_group",
            "commit": lambda: state.update(pending=False, commits=state["commits"] + 1),
        }]

    monkeypatch.setattr("hermes_cli.plugins.ainvoke_hook", hook)

    def fail_routing_persist() -> None:
        raise OSError("simulated mirror write failure")

    # Exercise the real partial-detach path: the canonical row is ended, but
    # the active routing mirror cannot be durably removed, so dispatch stops.
    monkeypatch.setattr(store, "_save", fail_routing_persist)
    runner._handle_message_with_agent = AsyncMock(return_value="must not run")  # noqa: SLF001

    assert await runner._handle_message(_event("consulta")) is None
    assert state == {"pending": True, "commits": 0}
    runner._handle_message_with_agent.assert_not_awaited()
    adapter.send.assert_not_awaited()
