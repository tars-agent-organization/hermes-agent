from hermes_state import AsyncSessionDB, SessionDB
"""Tests for gateway /status behavior and token persistence."""

from datetime import datetime
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import (
    AsyncSessionStore,
    SessionEntry,
    SessionSource,
    SessionStore,
    build_session_key,
)


def _make_source(platform: Platform = Platform.TELEGRAM) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str, *, platform: Platform = Platform.TELEGRAM) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(platform),
        message_id="m1",
    )


def _make_runner(session_entry: SessionEntry, *, platform: Platform = Platform.TELEGRAM):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {platform: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = AsyncSessionDB(MagicMock())
    runner._session_db._db.get_session_title.return_value = None
    # Default: no DB row → /status reports 0 tokens.  Tests that exercise
    # the populated path override this.
    runner._session_db._db.get_session.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._agent_cache = {}
    runner._agent_cache_lock = MagicMock()
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env_calls = []

    def _set_session_env(_context, *, trusted_request_metadata=None):
        tokens = []
        runner._set_session_env_calls.append((trusted_request_metadata, tokens))
        return tokens

    runner._set_session_env = _set_session_env
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_status_command_reads_token_totals_from_session_db():
    """Regression test for #17158: /status must source token totals from the
    SQLite SessionDB (where run_agent.py persists them) and sum all component
    counts, not from SessionEntry (which the agent never writes)."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,  # SessionEntry never gets written to — always 0.
    )
    runner = _make_runner(session_entry)
    runner._session_db._db.get_session.return_value = {
        "input_tokens": 1000,
        "output_tokens": 250,
        "cache_read_tokens": 500,
        "cache_write_tokens": 100,
        "reasoning_tokens": 50,
    }

    result = await runner._handle_message(_make_event("/status"))

    # 1000 + 250 + 500 + 100 + 50 = 1,900
    assert "**Lifetime tokens billed:** 1,900" in result


@pytest.mark.asyncio
async def test_status_command_includes_live_agent_model_and_context():
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(session_entry)
    runner._session_db._db.get_session.return_value = {
        "input_tokens": 1000,
        "output_tokens": 250,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "model": "openai/gpt-test",
    }
    running_agent = SimpleNamespace(
        model="openai/gpt-test",
        provider="openai",
        context_compressor=SimpleNamespace(
            last_prompt_tokens=12_345,
            context_length=100_000,
        ),
        interrupt=MagicMock(),
    )
    runner._running_agents[build_session_key(_make_source())] = running_agent

    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `openai/gpt-test` (openai)" in result
    assert "**Context:** 12,345 / 100,000 (12%)" in result
    assert "**Lifetime tokens billed:** 1,250" in result


@pytest.mark.asyncio
async def test_status_command_uses_most_recent_persisted_model_route(tmp_path):
    """Persisted status uses the latest coherent route, not the lifetime-dominant route."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    db = SessionDB(db_path=tmp_path / "state.db")
    runner._session_db = AsyncSessionDB(db)
    try:
        db.create_session("sess-1", "telegram", model="z-ai/glm-5.2")
        db.update_token_counts(
            "sess-1",
            model="z-ai/glm-5.2",
            billing_provider="nvidia",
            billing_base_url="https://integrate.api.nvidia.com/v1/",
            input_tokens=480,
            api_call_count=48,
        )
        db.update_token_counts(
            "sess-1",
            model="upstage/solar-pro4:free",
            billing_provider="nous",
            billing_base_url="https://inference-api.nousresearch.com/v1/",
            input_tokens=60,
            api_call_count=6,
        )
        # Reproduce the inconsistent legacy summary observed in #87227.
        db.update_session_model("sess-1", "z-ai/glm-5.2")
        db.update_session_billing_route(
            "sess-1",
            provider="nous",
            base_url="https://inference-api.nousresearch.com/v1/",
        )

        result = await runner._handle_message(_make_event("/status"))

        assert "**Model:** `upstage/solar-pro4:free` (nous)" in result
        assert "**Model:** `z-ai/glm-5.2` (nvidia)" not in result
    finally:
        db.close()


@pytest.mark.asyncio
async def test_status_command_prefers_rehydrated_session_model_override(tmp_path):
    """A committed /model switch is current before the selected model records usage."""
    source = _make_source()
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    session_entry = store.get_or_create_session(source)
    runner = _make_runner(session_entry)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    db = SessionDB(db_path=tmp_path / "state.db")
    runner._session_db = AsyncSessionDB(db)
    try:
        db.create_session(session_entry.session_id, "telegram", model="model-a")
        db.update_token_counts(
            session_entry.session_id,
            model="model-a",
            billing_provider="provider-a",
            input_tokens=480,
            api_call_count=48,
        )
        result = SimpleNamespace(
            new_model="model-b",
            target_provider="provider-b",
            provider_label="Provider B",
            api_key="secret",
            base_url="https://b.example/v1",
            api_mode=None,
            request_overrides={},
            runtime_capabilities={},
        )
        switch_ctx = SimpleNamespace(
            session_key=session_entry.session_key,
            current_model="model-a",
            persist_global=False,
            restore_snapshot=None,
        )
        await runner._record_model_switch(
            result, switch_ctx, source=source, one_turn=False, picker=False
        )
        # Simulate a restart: /status must lazily recover the durable override.
        runner._session_state(session_entry.session_key).conversation.model_override = None

        status = await runner._handle_message(_make_event("/status"))

        assert "**Model:** `model-b` (provider-b)" in status
        assert "**Model:** `model-a` (provider-a)" not in status
    finally:
        db.close()


def _runner_with_session_override(override: dict, *, last_prompt_tokens: int):
    """Runner whose session has a committed /model override but no resident agent (no compressor)."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    session_entry.last_prompt_tokens = last_prompt_tokens
    runner = _make_runner(session_entry)
    runner._session_state(session_entry.session_key).conversation.model_override = override
    return runner


@pytest.mark.asyncio
async def test_status_command_resolves_window_of_session_only_model_switch():
    """After a session-only /model switch with no turn yet, /status must show used / window for the
    switched-to route (looked up against THAT route's endpoint, and never the default route's
    ``model.context_length`` pin) instead of the occupancy-only line (#111436)."""
    override = {"model": "kimi-k3", "provider": "kimi-coding",
                "base_url": "https://api.kimi.com/coding", "api_key": "sk-override"}
    runner = _runner_with_session_override(override, last_prompt_tokens=79_455)
    config = {"model": {"default": "default-model", "provider": "openrouter", "context_length": 80_000}}

    with patch("gateway.run._load_gateway_config", return_value=config), patch(
        "gateway.run._resolve_runtime_agent_kwargs"
    ) as default_runtime, patch(
        "agent.model_metadata.get_model_context_length", return_value=1_048_576
    ) as lookup:
        result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `kimi-k3` (kimi-coding)" in result
    assert "**Context:** 79,455 / 1,048,576 (8%)" in result
    assert "80,000" not in result
    default_runtime.assert_not_called()
    kwargs = lookup.call_args.kwargs
    assert (kwargs["base_url"], kwargs["api_key"], kwargs["provider"]) == (
        override["base_url"], override["api_key"], "kimi-coding")
    assert kwargs["config_context_length"] is None


@pytest.mark.asyncio
async def test_status_command_keeps_occupancy_only_for_unknown_model_window():
    """A model the resolver cannot size falls to DEFAULT_FALLBACK_CONTEXT; /status must not present
    that invented number as the window (a catalog-listed model of the same size still counts)."""
    from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT

    runner = _runner_with_session_override(
        {"model": "proxy-mystery-model", "provider": "custom:proxy", "base_url": "http://127.0.0.1:1/v1"},
        last_prompt_tokens=4_321,
    )
    with patch("gateway.run._load_gateway_config", return_value={}), patch(
        "agent.model_metadata.get_model_context_length", return_value=DEFAULT_FALLBACK_CONTEXT
    ):
        result = await runner._handle_message(_make_event("/status"))
    assert "**Context:** ~4,321 tokens" in result
    assert f"{DEFAULT_FALLBACK_CONTEXT:,}" not in result

    runner = _runner_with_session_override(
        {"model": "grok-4", "provider": "xai", "base_url": ""}, last_prompt_tokens=4_321)
    with patch("gateway.run._load_gateway_config", return_value={}), patch(
        "agent.model_metadata.get_model_context_length", return_value=DEFAULT_FALLBACK_CONTEXT
    ):
        result = await runner._handle_message(_make_event("/status"))
    assert f"**Context:** 4,321 / {DEFAULT_FALLBACK_CONTEXT:,} (2%)" in result


@pytest.mark.asyncio
async def test_status_command_default_route_keeps_runtime_endpoint_and_context_pin():
    """No /model switch and no resident agent (first /status after a restart): the winner is the
    persisted route or the SessionDB row, which carry no endpoint of their own. The window must then
    be resolved against the default runtime route (custom base_url + key, ``model.context_length``
    pin intact) exactly as /context does, not against an empty endpoint that drops the pin."""
    config = {"model": {"default": "my-local-model", "provider": "custom",
                        "base_url": "http://127.0.0.1:1/v1", "context_length": 32_768}}
    runtime = {"model": "my-local-model", "provider": "custom",
               "base_url": "http://127.0.0.1:1/v1", "api_key": "local-key"}
    route = {"model": "my-local-model", "billing_provider": "custom"}
    for persisted_route, session_row in ((route, None), ({}, dict(route))):
        session_entry = SessionEntry(
            session_key=build_session_key(_make_source()),
            session_id="sess-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )
        session_entry.last_prompt_tokens = 9_000
        runner = _make_runner(session_entry)
        runner._session_db._db.get_session.return_value = session_row
        runner._session_db._db.get_recent_session_model_route.return_value = persisted_route

        with patch("gateway.run._load_gateway_config", return_value=config), patch(
            "gateway.run._resolve_runtime_agent_kwargs", return_value=runtime
        ) as default_runtime, patch(
            "agent.model_metadata.get_model_context_length",
            side_effect=lambda _model, **kw: kw["config_context_length"] or 8_192,
        ) as lookup:
            result = await runner._handle_message(_make_event("/status"))

        assert "**Model:** `my-local-model` (custom)" in result
        assert "**Context:** 9,000 / 32,768 (27%)" in result
        default_runtime.assert_called_once()
        kwargs = lookup.call_args.kwargs
        assert (kwargs["base_url"], kwargs["api_key"], kwargs["config_context_length"]) == (
            runtime["base_url"], runtime["api_key"], 32_768)


@pytest.mark.asyncio
async def test_agents_command_reports_active_agents_and_processes(monkeypatch):
    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(session_entry)
    running_agent = SimpleNamespace(
        session_id="sess-running",
        model="openrouter/test-model",
        interrupt=MagicMock(),
        get_activity_summary=lambda: {"seconds_since_activity": 0},
    )
    runner._running_agents[session_key] = running_agent
    runner._running_agents_ts = {session_key: time.time() - 8}
    runner._background_tasks = set()

    class _FakeRegistry:
        def list_sessions(self):
            return [
                {
                    "session_id": "proc-1",
                    "status": "running",
                    "uptime_seconds": 17,
                    "command": "sleep 30",
                }
            ]

    monkeypatch.setattr("tools.process_registry.process_registry", _FakeRegistry())
    monkeypatch.setattr("tools.async_delegation.active_count", lambda: 0)

    result = await runner._handle_message(_make_event("/agents"))

    assert "**Active delegated agents:**" not in result
    assert "**Running background processes:** 1" in result
    assert "proc-1" not in result
    assert "sleep 30" not in result
    assert "sess-running" not in result
    assert session_key not in result
    running_agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_agents_command_aggregates_delegations_from_other_profile_without_details(monkeypatch):
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(session_entry)
    runner.config.multiplex_profiles = True
    runner._background_tasks = set()

    class _FakeRegistry:
        def list_sessions(self):
            return []

    monkeypatch.setattr("tools.process_registry.process_registry", _FakeRegistry())
    import tools.async_delegation as async_delegation

    monkeypatch.setattr(
        async_delegation,
        "_records",
        {
            "deleg_other_profile": {
                "status": "running",
                "session_key": "agent:other:telegram:dm:private",
            },
            "deleg_other_session": {
                "status": "finalizing",
                "session_key": "agent:main:telegram:dm:elsewhere",
            },
        },
    )

    result = await runner._handle_message(_make_event("/agents"))

    assert "**Active delegated agents:** 2" in result
    assert "**Gateway async jobs:**" not in result
    assert "deleg_" not in result


@pytest.mark.asyncio
async def test_agents_command_counts_local_agent_background_task_and_delegation_once(monkeypatch):
    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(session_entry)
    runner._running_agents[session_key] = SimpleNamespace(session_id="local-session-secret")
    background_task = MagicMock()
    background_task.done.return_value = False
    runner._background_tasks = {background_task}

    class _FakeRegistry:
        def list_sessions(self):
            return []

    monkeypatch.setattr("tools.process_registry.process_registry", _FakeRegistry())
    import tools.async_delegation as async_delegation

    monkeypatch.setattr(
        async_delegation,
        "_records",
        {
            "delegation-secret": {
                "status": "running",
                "session_key": "agent:other:telegram:dm:private",
            },
        },
    )

    result = await runner._handle_message(_make_event("/agents"))

    assert result is not None
    assert "**Active delegated agents:** 1" in result
    assert "**Gateway async jobs:** 1" in result
    assert session_key not in result
    assert "local-session-secret" not in result
    assert "delegation-secret" not in result
    assert "agent:other:telegram:dm:private" not in result


@pytest.mark.asyncio
async def test_agents_command_lists_live_delegated_workers_separately_and_excludes_completed(monkeypatch):
    """A canonical snapshot expands a live batch into workers, never async jobs."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(session_entry)
    runner._background_tasks = {MagicMock(done=lambda: False)}

    class _FakeRegistry:
        def list_sessions(self):
            return [{
                "session_id": "proc_abcdef123456",
                "status": "running",
                "uptime_seconds": 17,
                "command": "curl https://secret.example --token private-token",
            }]

    monkeypatch.setattr("tools.process_registry.process_registry", _FakeRegistry())
    import tools.async_delegation as async_delegation

    now = time.time()
    monkeypatch.setattr(
        async_delegation,
        "_records",
        {
            "deleg_abcdef12": {
                "delegation_id": "deleg_abcdef12",
                "status": "running",
                "goal": "Review deployment token=private-token",
                "display_name": "Fix Triage",
                "display_task": "Map live state",
                "dispatched_at": now - 65,
            },
            "deleg_batch1234": {
                "delegation_id": "deleg_batch1234",
                "status": "running",
                "is_batch": True,
                "goals": ["Check API", "Audit docs"],
                "display_names": ["Security", "Integration"],
                "display_tasks": ["Review API", "Audit documentation"],
                "dispatched_at": now - 17,
            },
            "deleg_completed": {
                "delegation_id": "deleg_completed",
                "status": "completed",
                "goal": "must not appear",
                "dispatched_at": now - 100,
            },
        },
    )

    result = await runner._handle_message(_make_event("/agents"))

    assert "**Active delegated agents:** 3" in result
    assert "Fix Triage · Map live state · running" in result
    assert "Security · Review API" in result
    assert "Integration · Audit documentation" in result
    assert "private-token" not in result
    assert "must not appear" not in result
    assert "**Running background processes:** 1" in result
    assert "`proc…3456` · running" in result
    assert "curl" not in result
    assert "**Gateway async jobs:** 1" in result
    assert "deleg…atch" not in result


@pytest.mark.asyncio
async def test_agents_command_projects_only_public_metadata_for_live_single_batch_and_legacy_records(monkeypatch):
    """The gateway must not derive its public rows from raw delegation data."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()), session_id="sess-1",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm", total_tokens=0,
    )
    runner = _make_runner(session_entry)
    runner._background_tasks = set()

    class _FakeRegistry:
        def list_sessions(self):
            return []

    monkeypatch.setattr("tools.process_registry.process_registry", _FakeRegistry())
    import tools.async_delegation as async_delegation

    raw_secret = "token=private-token https://secret.example /private/path jane@example.com +1 555 555 5555"
    monkeypatch.setattr(async_delegation, "_records", {
        "deleg_single_private_handle": {
            "delegation_id": "deleg_single_private_handle", "status": "running",
            "goal": f"raw goal {raw_secret}", "context": f"raw context {raw_secret}",
            "display_name": "Fix\n*Triage*",
            "display_task": "Map [live](https://evil.example) state\x00 " + "x" * 120,
            "dispatched_at": time.time() - 10,
        },
        "deleg_batch_private_handle": {
            "delegation_id": "deleg_batch_private_handle", "status": "finalizing", "is_batch": True,
            "goals": [f"raw child one {raw_secret}", f"raw child two {raw_secret}"],
            "display_names": ["Security", "Integration"],
            "display_tasks": ["Review privacy boundary", "Check lifecycle"],
            "dispatched_at": time.time() - 20,
        },
        "deleg_legacy_private_handle": {
            "delegation_id": "deleg_legacy_private_handle", "status": "running",
            "goal": f"legacy raw goal {raw_secret}", "context": f"legacy raw context {raw_secret}",
            "dispatched_at": time.time() - 30,
        },
        "deleg_terminal_private_handle": {
            "delegation_id": "deleg_terminal_private_handle", "status": "completed",
            "display_name": "Completed agent", "display_task": "must not appear",
            "goal": f"terminal raw goal {raw_secret}", "dispatched_at": time.time() - 40,
        },
    })

    result = await runner._handle_message(_make_event("/agents"))

    assert "**Active delegated agents:** 4" in result
    assert result.count("Delegated agent · Task in progress") == 2
    assert "Security · Review privacy boundary" in result
    assert "Integration · Check lifecycle" in result
    assert "x" * 97 not in result
    for forbidden in (
        "deleg_", "private-token", "secret.example", "/private/path",
        "jane@example.com", "555 555", "raw goal", "raw context",
        "must not appear", "Completed agent", "[live]", "https://evil.example",
    ):
        assert forbidden not in result


@pytest.mark.asyncio
async def test_agents_command_falls_back_for_adversarial_public_metadata(monkeypatch):
    """The renderer repeats provenance checks for legacy/in-memory records."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()), session_id="sess-1",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm", total_tokens=0,
    )
    runner = _make_runner(session_entry)
    runner._background_tasks = set()
    monkeypatch.setattr(
        "tools.process_registry.process_registry", type("Registry", (), {"list_sessions": lambda _self: []})()
    )
    import tools.async_delegation as async_delegation

    goal = "Review private deployment credentials before production release"
    monkeypatch.setattr(async_delegation, "_records", {
        "copy": {
            "status": "running", "goal": goal, "context": "confidential",
            "display_name": "Security", "display_task": "Review private deployment",
            "dispatched_at": time.time() - 3,
        },
        "syntax": {
            "status": "running", "goal": "Audit release", "context": None,
            "display_name": "Audit\u202e", "display_task": r"MarkdownV2 \\ _ * [ ] ( ) ~ ` > # + - = | { } . !\u200b",
            "dispatched_at": time.time() - 2,
        },
    })

    result = await runner._handle_message(_make_event("/agents"))

    assert "Security · Task in progress · running" in result
    assert result.count("Delegated agent · Task in progress") == 1
    for unsafe in ("Review private deployment", "MarkdownV2", "\u202e", "\u200b"):
        assert unsafe not in result


@pytest.mark.asyncio
async def test_agents_command_keeps_safe_public_siblings_and_hides_rejected_fields(monkeypatch):
    """Legacy snapshot revalidation applies independent fallbacks without raw leakage."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()), session_id="sess-1",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm", total_tokens=0,
    )
    runner = _make_runner(session_entry)
    runner._background_tasks = set()
    monkeypatch.setattr(
        "tools.process_registry.process_registry", type("Registry", (), {"list_sessions": lambda _self: []})()
    )
    import tools.async_delegation as async_delegation

    copied_task = "Revisar credenciais privadas"
    monkeypatch.setattr(async_delegation, "_records", {
        "single": {
            "status": "running", "goal": "Revisar credenciais privadas antes da liberação de produção",
            "context": "confidencial", "display_name": "Análise", "display_task": copied_task,
            "dispatched_at": time.time() - 3,
        },
        "batch": {
            "status": "running", "is_batch": True, "goals": ["Auditar métricas", "Preparar relatório"],
            "display_names": ["Nome\u202e", "Operações"],
            "display_tasks": ["Sintetizar resultados", "token=synthetic-secret"],
            "dispatched_at": time.time() - 2,
        },
    })

    result = await runner._handle_message(_make_event("/agents"))

    assert "Análise · Task in progress · running" in result
    assert "Delegated agent · Sintetizar resultados · running" in result
    assert "Operações · Task in progress · running" in result
    for private in (copied_task, "synthetic-secret", "Nome\u202e"):
        assert private not in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot", "present", "absent"),
    [
        (
            {
                "delegated_agents": [
                    {"display_name": "Docs", "display_task": "Check documentation", "status": "running", "elapsed_seconds": 10}
                ],
                "processes": [],
                "async_jobs": [],
            },
            ["**Active delegated agents:** 1"],
            ["**Running background processes:**", "**Gateway async jobs:**", "- none", "Nenhum agente"],
        ),
        (
            {
                "delegated_agents": [],
                "processes": [{"handle": "proc…1234", "status": "running", "elapsed_seconds": 10}],
                "async_jobs": [],
            },
            ["**Running background processes:** 1"],
            ["**Active delegated agents:**", "**Gateway async jobs:**", "- none", "Nenhum agente"],
        ),
        (
            {
                "delegated_agents": [],
                "processes": [],
                "async_jobs": [{"category": "gateway task", "status": "running"}],
            },
            ["**Gateway async jobs:** 1"],
            ["**Active delegated agents:**", "**Running background processes:**", "- none", "Nenhum agente"],
        ),
        (
            {
                "delegated_agents": [
                    {"display_name": "Docs", "display_task": "Check documentation", "status": "running", "elapsed_seconds": 10}
                ],
                "processes": [{"handle": "proc…1234", "status": "running", "elapsed_seconds": 10}],
                "async_jobs": [{"category": "gateway task", "status": "running"}],
            },
            ["**Active delegated agents:** 1", "**Running background processes:** 1", "**Gateway async jobs:** 1"],
            ["- none", "Nenhum agente"],
        ),
        (
            {"delegated_agents": [], "processes": [], "async_jobs": []},
            ["No active agents or running tasks."],
            ["**Active delegated agents:**", "**Running background processes:**", "**Gateway async jobs:**", "- none"],
        ),
    ],
)
async def test_agents_command_omits_empty_activity_sections(monkeypatch, snapshot, present, absent):
    """Only live activity sources render; an all-idle snapshot has one neutral line."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(session_entry)
    monkeypatch.setattr("gateway.activity_snapshot.build_activity_snapshot", lambda _runner: snapshot)

    result = await runner._handle_message(_make_event("/agents"))

    assert "Active Agents & Tasks" in result
    for value in present:
        assert value in result
    assert [result.index(value) for value in present] == sorted(result.index(value) for value in present)
    for value in absent:
        assert value not in result


@pytest.mark.asyncio
async def test_tasks_alias_routes_to_agents_command(monkeypatch):
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(session_entry)
    runner._background_tasks = set()

    class _FakeRegistry:
        def list_sessions(self):
            return []

    monkeypatch.setattr("tools.process_registry.process_registry", _FakeRegistry())

    agents_result = await runner._handle_message(_make_event("/agents"))
    result = await runner._handle_message(_make_event("/tasks"))

    assert "Active Agents & Tasks" in result
    assert result == agents_result


@pytest.mark.asyncio
async def test_handle_message_forwards_trusted_request_metadata_to_session_env(monkeypatch):
    import gateway.run as gateway_run

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner.session_store.load_transcript.return_value = [{"role": "user", "content": "earlier"}]
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "model": "openai/test-model",
        }
    )
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100000,
    )
    metadata = {"trusted_ingress": "fixture-contract"}
    event = MessageEvent(
        text="hello",
        source=_make_source(),
        message_id="m1",
        metadata=metadata,
    )

    result = await runner._handle_message(event)

    assert result == "ok"
    assert len(runner._set_session_env_calls) == 1
    received_metadata, returned_tokens = runner._set_session_env_calls[0]
    assert received_metadata is metadata
    assert returned_tokens == []


@pytest.mark.asyncio
async def test_first_run_slack_home_channel_onboarding_uses_parent_command(monkeypatch):
    import gateway.run as gateway_run

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source(Platform.SLACK)),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.SLACK,
        chat_type="dm",
    )
    runner = _make_runner(session_entry, platform=Platform.SLACK)
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = False
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "model": "openai/test-model",
        }
    )

    monkeypatch.delenv("SLACK_HOME_CHANNEL", raising=False)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100000,
    )

    result = await runner._handle_message(_make_event("hello", platform=Platform.SLACK))

    assert result == "ok"
    runner.adapters[Platform.SLACK].send.assert_awaited_once()
    onboarding = runner.adapters[Platform.SLACK].send.await_args.args[1]
    assert "/hermes sethome" in onboarding
    assert "Type /sethome" not in onboarding


@pytest.mark.asyncio
async def test_handle_message_stale_result_keeps_newer_generation_callback(monkeypatch):
    import gateway.run as gateway_run

    class _Adapter:
        def __init__(self):
            self._post_delivery_callbacks = {}

        async def send(self, *args, **kwargs):
            return None

        def pop_post_delivery_callback(self, session_key, *, generation=None):
            entry = self._post_delivery_callbacks.get(session_key)
            if entry is None:
                return None
            if isinstance(entry, tuple):
                entry_generation, callback = entry
                if generation is not None and entry_generation != generation:
                    return None
                self._post_delivery_callbacks.pop(session_key, None)
                return callback
            if generation is not None:
                return None
            return self._post_delivery_callbacks.pop(session_key, None)

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner.session_store.load_transcript.return_value = [{"role": "user", "content": "earlier"}]
    session_key = session_entry.session_key
    adapter = _Adapter()
    runner.adapters[Platform.TELEGRAM] = adapter

    async def _stale_result(**kwargs):
        # Simulate a newer run claiming the callback slot before the stale run unwinds.
        runner._session_run_generation[session_key] = 2
        adapter._post_delivery_callbacks[session_key] = (2, lambda: None)
        return {
            "final_response": "late reply",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 80,
            "input_tokens": 120,
            "output_tokens": 45,
            "model": "openai/test-model",
        }

    runner._run_agent = AsyncMock(side_effect=_stale_result)

    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100000,
    )

    result = await runner._handle_message(_make_event("hello"))

    assert result is None
    assert session_key in adapter._post_delivery_callbacks
    assert adapter._post_delivery_callbacks[session_key][0] == 2


@pytest.mark.asyncio
async def test_status_command_bypasses_active_session_guard():
    """When an agent is running, /status must be dispatched immediately via
    base.handle_message — not queued or treated as an interrupt (#5046)."""
    import asyncio
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session import build_session_key
    from gateway.config import Platform, PlatformConfig

    source = _make_source()
    session_key = build_session_key(source)

    handler_called_with = []

    async def fake_handler(event):
        handler_called_with.append(event)
        return "📊 **Hermes Gateway Status**\n**Agent Running:** Yes ⚡"

    # Concrete subclass to avoid abstract method errors
    class _ConcreteAdapter(BasePlatformAdapter):
        platform = Platform.TELEGRAM

        async def connect(self, *, is_reconnect: bool = False): pass
        async def disconnect(self): pass
        async def send(self, chat_id, content, **kwargs): pass
        async def get_chat_info(self, chat_id): return {}

    platform_config = PlatformConfig(enabled=True, token="***")
    adapter = _ConcreteAdapter(platform_config, Platform.TELEGRAM)
    adapter.set_message_handler(fake_handler)

    sent = []

    async def fake_send_with_retry(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)

    adapter._send_with_retry = fake_send_with_retry

    # Simulate an active session
    interrupt_event = asyncio.Event()
    adapter._active_sessions[session_key] = interrupt_event

    event = MessageEvent(
        text="/status",
        source=source,
        message_id="m1",
        message_type=MessageType.COMMAND,
    )
    await adapter.handle_message(event)

    assert handler_called_with, "/status handler was never called (event was queued or dropped)"
    assert sent, "/status response was never sent"
    assert "Agent Running" in sent[0]
    assert not interrupt_event.is_set(), "/status incorrectly triggered an agent interrupt"
    assert session_key not in adapter._pending_messages, "/status was incorrectly queued"


@pytest.mark.asyncio
async def test_profile_command_reports_source_stamped_profile(monkeypatch, tmp_path):
    """On a multiplexed gateway, /profile reports the profile SERVING the
    source (source.profile — URL prefix / per-credential adapter / room map),
    not the multiplexer's active profile, which is always the default and
    made /profile answer "default" in every persona chat."""
    hermes_home = tmp_path / ".hermes"
    profile_home = hermes_home / "profiles" / "milo"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n")  # identity marker: a bare dir is not a profile

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner.config.multiplex_profiles = True
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    event = _make_event("/profile")
    event.source.profile = "milo"

    result = await runner._handle_profile_command(event)

    assert "**Profile:** `milo`" in result
    # The reply renders display_hermes_home() for the routed profile, which abbreviates a home
    # under $HOME to ``~/…``; compare against the same rendering rather than the raw path.
    from gateway.run import _profile_runtime_scope
    from hermes_constants import display_hermes_home

    with _profile_runtime_scope(profile_home):
        expected_home = display_hermes_home()
    assert f"**Home:** `{expected_home}`" in result


# ── /context command tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_context_command_keeps_configured_window_without_resident_agent():
    """The no-agent fallback must not replace a custom-provider context pin."""
    model = "unsloth/Qwen3.8-27B-GGUF:Q8_0"
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-context-pin",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    session_entry.last_prompt_tokens = 66_570
    runner = _make_runner(session_entry)
    runner._session_db._db.get_session.return_value = {"model": model}

    config = {
        "model": {
            "default": model,
            "provider": "custom-local-qwen",
            "context_length": 262_144,
        },
        "custom_providers": [
            {
                "name": "custom-local-qwen",
                "base_url": "http://127.0.0.1:8080/v1",
                "models": {},
            }
        ],
    }
    runtime = {
        "provider": "custom-local-qwen",
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key": "",
    }

    with patch("gateway.run._load_gateway_config", return_value=config), patch(
        "gateway.run._resolve_runtime_agent_kwargs", return_value=runtime
    ), patch(
        "hermes_cli.config.get_compatible_custom_providers",
        return_value=config["custom_providers"],
    ), patch(
        "agent.model_metadata.get_model_context_length",
        side_effect=lambda *args, **kwargs: kwargs.get("config_context_length") or 131_072,
    ) as context_lookup:
        result = await runner._handle_context_command(_make_event("/context"))

    assert "Window: 262,144 tokens" in result
    assert "In use: 66,570 / 262,144 (25%)" in result
    assert "131,072" not in result
    assert context_lookup.call_count == 1
    assert context_lookup.call_args.kwargs["config_context_length"] == 262_144


def _stub_agent(**overrides) -> SimpleNamespace:
    """Build a stub agent with the attributes _handle_context_command reads."""
    props = dict(
        model="openai/gpt-test",
        context_compressor=SimpleNamespace(
            last_prompt_tokens=47_231,
            context_length=200_000,
            threshold_tokens=100_000,
            threshold_percent=0.5,
            compression_count=2,
            _last_compression_savings_pct=63.0,
        ),
        session_api_calls=47,
        session_input_tokens=410_000,
        session_output_tokens=38_000,
        session_reasoning_tokens=12_000,
        session_total_tokens=3_158_641,
        session_cache_read_tokens=2_900_000,
        session_cache_write_tokens=48_000,
    )
    props.update(overrides)
    return SimpleNamespace(**props)


@pytest.mark.asyncio
async def test_context_all_appends_expanded_listings():
    """/context all appends per-toolset and per-skill cost listings."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-6",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    agent = _stub_agent()
    runner._running_agents[session_entry.session_key] = agent

    fake_payload = {
        "categories": [
            {"id": "skills", "label": "Skills", "tokens": 2_000},
        ],
        "context_max": 200_000,
        "context_percent": 24,
        "context_used": 47_231,
        "estimated_total": 2_000,
        "model": "openai/gpt-test",
    }
    fake_details = {
        "skills": [
            {"name": "hermes-agent", "index_tokens": 30, "skill_md_tokens": 2_500},
        ],
        "toolsets": [
            {"toolset": "terminal", "tool_count": 4, "schema_tokens": 5_100},
        ],
    }
    from unittest.mock import patch as _patch
    with _patch(
        "agent.context_breakdown.compute_session_context_breakdown",
        return_value=fake_payload,
    ), _patch(
        "agent.context_breakdown.compute_context_details",
        return_value=fake_details,
    ):
        result = await runner._handle_context_command(_make_event("/context all"))

    assert "Toolsets by schema cost" in result
    assert "terminal" in result and "5,100 tokens" in result
    assert "Skills by cost" in result
    assert "hermes-agent" in result
    # Expanded view drops the hint
    assert "Use /context all" not in result
