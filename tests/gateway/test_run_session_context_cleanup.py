"""Regression tests for request-local gateway session-context cleanup."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource
from gateway.session_context import (
    clear_session_vars,
    get_session_env,
    get_trusted_request_metadata,
    set_session_vars,
)


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.LOCAL,
        chat_id="child-chat",
        chat_type="dm",
        user_id="child-user",
        user_name="Child",
    )


def _event(source: SessionSource) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
    )


def _runner(prepare_message_text) -> GatewayRunner:
    runner: Any = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    runner._turn_leases = None
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_telegram_topic_lane = lambda _source: False
    runner._pinned_session_context_prompt = lambda *_args: "context"
    runner._mark_durable_active_turn = AsyncMock()
    runner._voice_channel_sidecar_note = lambda *_args: None
    runner._prepare_profile_scoped_inbound_message_text = prepare_message_text
    runner.hooks = SimpleNamespace(emit=AsyncMock())

    now = datetime.now(timezone.utc)
    entry = SessionEntry(
        session_key="agent:main:local:dm:child-chat",
        session_id="session-child",
        created_at=now - timedelta(seconds=1),
        updated_at=now,
        platform=Platform.LOCAL,
        chat_type="dm",
    )
    runner._async_session_store = SimpleNamespace(
        _store=object(),
        get_or_create_session=AsyncMock(return_value=entry),
        load_transcript=AsyncMock(
            return_value=[{"role": "assistant", "content": "prior turn"}]
        ),
    )
    runner.session_store = runner._async_session_store._store
    return runner


def _context_snapshot() -> tuple[str, str, str]:
    return (
        get_session_env("HERMES_SESSION_PLATFORM", ""),
        get_session_env("HERMES_SESSION_USER_ID", ""),
        get_session_env("HERMES_SESSION_KEY", ""),
    )


async def _exercise(
    prepare_message_text,
    after: list[tuple[str, str, str]],
    trusted_request_metadata=None,
):
    parent_tokens = set_session_vars(
        platform="discord",
        chat_id="parent-chat",
        chat_type="group",
        user_id="parent-user",
        session_key="agent:main:discord:group:parent-chat",
    )
    source = _source()
    runner = _runner(prepare_message_text)
    try:
        return await runner._handle_message_with_agent(
            _event(source),
            source,
            "agent:main:local:dm:child-chat",
            1,
            trusted_request_metadata,
        )
    finally:
        after.append(_context_snapshot())
        clear_session_vars(parent_tokens)


@pytest.mark.asyncio
async def test_session_context_is_restored_on_pre_agent_early_return():
    observed = []

    async def prepare_message_text(**_kwargs):
        observed.append(_context_snapshot())
        return None

    after = []
    result = await _exercise(prepare_message_text, after)

    assert observed == [
        ("local", "child-user", "agent:main:local:dm:child-chat")
    ]
    assert result is None
    assert after == [("", "", "")]


@pytest.mark.asyncio
async def test_session_context_is_restored_on_pre_agent_exception():
    observed = []

    async def prepare_message_text(**_kwargs):
        observed.append(_context_snapshot())
        raise RuntimeError("pre-agent failure")

    after = []
    with pytest.raises(RuntimeError, match="pre-agent failure"):
        await _exercise(prepare_message_text, after)

    assert observed == [
        ("local", "child-user", "agent:main:local:dm:child-chat")
    ]
    assert after == [("", "", "")]


@pytest.mark.asyncio
async def test_session_context_is_restored_on_pre_agent_cancelled_error():
    observed = []

    async def prepare_message_text(**_kwargs):
        observed.append(_context_snapshot())
        raise asyncio.CancelledError

    after = []
    with pytest.raises(asyncio.CancelledError):
        await _exercise(prepare_message_text, after)

    assert observed == [
        ("local", "child-user", "agent:main:local:dm:child-chat")
    ]
    assert after == [("", "", "")]


@pytest.mark.asyncio
@pytest.mark.parametrize("relayed", [False, True])
@pytest.mark.parametrize("outcome", ["return", "error", "cancel"])
async def test_trusted_metadata_binds_inside_agent_cleanup_boundary(
    relayed,
    outcome,
):
    observed = []

    async def prepare_message_text(**_kwargs):
        observed.append(get_trusted_request_metadata())
        if outcome == "error":
            raise RuntimeError("synthetic failure")
        if outcome == "cancel":
            raise asyncio.CancelledError
        return None

    trusted = {
        "adapter_signal": {"state": "captured"},
        "hermes_relayed": relayed,
        "hermes_ingress_direct": not relayed,
    }
    after = []

    if outcome == "error":
        with pytest.raises(RuntimeError, match="synthetic failure"):
            await _exercise(prepare_message_text, after, trusted)
    elif outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await _exercise(prepare_message_text, after, trusted)
    else:
        assert await _exercise(prepare_message_text, after, trusted) is None

    assert observed == [trusted]
    assert get_trusted_request_metadata() is None
    assert after == [("", "", "")]
