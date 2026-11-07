"""Native runner -> Slack Block Kit -> authenticated click -> real gate.

Only Slack's HTTP transport and identity policy are fakes. No external calls.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.run_turn_runner import TurnRunner
from plugins.platforms.slack.adapter import SlackAdapter
from tools import approval, approval_context


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setattr(approval_context, "_get_approval_config",
                        lambda: {"mode": "manual", "slack_owner_id": "U_OWNER"})
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="fake"))
    adapter._app = Mock()
    adapter._is_interactive_user_authorized = Mock(return_value=True)
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ts": "10.2"})
    adapter._get_client = Mock(return_value=client)
    adapter._dm_target = AsyncMock(side_effect=lambda chat, meta: chat)
    source = SessionSource(platform=Platform.SLACK, chat_id="C_ORIGIN", chat_type="thread",
                           thread_id="10.1", user_id="U_OWNER", scope_id="T_ORIGIN")
    ctx = SimpleNamespace(source=source, _status_adapter=adapter, _status_chat_id="C_ORIGIN",
                          _status_thread_metadata={"thread_id": "10.1", "scope_id": "T_ORIGIN"},
                          session_key="agent:lerian:slack:T_ORIGIN:C_ORIGIN:10.1")
    turn = SimpleNamespace(_ctx=ctx, _runner=SimpleNamespace(_is_user_authorized=Mock(return_value=True)),
                           _close_native_stream_boundary=Mock())
    yield adapter, client, turn
    approval.unregister_gateway_notify(ctx.session_key)


async def start(harness):
    _adapter, _client, turn = harness
    loop = asyncio.get_running_loop()
    ready = asyncio.Event()
    executed = []
    def schedule(coro, label):
        return asyncio.run_coroutine_threadsafe(coro, loop)
    turn._schedule = schedule
    def notify(data):
        try:
            TurnRunner._approval_notify_sync(turn, data)
        finally:
            loop.call_soon_threadsafe(ready.set)
    def run():
        token = approval_context.set_current_session_key(turn._ctx.session_key)
        try:
            approval.register_gateway_notify(turn._ctx.session_key, notify)
            result = approval.check_all_command_guards("rm -rf ./fixture", "local")
            if result["approved"]:
                executed.append("rm -rf ./fixture")
            return result
        finally:
            approval_context.reset_current_session_key(token)
    worker = asyncio.create_task(asyncio.to_thread(run))
    await asyncio.wait_for(ready.wait(), 5)
    return worker, executed


def click_payload(client, *, user="U_OWNER", channel="C_ORIGIN", thread="10.1", team="T_ORIGIN", choice="once"):
    blocks = client.chat_postMessage.call_args.kwargs["blocks"]
    button = next(b for block in blocks if block["type"] == "actions"
                  for b in block["elements"] if b["action_id"] == ("hermes_deny" if choice == "deny" else "hermes_approve_once"))
    body = {"team": {"id": team}, "user": {"id": user, "name": user}, "channel": {"id": channel},
            "message": {"ts": "10.2", "thread_ts": thread, "blocks": blocks}}
    return body, button


@pytest.mark.asyncio
@pytest.mark.parametrize("choice, expected", [("once", 1), ("deny", 0)])
async def test_delivery_single_execution_and_duplicate(harness, choice, expected):
    adapter, client, turn = harness
    worker, executed = await start(harness)
    try:
        posted = client.chat_postMessage.call_args.kwargs
        assert posted["channel"] == "C_ORIGIN" and posted["thread_ts"] == "10.1"
        assert "rm -rf ./fixture" in str(posted["blocks"])
        assert "U_OWNER" in str(posted["blocks"]) and "this invocation only" in str(posted["blocks"])
        body, button = click_payload(client, choice=choice)
        await adapter._handle_approval_action(AsyncMock(), body, button)
        outcome = await asyncio.wait_for(worker, 5)
        assert outcome["approved"] is bool(expected)
        await adapter._handle_approval_action(AsyncMock(), body, button)
        assert len(executed) == expected
        assert not approval.has_blocking_approval(turn._ctx.session_key)
    finally:
        approval.unregister_gateway_notify(turn._ctx.session_key)
        await asyncio.wait_for(worker, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", [{"user": "U_OTHER"}, {"channel": "C_OTHER"},
                                     {"thread": "99.1"}, {"team": "T_OTHER"}])
async def test_other_origin_cannot_consume_card(harness, mismatch):
    adapter, client, turn = harness
    worker, executed = await start(harness)
    try:
        body, button = click_payload(client, **mismatch)
        await adapter._handle_approval_action(AsyncMock(), body, button)
        assert approval.has_blocking_approval(turn._ctx.session_key)
        assert not worker.done() and not executed
        body, button = click_payload(client, choice="deny")
        await adapter._handle_approval_action(AsyncMock(), body, button)
        assert not (await asyncio.wait_for(worker, 5))["approved"]
    finally:
        approval.unregister_gateway_notify(turn._ctx.session_key)
        await asyncio.wait_for(worker, 5)


@pytest.mark.asyncio
async def test_failed_delivery_has_no_pending_request_or_text_fallback(harness):
    _, client, turn = harness
    client.chat_postMessage.side_effect = RuntimeError("transport unavailable")
    worker, executed = await start(harness)
    result = await asyncio.wait_for(worker, 5)
    assert not result["approved"] and result["outcome"] == "notify_failed"
    assert client.chat_postMessage.call_count == 1
    assert not executed and not approval.has_blocking_approval(turn._ctx.session_key)


@pytest.mark.asyncio
async def test_evicted_binding_never_falls_back_to_fifo(harness):
    adapter, client, turn = harness
    worker, executed = await start(harness)
    adapter._approval_bindings.clear()
    body, button = click_payload(client)
    await adapter._handle_approval_action(AsyncMock(), body, button)
    assert not executed and not worker.done()
    approval.unregister_gateway_notify(turn._ctx.session_key)
    assert not (await asyncio.wait_for(worker, 5))["approved"]


@pytest.mark.asyncio
async def test_revoked_authorization_cannot_execute(harness):
    adapter, client, turn = harness
    worker, executed = await start(harness)
    turn._runner._is_user_authorized.return_value = False
    body, button = click_payload(client)
    await adapter._handle_approval_action(AsyncMock(), body, button)
    assert not executed and not worker.done()
    approval.unregister_gateway_notify(turn._ctx.session_key)
    assert not (await asyncio.wait_for(worker, 5))["approved"]


@pytest.mark.asyncio
async def test_ambiguous_delivery_is_not_active(harness, monkeypatch):
    monkeypatch.setattr("gateway.run._approval_send_outcome", lambda *a, **kw: "ambiguous")
    _, _, turn = harness
    worker, executed = await start(harness)
    result = await asyncio.wait_for(worker, 5)
    assert not result["approved"] and result["outcome"] == "notify_failed"
    assert not executed and not approval.has_blocking_approval(turn._ctx.session_key)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["thread_id", "scope_id", "user_id", "chat_id"])
async def test_source_mismatch_refuses_delivery(harness, field):
    _, client, turn = harness
    setattr(turn._ctx.source, field, "different")
    worker, executed = await start(harness)
    assert not (await asyncio.wait_for(worker, 5))["approved"]
    client.chat_postMessage.assert_not_called()
    assert not executed


@pytest.mark.asyncio
async def test_typed_approve_explains_exact_card_instead_of_fifo(harness):
    from gateway.slash_commands import GatewaySlashCommandsMixin
    adapter, client, turn = harness
    worker, executed = await start(harness)
    slash = SimpleNamespace(_session_key_for_source=lambda source: turn._ctx.session_key)
    _, reply = GatewaySlashCommandsMixin._blocking_approval_or_stale(
        slash, SimpleNamespace(source=turn._ctx.source), "stale", "none")
    assert "specific command card" in reply
    assert not executed
    body, button = click_payload(client, choice="deny")
    await adapter._handle_approval_action(AsyncMock(), body, button)
    assert not (await asyncio.wait_for(worker, 5))["approved"]


@pytest.mark.asyncio
async def test_command_policy_revoked_during_wait(harness, monkeypatch):
    adapter, client, _turn = harness
    worker, executed = await start(harness)
    monkeypatch.setattr(approval_context, "_get_approval_config",
                        lambda: {"mode": "manual", "slack_owner_id": "U_OWNER", "deny": ["rm *"]})
    body, button = click_payload(client)
    await adapter._handle_approval_action(AsyncMock(), body, button)
    assert not (await asyncio.wait_for(worker, 5))["approved"]
    assert not executed
