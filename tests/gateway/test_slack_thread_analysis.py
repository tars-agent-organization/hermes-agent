import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_slack_mock():
    if "slack_bolt" in sys.modules:
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    sys.modules["slack_bolt"] = slack_bolt
    sys.modules["slack_bolt.async_app"] = slack_bolt.async_app
    handler_mod = MagicMock()
    handler_mod.AsyncSocketModeHandler = MagicMock
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode.async_handler"] = handler_mod
    sdk_mod = MagicMock()
    sdk_mod.web = MagicMock()
    sdk_mod.web.async_client = MagicMock()
    sdk_mod.web.async_client.AsyncWebClient = MagicMock
    sys.modules["slack_sdk"] = sdk_mod
    sys.modules["slack_sdk.web"] = sdk_mod.web
    sys.modules["slack_sdk.web.async_client"] = sdk_mod.web.async_client


_ensure_slack_mock()

from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


def _adapter():
    adapter = SlackAdapter(
        PlatformConfig(
            enabled=True,
            token="xoxb-test-token",
            extra={"require_mention": True, "strict_mention": False},
        )
    )
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._bot_user_id = "U_BOT"
    adapter._team_bot_user_ids = {"T1": "U_BOT"}
    adapter._team_clients = {"T1": adapter._app.client}
    adapter._channel_team = {"C1": "T1", "D1": "T1"}
    adapter._running = True
    adapter.handle_message = AsyncMock()
    adapter._resolve_user_name = AsyncMock(return_value="Jeff")
    adapter._fetch_thread_parent_text = AsyncMock(return_value="")
    return adapter


def _thread_event(text="continuação", *, ts="1000.2"):
    return {
        "text": text,
        "user": "U_OWNER",
        "channel": "C1",
        "channel_type": "channel",
        "team": "T1",
        "ts": ts,
        "thread_ts": "1000.0",
    }


@pytest.mark.parametrize(
    "text",
    [
        "<@U_BOT> analisa esta thread",
        "TARS, resume o que foi discutido nessa thread",
        "<@U_BOT> me atualiza sobre essa discussão",
        "<@U_BOT> summarize this thread",
    ],
)
def test_detects_explicit_thread_analysis_requests(text):
    assert _adapter()._requests_thread_analysis(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "analisa este arquivo",
        "qual é o status?",
        "conversa entre humanos",
    ],
)
def test_does_not_treat_regular_requests_as_thread_analysis(text):
    assert _adapter()._requests_thread_analysis(text) is False


@pytest.mark.asyncio
async def test_processed_thread_reply_gets_reaction_without_new_mention():
    adapter = _adapter()
    adapter._has_active_session_for_thread = MagicMock(return_value=True)

    await adapter._handle_slack_message(_thread_event())

    assert ("T1", "1000.2") in adapter._reacting_message_ids
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_analysis_request_refreshes_thread_even_with_active_session():
    adapter = _adapter()
    adapter._has_active_session_for_thread = MagicMock(return_value=True)
    adapter._fetch_thread_context = AsyncMock(
        return_value="[Thread context]\nAlice: decisão tomada\n[End]\n\n"
    )

    await adapter._handle_slack_message(
        _thread_event("<@U_BOT> analisa esta thread")
    )

    adapter._fetch_thread_context.assert_awaited_once_with(
        channel_id="C1",
        thread_ts="1000.0",
        current_ts="1000.2",
        team_id="T1",
        limit=200,
        refresh=True,
    )
    message = adapter.handle_message.await_args.args[0]
    assert "Alice: decisão tomada" in message.channel_context
    assert "analisa esta thread" in message.text


@pytest.mark.asyncio
async def test_regular_reply_with_active_session_does_not_refetch_thread():
    adapter = _adapter()
    adapter._has_active_session_for_thread = MagicMock(return_value=True)
    adapter._fetch_thread_context = AsyncMock(return_value="unexpected")

    await adapter._handle_slack_message(_thread_event("continuação normal"))

    adapter._fetch_thread_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_bypasses_short_thread_cache_and_requests_wider_context():
    adapter = _adapter()
    adapter._thread_context_cache["C1:1000.0:T1"] = MagicMock(
        content="stale short context",
        fetched_at=10**12,
    )
    adapter._app.client.conversations_replies = AsyncMock(
        return_value={
            "messages": [
                {"ts": "1000.0", "user": "U1", "text": "fresh parent"},
                {"ts": "1000.2", "user": "U_OWNER", "text": "current"},
            ]
        }
    )
    adapter._user_name_cache = {"U1": "Alice"}

    context = await adapter._fetch_thread_context(
        channel_id="C1",
        thread_ts="1000.0",
        current_ts="1000.2",
        team_id="T1",
        limit=200,
        refresh=True,
    )

    assert "fresh parent" in context
    assert "stale short context" not in context
    adapter._app.client.conversations_replies.assert_awaited_once_with(
        channel="C1",
        ts="1000.0",
        limit=200,
        inclusive=True,
    )
