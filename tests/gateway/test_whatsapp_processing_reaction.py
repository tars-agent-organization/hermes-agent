import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


def _adapter(*, enabled=True):
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(
        enabled=True,
        extra={
            "group_processing_reaction": enabled,
            "mention_patterns": [r"\bTars\b"],
        },
    )
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._running = True
    adapter._bridge_port = 3000
    adapter._http_session = MagicMock()
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._active_processing_reactions = set()
    return adapter


def _event(*, group=True, body="Tars, responde", **overrides):
    raw = {
        "isGroup": group,
        "body": body,
        "chatId": "120363000000000000@g.us" if group else "15550001111@s.whatsapp.net",
        "senderId": "15550001111@s.whatsapp.net",
        "messageId": "message-1",
        "mentionedIds": [],
        "botIds": ["15550009999@s.whatsapp.net"],
        "quotedParticipant": "",
    }
    raw.update(overrides)
    return MessageEvent(
        text=body,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.WHATSAPP,
            chat_id=raw["chatId"],
            chat_type="group" if group else "dm",
            user_id=raw["senderId"],
        ),
        raw_message=raw,
        message_id=raw["messageId"],
    )


@pytest.mark.parametrize(
    "event",
    [
        _event(body="@15550009999 responde", mentionedIds=["15550009999@s.whatsapp.net"]),
        _event(body="Tars, responde"),
        _event(body="respondendo", quotedParticipant="15550009999@s.whatsapp.net"),
        _event(body="/status"),
    ],
)
def test_group_processing_reaction_accepts_only_direct_triggers(event):
    assert _adapter()._should_use_group_processing_reaction(event) is True


def test_group_processing_reaction_covers_an_admitted_group_followup():
    # Admission happens before the processing hook. A follow-up such as
    # "Ta ai?" may be admitted by free-response/conversational routing even
    # without another literal mention, and must receive the same progress ack.
    assert _adapter()._should_use_group_processing_reaction(
        _event(body="Ta ai?")
    ) is True


def test_group_processing_reaction_rejects_dm_and_disabled_config():
    assert _adapter()._should_use_group_processing_reaction(
        _event(group=False, body="Tars, responde")
    ) is False
    assert _adapter(enabled=False)._should_use_group_processing_reaction(
        _event(body="Tars, responde")
    ) is False


@pytest.mark.asyncio
async def test_processing_lifecycle_adds_eyes_then_removes_it():
    adapter = _adapter()
    event = _event()
    adapter._set_processing_reaction = AsyncMock(return_value=True)

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, object())

    assert adapter._set_processing_reaction.await_args_list == [
        ((event, "👀"),),
        ((event, ""),),
    ]
    assert adapter._active_processing_reactions == set()


@pytest.mark.asyncio
async def test_failed_reaction_is_not_removed():
    adapter = _adapter()
    directed = _event()
    adapter._set_processing_reaction = AsyncMock(return_value=False)

    await adapter.on_processing_start(directed)
    await adapter.on_processing_complete(directed, object())

    adapter._set_processing_reaction.assert_awaited_once_with(directed, "👀")
    assert adapter._active_processing_reactions == set()


@pytest.mark.asyncio
async def test_set_processing_reaction_posts_exact_group_message_target():
    adapter = _adapter()
    event = _event()
    response = MagicMock()
    response.status = 200
    context = AsyncMock()
    context.__aenter__.return_value = response
    context.__aexit__.return_value = None
    adapter._http_session.post.return_value = context

    assert await adapter._set_processing_reaction(event, "👀") is True

    _, kwargs = adapter._http_session.post.call_args
    assert kwargs["json"] == {
        "chatId": "120363000000000000@g.us",
        "messageId": "message-1",
        "participant": "15550001111@s.whatsapp.net",
        "emoji": "👀",
    }


@pytest.mark.asyncio
async def test_text_batch_preserves_later_direct_message_as_reaction_target():
    adapter = _adapter()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._flush_text_batch = AsyncMock()
    ordinary = _event(body="conversa entre humanos")
    directed = _event(body="Tars, responde", messageId="message-2")

    adapter._enqueue_text_event(ordinary)
    adapter._enqueue_text_event(directed)
    await asyncio.sleep(0)

    pending = next(iter(adapter._pending_text_batches.values()))
    assert adapter._should_use_group_processing_reaction(pending) is True
    assert adapter._processing_reaction_target(pending) == (
        "120363000000000000@g.us",
        "message-2",
        "15550001111@s.whatsapp.net",
    )
