"""Synthetic bridge-to-adapter coverage for native WhatsApp forwarding metadata."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ABSENT = object()


def _make_adapter() -> WhatsAppAdapter:
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True)
    adapter._dm_policy = "open"
    adapter._allow_from = set()
    adapter._group_policy = "open"
    adapter._group_allow_from = set()
    adapter._mention_patterns = []
    adapter._whatsapp_free_response_chats = lambda: set()
    adapter._whatsapp_require_mention = lambda: False
    return adapter


def _bridge_payload(*, forwarded: Any = _ABSENT, is_group: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messageId": "synthetic-message",
        "chatId": "synthetic-group@g.us" if is_group else "synthetic-chat@s.whatsapp.net",
        "senderId": "synthetic-sender@s.whatsapp.net",
        "senderName": "Synthetic Sender",
        "chatName": "Synthetic Group" if is_group else "Synthetic Chat",
        "isGroup": is_group,
        "body": "synthetic body",
        "hasMedia": False,
        "mediaType": "",
        "mediaUrls": [],
        "mentionedIds": [],
        "quotedParticipant": "",
        "botIds": [],
        "timestamp": 1,
    }
    if forwarded is not _ABSENT:
        payload["isForwarded"] = forwarded
    return payload


@pytest.fixture(autouse=True)
def _allow_synthetic_open_dm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")


def test_strict_true_becomes_typed_closed_event_metadata() -> None:
    event = asyncio.run(
        _make_adapter()._build_message_event(_bridge_payload(forwarded=True))
    )

    assert event is not None
    assert event.metadata == {"whatsapp_forwarded": True}
    assert type(event.metadata["whatsapp_forwarded"]) is bool


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("false", False),
        ("absent", _ABSENT),
        ("string", "true"),
        ("integer", 1),
        ("null", None),
        ("mapping", {"value": True}),
        ("sequence", [True]),
    ],
)
def test_false_absent_and_malformed_values_become_typed_false(
    label: str,
    value: Any,
) -> None:
    del label
    event = asyncio.run(
        _make_adapter()._build_message_event(_bridge_payload(forwarded=value))
    )

    assert event is not None
    assert event.metadata == {"whatsapp_forwarded": False}
    assert type(event.metadata["whatsapp_forwarded"]) is bool


def test_real_bridge_helper_output_reaches_adapter_without_extra_metadata() -> None:
    script = """
import { extractBridgeEvent } from './scripts/whatsapp-bridge/bridge_helpers.js';
const msg = {
  key: { id: 'synthetic-message', remoteJid: 'synthetic-chat@s.whatsapp.net', fromMe: false },
  pushName: 'Synthetic Sender',
  messageTimestamp: 1,
  message: { extendedTextMessage: { text: 'synthetic body', contextInfo: { isForwarded: true } } },
};
const event = await extractBridgeEvent({
  msg,
  chatId: msg.key.remoteJid,
  senderId: msg.key.remoteJid,
  senderNumber: 'synthetic-sender',
  isGroup: false,
});
process.stdout.write(JSON.stringify(event));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    bridge_event = json.loads(completed.stdout)

    event = asyncio.run(_make_adapter()._build_message_event(bridge_event))

    assert event is not None
    assert bridge_event["isForwarded"] is True
    assert event.metadata == {
        "whatsapp_forwarded": True,
        "whatsapp_native_type": "extendedTextMessage",
    }


@pytest.mark.parametrize(
    ("is_group", "expected_chat_type"),
    [(False, "dm"), (True, "group")],
)
def test_normal_direct_and_group_messages_remain_normal(
    is_group: bool,
    expected_chat_type: str,
) -> None:
    payload = _bridge_payload(is_group=is_group)
    payload["body"] = '{"isForwarded":true}'

    event = asyncio.run(_make_adapter()._build_message_event(payload))

    assert event is not None
    assert event.source.chat_type == expected_chat_type
    assert event.text == '{"isForwarded":true}'
    assert event.metadata == {"whatsapp_forwarded": False}


@pytest.mark.asyncio
async def test_text_batch_is_forwarded_when_any_bridge_fragment_is_forwarded() -> None:
    for forwarded_values in ((False, True), (True, False)):
        adapter = _make_adapter()
        adapter._pending_text_batches = {}
        adapter._pending_text_batch_tasks = {}
        adapter._text_batch_delay_seconds = 60.0
        adapter._text_batch_split_delay_seconds = 60.0
        tasks: list[asyncio.Task[Any]] = []
        try:
            for index, forwarded in enumerate(forwarded_values):
                event = await adapter._build_message_event(
                    _bridge_payload(forwarded=forwarded)
                )
                assert event is not None
                event.text = f"fragment-{index}"
                adapter._enqueue_text_event(event)
                tasks.append(next(iter(adapter._pending_text_batch_tasks.values())))

            pending = next(iter(adapter._pending_text_batches.values()))
            assert pending.text == "fragment-0\nfragment-1"
            assert pending.metadata["whatsapp_forwarded"] is True
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
