from pathlib import Path

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.mark.asyncio
async def test_local_sticker_is_routed_as_webp_image(tmp_path: Path):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    home = tmp_path / "hermes"
    image_dir = home / "cache" / "images"
    image_dir.mkdir(parents=True)
    sticker = image_dir / "sticker_test.webp"
    sticker.write_bytes(b"RIFF\x04\x00\x00\x00WEBP")

    token = set_hermes_home_override(str(home))
    try:
        adapter = WhatsAppAdapter(PlatformConfig(
            enabled=True,
            extra={"dm_policy": "open"},
        ))
        adapter._should_process_message = lambda data: True
        event = await adapter._build_message_event({
            "mediaType": "sticker",
            "hasMedia": True,
            "mediaUrls": [str(sticker)],
            "mime": "image/webp",
            "isGroup": False,
            "chatId": "15551234567",
            "senderId": "15551234567",
            "senderName": "Alice",
            "body": "",
            "messageId": "sticker-1",
        })
    finally:
        reset_hermes_home_override(token)

    assert event is not None
    assert event.message_type == MessageType.STICKER
    assert event.media_urls == [str(sticker)]
    assert event.media_types == ["image/webp"]
