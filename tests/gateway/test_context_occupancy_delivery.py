from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner


class Adapter:
    def __init__(self, *, success=True):
        self.success = success
        self.sent = []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return SendResult(success=self.success)


def runner_for(home_chat_id, adapter=None):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {} if adapter is None else {Platform.WHATSAPP: adapter}
    runner.config = SimpleNamespace(
        get_home_channel=lambda platform: (
            SimpleNamespace(chat_id=home_chat_id) if home_chat_id is not None else None
        )
    )
    return runner


@pytest.mark.asyncio
async def test_private_alert_goes_only_to_whatsapp_home_with_non_conversational_metadata():
    adapter = Adapter()
    runner = runner_for("private@s.whatsapp.net", adapter)
    assert await runner._deliver_private_context_alert("alert") is True
    assert len(adapter.sent) == 1
    chat_id, content, metadata = adapter.sent[0]
    assert (chat_id, content) == ("private@s.whatsapp.net", "alert")
    # WhatsApp's helper leaves metadata unchanged; importantly this path does
    # not append anything to a conversational transcript.
    assert metadata is None


@pytest.mark.asyncio
@pytest.mark.parametrize("home_chat_id", [None, "group@g.us"])
async def test_private_alert_fails_closed_without_private_home(home_chat_id):
    adapter = Adapter()
    runner = runner_for(home_chat_id, adapter)
    assert await runner._deliver_private_context_alert("alert") is False
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_private_alert_fails_closed_without_adapter_or_on_send_failure():
    assert await runner_for("private@s.whatsapp.net")._deliver_private_context_alert("alert") is False
    adapter = Adapter(success=False)
    assert await runner_for("private@s.whatsapp.net", adapter)._deliver_private_context_alert("alert") is False
    assert len(adapter.sent) == 1