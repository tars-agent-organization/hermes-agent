"""Queued requests must not run tools with the opening message's authority."""

import asyncio
from dataclasses import replace

import pytest

from gateway.platforms.base import MessageEvent
from gateway.session_context import (
    clear_session_vars,
    get_session_env,
    get_trusted_request_metadata,
    set_session_vars,
)
from tests.gateway.test_queued_followup_processing_hooks import (
    SESSION_KEY,
    HookRecordingAdapter,
    _install_fake_agent,
    _make_runner,
    _source,
)


@pytest.fixture(autouse=True)
def no_operator_dotenv(monkeypatch):
    monkeypatch.setattr("hermes_cli.env_loader.load_hermes_dotenv", lambda **kwargs: None)


@pytest.mark.asyncio
@pytest.mark.parametrize("sender", ["original-user", "other-user"])
@pytest.mark.parametrize("metadata", [{"document": "queued-upload"}, {}, None])
async def test_inband_queue_binds_new_request_not_prior_authority(monkeypatch, tmp_path, sender, metadata):
    observed = []

    class Agent:
        def __init__(self, **kwargs):
            self.tools = []

        def run_conversation(self, message, **kwargs):
            observed.append((get_session_env("HERMES_SESSION_USER_ID"),
                             get_session_env("HERMES_SESSION_MESSAGE_ID"),
                             dict(get_trusted_request_metadata() or {})))
            return {"final_response": "done", "messages": [], "api_calls": 1}

    _install_fake_agent(monkeypatch, tmp_path, Agent)
    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)
    first = replace(_source(), user_id="original-user", message_id="upload")
    pending = MessageEvent(text="read the document", message_id="request",
                           source=replace(first, user_id=sender, message_id="request"),
                           metadata=metadata or {})
    adapter._pending_messages[SESSION_KEY] = pending
    tokens = set_session_vars(platform="telegram", user_id="original-user", message_id="upload",
                              trusted_request_metadata={"document": "prior-upload"})
    try:
        await runner._run_agent(message="upload", context_prompt="", history=[], source=first,
                                session_id="queued-context", session_key=SESSION_KEY)
        assert len(observed) == 2
        assert observed[0] == ("original-user", "upload", {"document": "prior-upload"})
        assert observed[1][:2] == (sender, "request")
        assert observed[1][2].get("document") == (metadata or {}).get("document")
        assert get_session_env("HERMES_SESSION_USER_ID") == "original-user"
        assert get_session_env("HERMES_SESSION_MESSAGE_ID") == "upload"
        assert dict(get_trusted_request_metadata()) == {"document": "prior-upload"}
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_queue_context_restores_after_failure(monkeypatch, tmp_path, failure):
    class Agent:
        def __init__(self, **kwargs):
            self.tools = []

        def run_conversation(self, message, **kwargs):
            if message == "follow-up":
                assert get_session_env("HERMES_SESSION_MESSAGE_ID") == "request"
                raise failure()
            return {"final_response": "done", "messages": [], "api_calls": 1}

    _install_fake_agent(monkeypatch, tmp_path, Agent)
    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)
    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="follow-up", source=_source(), message_id="request", metadata={"new": True})
    tokens = set_session_vars(message_id="upload", trusted_request_metadata={"old": True})
    try:
        with pytest.raises(failure):
            await runner._run_agent(message="first", context_prompt="", history=[], source=_source(),
                                    session_id="queued-failure", session_key=SESSION_KEY)
        assert get_session_env("HERMES_SESSION_MESSAGE_ID") == "upload"
        assert dict(get_trusted_request_metadata()) == {"old": True}
    finally:
        clear_session_vars(tokens)
