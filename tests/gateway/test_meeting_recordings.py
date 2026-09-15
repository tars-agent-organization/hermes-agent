"""Behavioral tests for the private meeting transcript inbox."""

import json
import stat

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.meeting_recordings import MeetingRecordingStore
from gateway.platforms.api_server import APIServerAdapter


RECORDING_ID = "meeting-12345678"


def _payload(**changes):
    payload = {
        "recording_id": RECORDING_ID,
        "transcript": "Jefferson: Vamos publicar na sexta-feira.",
        "source": "in_person",
        "context": "professional",
        "delivery": "slack_dm",
        "started_at": "2026-09-15T10:00:00Z",
        "ended_at": "2026-09-15T10:30:00Z",
        "title": "Reunião de produto",
    }
    payload.update(changes)
    return payload


def test_store_is_idempotent_and_restricts_permissions(tmp_path):
    db_path = tmp_path / "meeting_recordings.db"
    store = MeetingRecordingStore(db_path)

    first, first_created = store.save(_payload())
    second, second_created = store.save(_payload())

    assert first_created is True
    assert second_created is False
    assert second == first
    assert store.get(RECORDING_ID).transcript.endswith("sexta-feira.")
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600


def test_store_rejects_conflicting_retry_and_empty_transcript(tmp_path):
    store = MeetingRecordingStore(tmp_path / "meeting_recordings.db")
    store.save(_payload())

    with pytest.raises(ValueError, match="meeting_recording_conflict"):
        store.save(_payload(transcript="Outro conteúdo"))
    with pytest.raises(ValueError, match="invalid_meeting_transcript"):
        store.save(_payload(recording_id="meeting-other", transcript=""))


@pytest.mark.asyncio
async def test_authenticated_endpoint_persists_without_echoing_transcript(tmp_path):
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "server-secret-key"})
    )
    store = MeetingRecordingStore(tmp_path / "meeting_recordings.db")
    adapter._meeting_recording_store = lambda: store
    app = web.Application()
    app.router.add_post(
        "/v1/meeting-recordings",
        adapter._handle_meeting_recording,
    )

    async with TestClient(TestServer(app)) as client:
        denied = await client.post("/v1/meeting-recordings", json=_payload())
        accepted = await client.post(
            "/v1/meeting-recordings",
            headers={"Authorization": "Bearer server-secret-key"},
            json=_payload(),
        )
        accepted_body = await accepted.json()
        replay = await client.post(
            "/v1/meeting-recordings",
            headers={"Authorization": "Bearer server-secret-key"},
            json=_payload(),
        )

    assert denied.status == 401
    assert accepted.status == 201
    assert replay.status == 200
    assert accepted_body["persisted"] is True
    assert accepted_body["recording_id"] == RECORDING_ID
    assert "transcript" not in json.dumps(accepted_body)
    assert store.get(RECORDING_ID) is not None
