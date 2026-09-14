"""Behavioral tests for the private iOS APNs bridge."""

import json
import stat
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from gateway.apns import (
    APNSDeliveryError,
    APNSDeviceStore,
    APNSPushService,
    APNSSettings,
    privacy_safe_payload,
)
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


INSTALLATION_ID = "ios-installation-1234"
DEVICE_TOKEN = "a" * 64


def _private_key_pem() -> str:
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def _settings() -> APNSSettings:
    return APNSSettings(
        enabled=True,
        team_id="TEAM123456",
        key_id="KEY1234567",
        topic="local.tars.MetaWearablesPhase0",
        auth_key=_private_key_pem(),
    )


class _FakeResponse:
    def __init__(self, status_code=200, body=None, headers=None):
        self.status_code = status_code
        self._body = body or {}
        self.headers = headers or {}

    def json(self):
        return self._body


class _FakeHTTPClient:
    def __init__(self, response, capture, **kwargs):
        self.response = response
        self.capture = capture
        self.capture["client_options"] = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, content, headers):
        self.capture.update({"url": url, "content": content, "headers": headers})
        return self.response


def test_device_registry_rotates_token_and_restricts_permissions(tmp_path):
    store = APNSDeviceStore(tmp_path / "devices.db")
    first = store.register(INSTALLATION_ID, DEVICE_TOKEN.upper(), "sandbox")
    assert first.token == DEVICE_TOKEN
    assert first.enabled is True
    assert stat.S_IMODE((tmp_path / "devices.db").stat().st_mode) == 0o600

    rotated = store.register(INSTALLATION_ID, "b" * 64, "production")
    assert rotated.token == "b" * 64
    assert rotated.environment == "production"
    assert store.delete(INSTALLATION_ID) is True
    assert store.get(INSTALLATION_ID) is None


@pytest.mark.parametrize(
    ("installation_id", "token", "environment", "error"),
    [
        ("bad id", DEVICE_TOKEN, "sandbox", "invalid_installation_id"),
        (INSTALLATION_ID, "not-a-token", "sandbox", "invalid_device_token"),
        (INSTALLATION_ID, DEVICE_TOKEN, "development", "invalid_apns_environment"),
    ],
)
def test_device_registry_rejects_unbounded_identifiers(
    tmp_path, installation_id, token, environment, error
):
    store = APNSDeviceStore(tmp_path / "devices.db")
    with pytest.raises(ValueError, match=error):
        store.register(installation_id, token, environment)


def test_push_payload_contains_pointer_but_no_private_content():
    payload = privacy_safe_payload("run.completed", "run_123")
    encoded = json.dumps(payload)
    assert payload["tars"] == {"event": "run.completed", "run_id": "run_123"}
    assert "run output" not in encoded.lower()
    assert "prompt" not in encoded.lower()


@pytest.mark.asyncio
async def test_sender_uses_http2_token_auth_and_apple_topic(tmp_path):
    store = APNSDeviceStore(tmp_path / "devices.db")
    store.register(INSTALLATION_ID, DEVICE_TOKEN, "sandbox")
    capture = {}
    response = _FakeResponse(200, headers={"apns-id": "apple-id"})
    service = APNSPushService(
        _settings(),
        store,
        client_factory=lambda **kwargs: _FakeHTTPClient(response, capture, **kwargs),
    )

    apns_id = await service.send(INSTALLATION_ID, "run.completed", "run_123")

    assert apns_id == "apple-id"
    assert capture["client_options"]["http2"] is True
    assert capture["url"].endswith(f"/3/device/{DEVICE_TOKEN}")
    assert capture["headers"]["apns-topic"] == "local.tars.MetaWearablesPhase0"
    assert capture["headers"]["authorization"].startswith("bearer ")
    body = json.loads(capture["content"])
    assert body["tars"]["run_id"] == "run_123"


@pytest.mark.asyncio
async def test_sender_disables_token_apple_marks_unregistered(tmp_path):
    store = APNSDeviceStore(tmp_path / "devices.db")
    store.register(INSTALLATION_ID, DEVICE_TOKEN, "production")
    service = APNSPushService(
        _settings(),
        store,
        client_factory=lambda **kwargs: _FakeHTTPClient(
            _FakeResponse(410, {"reason": "Unregistered"}), {}, **kwargs
        ),
    )

    with pytest.raises(APNSDeliveryError, match="Unregistered"):
        await service.send(INSTALLATION_ID, "run.failed", "run_123")
    assert store.get(INSTALLATION_ID).enabled is False


@pytest.mark.asyncio
async def test_registration_endpoint_is_authenticated_and_never_echoes_token(tmp_path):
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "key": "server-secret-key",
                "apns": {
                    "enabled": True,
                    "team_id": "TEAM123456",
                    "key_id": "KEY1234567",
                    "topic": "local.tars.MetaWearablesPhase0",
                },
            },
        )
    )
    store = APNSDeviceStore(tmp_path / "devices.db")
    service = SimpleNamespace(store=store, settings=SimpleNamespace(configured=False))
    adapter._apns_service = lambda: service

    app = web.Application()
    app.router.add_put(
        "/v1/push/devices/{installation_id}", adapter._handle_register_push_device
    )
    async with TestClient(TestServer(app)) as client:
        unauthorized = await client.put(
            f"/v1/push/devices/{INSTALLATION_ID}",
            json={"token": DEVICE_TOKEN, "environment": "sandbox"},
        )
        assert unauthorized.status == 401

        response = await client.put(
            f"/v1/push/devices/{INSTALLATION_ID}",
            headers={"Authorization": "Bearer server-secret-key"},
            json={"token": DEVICE_TOKEN, "environment": "sandbox"},
        )
        assert response.status == 200
        body = await response.json()

    assert body["status"] == "registered"
    assert body["delivery_configured"] is False
    assert DEVICE_TOKEN not in json.dumps(body)
    assert store.get(INSTALLATION_ID).token == DEVICE_TOKEN
