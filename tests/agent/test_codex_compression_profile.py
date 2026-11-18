"""Profile opt-out for opaque Codex zstd failures (upstream #148898).

Only synthetic local HTTP requests; no provider requests or account state.
"""
import asyncio
import json

import pytest

from agent.process_bootstrap import build_keepalive_http_client
from hermes_cli.config import atomic_config_write
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

zstd = pytest.importorskip("compression.zstd")
OFFICIAL = "https://chatgpt.com/backend-api/codex"
PAYLOAD = {"model": "synthetic", "input": [{"role": "user", "content": "synthetic data " * 100}]}


def _profile_client(home, *, compression=None, async_mode=False):
    home.mkdir()
    config = {} if compression is None else {"agent": {"codex_request_compression": compression}}
    atomic_config_write(home / "config.yaml", config)
    token = set_hermes_home_override(home)
    try:
        return build_keepalive_http_client(OFFICIAL, async_mode=async_mode)
    finally:
        reset_hermes_home_override(token)


def _prepare(client):
    request = client.build_request("POST", OFFICIAL + "/responses", json=PAYLOAD)
    for hook in client.event_hooks["request"]:
        hook(request)
    return request


def test_opt_out_sends_identical_plain_json(tmp_path):
    with _profile_client(tmp_path / "disabled", compression=False) as client:
        request = _prepare(client)
    assert "content-encoding" not in request.headers
    assert json.loads(request.content) == PAYLOAD
    assert int(request.headers["content-length"]) == len(request.content)


def test_async_opt_out_sends_plain_json(tmp_path):
    async def prepare():
        async with _profile_client(tmp_path / "disabled", compression=False, async_mode=True) as client:
            request = client.build_request("POST", OFFICIAL + "/responses", json=PAYLOAD)
            for hook in client.event_hooks["request"]:
                await hook(request)
            return request

    request = asyncio.run(prepare())
    assert "content-encoding" not in request.headers
    assert json.loads(request.content) == PAYLOAD


@pytest.mark.parametrize("compression", [None, True])
def test_other_profiles_retain_existing_compression(tmp_path, compression):
    with _profile_client(tmp_path / "unchanged", compression=compression) as client:
        request = _prepare(client)
    assert request.headers["content-encoding"] == "zstd"
    assert json.loads(zstd.decompress(request.content)) == PAYLOAD


def test_profile_clients_keep_separate_encoding_after_scope_changes(tmp_path):
    with _profile_client(tmp_path / "plain", compression=False) as plain:
        with _profile_client(tmp_path / "compressed", compression=True) as compressed:
            plain_request = _prepare(plain)
            compressed_request = _prepare(compressed)
    assert "content-encoding" not in plain_request.headers
    assert compressed_request.headers["content-encoding"] == "zstd"
    assert plain_request.content == zstd.decompress(compressed_request.content)
