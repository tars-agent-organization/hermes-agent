"""Clock corrections must not let a scheduler steal another gateway's home."""

import os
import psutil
import pytest

from gateway import status


@pytest.mark.platforms("macos")
def test_gateway_identity_survives_clock_adjustment(monkeypatch, tmp_path):
    from gateway.run import _cron_profile_gate

    pid = os.getpid()
    before = status.get_process_start_time(pid)
    record = {"pid": pid, "start_time": before, "kind": "hermes-gateway"}
    # Only the clock adjustment layer changes; the real kernel process stays alive.
    monkeypatch.setattr(psutil._psplatform, "INIT_BOOT_TIME", psutil.boot_time() + 300)
    assert status.get_process_start_time(pid) == before
    assert status._live_pid_from_record(record) == pid
    monkeypatch.setattr(status, "is_gateway_runtime_lock_active", lambda *_: True)
    monkeypatch.setattr(status, "_read_pid_record", lambda *_: record)
    monkeypatch.setattr(status, "_read_gateway_lock_record", lambda *_: record)
    monkeypatch.setattr(status, "_record_matches_live_gateway_pid", lambda *a, **k: True)
    # The scheduler is another process; use the genuine identity reader and gate.
    monkeypatch.setattr("gateway.run.os.getpid", lambda: pid + 1)
    assert _cron_profile_gate("default", tmp_path) is False


@pytest.mark.platforms("macos")
def test_changed_process_incarnation_is_still_rejected(monkeypatch):
    pid = os.getpid()
    before = status.get_process_start_time(pid)
    monkeypatch.setattr(psutil._psplatform, "INIT_BOOT_TIME", psutil.boot_time() - 300)
    assert status._live_pid_from_record({"pid": pid, "start_time": before - 1000}) is None
