"""An excluded home must remain untouched during ticker startup as well as ticks."""

import threading

from cron.scheduler_provider import InProcessCronScheduler
from hermes_constants import get_hermes_home


def test_startup_recovery_and_heartbeat_respect_owner_gate(tmp_path, monkeypatch):
    own, foreign = tmp_path / "own", tmp_path / "foreign"
    own.mkdir()
    foreign.mkdir()
    recovered, heartbeats, ticks = [], [], []
    stop = threading.Event()
    provider = InProcessCronScheduler()
    monkeypatch.setattr(provider, "recover_interrupted", lambda: recovered.append(get_hermes_home()) or 0)
    monkeypatch.setattr("cron.jobs.record_ticker_heartbeat", lambda **kw: heartbeats.append(get_hermes_home()))

    def tick(**_kw):
        ticks.append(get_hermes_home())
        stop.set()

    monkeypatch.setattr("cron.scheduler.tick", tick)
    provider.start(stop, interval=0, profile_homes=[("own", own), ("foreign", foreign)],
                   profile_gate=lambda name, home: home == own)
    assert recovered == [own]
    assert set(heartbeats) == {own}
    assert ticks == [own]
    assert not (foreign / "cron").exists()
