from concurrent.futures import ThreadPoolExecutor

from gateway.context_occupancy import ContextOccupancyMonitor
from hermes_state import SessionDB


def test_monitor_ignores_invalid_samples_and_fires_boundary_once(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    monitor = ContextOccupancyMonitor(db)

    assert monitor.observe(logical_session_id="s", prompt_tokens=-1,
                           effective_context_length=1000, model="m", provider="p") == []
    assert monitor.observe(logical_session_id="s", prompt_tokens=249,
                           effective_context_length=1000, model="m", provider="p") == []
    alerts = monitor.observe(logical_session_id="s", prompt_tokens=250,
                             effective_context_length=1000, model="m", provider="p")
    assert [alert.threshold_percent for alert in alerts] == [25]
    assert monitor.observe(logical_session_id="s", prompt_tokens=250,
                           effective_context_length=1000, model="m", provider="p") == []


def test_jump_fires_all_thresholds_and_compression_does_not_rearm(tmp_path):
    monitor = ContextOccupancyMonitor(SessionDB(tmp_path / "state.db"))

    alerts = monitor.observe(logical_session_id="s", prompt_tokens=800,
                             effective_context_length=1000, model="m", provider="p")
    assert [alert.threshold_percent for alert in alerts] == [25, 50, 75]
    assert monitor.observe(logical_session_id="s", prompt_tokens=100,
                           effective_context_length=1000, model="m", provider="p") == []
    assert monitor.observe(logical_session_id="s", prompt_tokens=800,
                           effective_context_length=1000, model="m", provider="p") == []


def test_state_is_durable_and_claim_is_atomic(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.claim_context_thresholds("s", prompt_tokens=250, effective_context_length=1000)
    db.close()
    reopened = SessionDB(path)
    assert reopened.get_context_occupancy_state("s")["fired_mask"] == 1

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda _: reopened.claim_context_thresholds(
                "concurrent", prompt_tokens=800, effective_context_length=1000
            ),
            range(2),
        ))
    assert sorted(results) == [[], [25, 50, 75]]


def test_compression_lineage_root_stops_at_branch(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", "whatsapp")
    db.end_session("root", "compression")
    db.create_session("continuation", "whatsapp", parent_session_id="root")
    assert db.get_compression_lineage_root("continuation") == "root"

    # Even when the parent has a compression end marker, explicit branch
    # provenance makes this a fresh logical session rather than a continuation.
    db.create_session("branch_parent", "whatsapp")
    db.end_session("branch_parent", "compression")
    db.create_session(
        "branch", "whatsapp", parent_session_id="branch_parent",
        model_config={"_branched_from": "branch_parent"},
    )
    assert db.get_compression_lineage_root("branch") == "branch"