import logging
from types import SimpleNamespace


def test_available_context_marks_injected_memory_without_claiming_consultation():
    from gateway.run import _response_available_context

    agent = SimpleNamespace(
        _memory_store=object(), _memory_enabled=True, _user_profile_enabled=False
    )

    assert _response_available_context(agent, [{"role": "user", "content": "prior"}]) == {
        "direct_user_input",
        "session_history",
        "persistent_memory",
    }


def test_provenance_log_is_one_safe_structured_info_line(caplog):
    from agent.provenance import ResponseProvenance
    from gateway.run import _log_response_provenance

    provenance = ResponseProvenance(available_context={"direct_user_input"})
    provenance.record_tool_call("memory")

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _log_response_provenance(provenance)

    records = [record.getMessage() for record in caplog.records if "response provenance:" in record.getMessage()]
    assert records == [
        "response provenance: consulted_sources=[persistent_memory] tools=[memory] "
        "available_context=[direct_user_input] document_counts={} policy_scopes=[]"
    ]


def test_provenance_logging_failure_is_sanitized(caplog):
    from gateway.run import _log_response_provenance

    class BrokenProvenance:
        def format_log_line(self):
            raise RuntimeError("query=/private/path secret=abc")

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        _log_response_provenance(BrokenProvenance())

    assert "response provenance logging failed" in caplog.text
    assert "/private/path" not in caplog.text
    assert "secret=abc" not in caplog.text
