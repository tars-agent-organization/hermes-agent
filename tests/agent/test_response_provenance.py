import concurrent.futures


def test_ordinary_response_has_only_available_context():
    from agent.provenance import ResponseProvenance

    provenance = ResponseProvenance(
        available_context={"direct_user_input", "session_history", "persistent_memory"}
    )

    assert provenance.as_log_fields() == {
        "consulted_sources": [],
        "tools": [],
        "available_context": [
            "persistent_memory",
            "session_history",
            "direct_user_input",
        ],
        "document_counts": {},
        "policy_scopes": [],
    }


def test_real_tool_calls_map_to_safe_consulted_sources_and_dedupe_stably():
    from agent.provenance import ResponseProvenance

    provenance = ResponseProvenance()
    for tool_name in ("terminal", "web_extract", "read_file", "memory", "session_search", "web_search"):
        provenance.record_tool_call(tool_name)

    fields = provenance.as_log_fields()
    assert fields["consulted_sources"] == [
        "persistent_memory",
        "session_history",
        "web",
        "file",
        "terminal",
    ]
    assert fields["tools"] == [
        "memory", "read_file", "session_search", "terminal", "web_extract", "web_search"
    ]



def test_post_tool_observer_records_only_dispatched_tools():
    from agent.provenance import response_provenance_context
    from model_tools import _emit_post_tool_call_hook

    with response_provenance_context() as provenance:
        _emit_post_tool_call_hook(
            function_name="memory", function_args={"content": "do not log"}, result="{}"
        )
        _emit_post_tool_call_hook(
            function_name="terminal", function_args={"command": "do not log"}, result="{}", status="blocked"
        )

    assert provenance.as_log_fields()["consulted_sources"] == ["persistent_memory"]
    assert provenance.as_log_fields()["tools"] == ["memory"]


def test_shared_projection_metadata_is_private_and_aggregated():
    from agent.provenance import ResponseProvenance, register_tool_source

    assert register_tool_source("contact_memory_lookup", "obsidian_shared_projection")
    provenance = ResponseProvenance()
    provenance.record_tool_call("contact_memory_lookup")
    provenance.record_source_metadata(
        "obsidian_shared_projection",
        document_count=3,
        policy_scope="explicitly_shareable_only",
    )
    rendered = provenance.format_log_line()

    assert "obsidian_shared_projection" in rendered
    assert "document_counts={obsidian_shared_projection:3}" in rendered
    assert "policy_scopes=[explicitly_shareable_only]" in rendered
    for forbidden in ("Rosane", "5511", "/Users/", "sha256", "query", "content"):
        assert forbidden not in rendered



def test_plugin_can_register_a_safe_custom_source_but_not_identifier_like_labels():
    from agent.provenance import ResponseProvenance, register_tool_source

    assert register_tool_source("calendar_lookup", "calendar")
    assert not register_tool_source("customer_lookup", "customer_5511999999999")
    provenance = ResponseProvenance()
    provenance.record_tool_call("calendar_lookup")
    provenance.record_source_metadata("calendar", document_count=2, policy_scope="read_only")

    assert provenance.as_log_fields()["consulted_sources"] == ["calendar"]
    assert provenance.as_log_fields()["document_counts"] == {"calendar": 2}


def test_deferred_delivery_source_counts_only_after_secure_submission():
    from agent.provenance import (
        ResponseProvenance,
        record_successful_tool_call,
        register_tool_source,
    )

    assert register_tool_source(
        "deliver_authorized_document",
        "obsidian_authorized_document",
        defer_until_success=True,
    )
    provenance = ResponseProvenance()
    provenance.record_tool_call("deliver_authorized_document")
    assert provenance.as_log_fields()["consulted_sources"] == []
    assert provenance.as_log_fields()["tools"] == []

    record_successful_tool_call("deliver_authorized_document", provenance=provenance)
    provenance.record_source_metadata(
        "obsidian_authorized_document",
        document_count=2,
        policy_scope="authorized_document_delivery",
    )
    assert provenance.as_log_fields()["consulted_sources"] == [
        "obsidian_authorized_document"
    ]
    assert provenance.as_log_fields()["tools"] == ["deliver_authorized_document"]
    assert provenance.as_log_fields()["document_counts"] == {
        "obsidian_authorized_document": 2
    }


def test_plugin_context_supports_deferred_provenance_registration_for_submission_tools():
    from agent.provenance import ResponseProvenance
    from hermes_cli.plugins import PluginContext

    assert PluginContext.register_provenance_source(
        object(),
        "deferred_submission_fixture",
        "authorized_document_delivery",
        defer_until_success=True,
    )
    provenance = ResponseProvenance()
    provenance.record_tool_call("deferred_submission_fixture")
    assert provenance.as_log_fields()["consulted_sources"] == []


def test_unknown_plugin_source_and_tool_are_coarsened_without_raw_metadata():
    from agent.provenance import ResponseProvenance

    provenance = ResponseProvenance()
    provenance.record_tool_call("customer_5511999999999_lookup")
    provenance.record_source_metadata(
        "source-with/private/path",
        document_count=999999999,
        policy_scope="patient_record",
    )
    rendered = provenance.format_log_line()

    assert "other_tool" in rendered
    assert "customer_5511999999999_lookup" not in rendered
    assert "source-with/private/path" not in rendered
    assert "patient_record" not in rendered
    assert "999999999" not in rendered


def test_contextvar_collectors_are_isolated_across_concurrent_responses():
    from agent.provenance import current_response_provenance, response_provenance_context

    def collect(tool_name):
        with response_provenance_context(available_context={"direct_user_input"}) as provenance:
            current_response_provenance().record_tool_call(tool_name)
            return provenance.as_log_fields()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(collect, ("memory", "session_search")))

    assert first["consulted_sources"] == ["persistent_memory"]
    assert first["tools"] == ["memory"]
    assert second["consulted_sources"] == ["session_history"]
    assert second["tools"] == ["session_search"]
