"""Regression tests for structural and textual 1Password redaction boundaries."""

from __future__ import annotations

import gzip
import json


_PRIVATE_URI = "op://SYNTHETIC_PRIVATE_VAULT/SYNTHETIC PRIVATE ITEM/SYNTHETIC PRIVATE FIELD"
_PRIVATE_VALUE = "SYNTHETIC_PRIVATE_CONFIG_VALUE_6db1"
_NESTED_VALUE = "SYNTHETIC_NESTED_CONFIG_VALUE_7ac2"


def test_structural_redaction_masks_every_scalar_below_secrets_source_env():
    from agent.redact import redact_config_structure

    config = {
        "secrets": {
            "onepassword": {
                "enabled": True,
                "env": {
                    "API_SERVER_KEY": _PRIVATE_URI,
                    "NESTED": {
                        "string": _PRIVATE_VALUE,
                        "list": [_NESTED_VALUE, 123, True, None],
                    },
                },
            },
            "plugin_source": {
                "env": {"ANY_NAME": _PRIVATE_VALUE},
            },
        },
        "benign": {"env": {"visible": "keep-me"}},
    }

    redacted = redact_config_structure(config)
    rendered = json.dumps(redacted, sort_keys=True)

    for forbidden in (_PRIVATE_URI, _PRIVATE_VALUE, _NESTED_VALUE, "123"):
        assert forbidden not in rendered
    assert redacted["benign"]["env"]["visible"] == "keep-me"
    assert redacted["secrets"]["onepassword"]["enabled"] is True
    assert "[REDACTED]" in rendered


def test_textual_redaction_removes_full_quoted_and_unquoted_op_references(monkeypatch):
    from agent import redact

    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    text = (
        f'json={{"reference":"{_PRIVATE_URI}"}}\n'
        f"plain reference={_PRIVATE_URI}\n"
        "safe line"
    )

    output = redact.redact_sensitive_text(text)

    assert _PRIVATE_URI not in output
    assert "SYNTHETIC_PRIVATE_VAULT" not in output
    assert "SYNTHETIC PRIVATE ITEM" not in output
    assert "safe line" in output


def test_config_display_redactor_uses_structural_env_rule():
    from hermes_cli.config import redact_config_value

    config = {
        "secrets": {
            "onepassword": {
                "env": {"API_SERVER_KEY": _PRIVATE_URI}
            }
        }
    }
    output = redact_config_value(config)

    assert _PRIVATE_URI not in str(output)
    assert output["secrets"]["onepassword"]["env"]["API_SERVER_KEY"] == "[REDACTED]"


def test_doctor_output_boundary_redacts_op_reference(capsys):
    from hermes_cli.doctor import check_warn

    check_warn("Synthetic diagnostic", _PRIVATE_URI)
    output = capsys.readouterr().out

    assert _PRIVATE_URI not in output
    assert "SYNTHETIC_PRIVATE_VAULT" not in output


def test_debug_report_redacts_caller_supplied_dump_and_log_payload():
    from hermes_cli.debug import LogSnapshot, collect_debug_report

    snapshots = {
        name: LogSnapshot(
            path=None,
            full_text=f"{name}:{_PRIVATE_URI}",
            tail_text=f"{name}:{_PRIVATE_URI}",
        )
        for name in ("agent", "errors", "gateway", "gui", "desktop")
    }
    output = collect_debug_report(
        dump_text=f"dump:{_PRIVATE_URI}", log_snapshots=snapshots
    )

    assert _PRIVATE_URI not in output
    assert "SYNTHETIC_PRIVATE_VAULT" not in output


def test_serialized_debug_bundle_reapplies_redaction_before_json():
    from hermes_cli.debug import build_nous_bundle

    raw = build_nous_bundle(
        {"report": f"report:{_PRIVATE_URI}", "agent.log": _PRIVATE_VALUE},
        redact=True,
    )
    decoded = gzip.decompress(raw).decode("utf-8")

    assert _PRIVATE_URI not in decoded
    assert _PRIVATE_VALUE not in decoded
    assert json.loads(decoded)["redacted"] is True


def test_dump_output_reapplies_textual_redaction(monkeypatch, tmp_path, capsys):
    from hermes_cli import dump

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(dump, "load_hermes_dotenv", lambda **kwargs: [])
    monkeypatch.setattr(dump, "_get_git_commit", lambda path: "candidate")
    monkeypatch.setattr(dump, "_get_git_commit_date", lambda path: "")
    monkeypatch.setattr(
        dump,
        "load_config",
        lambda: {
            "model": "synthetic/model",
            "terminal": {"backend": "local"},
            "fallback_providers": [_PRIVATE_URI],
            "toolsets": [],
        },
    )
    monkeypatch.setattr(dump, "_gateway_status", lambda: "stopped")
    monkeypatch.setattr(dump, "_count_skills", lambda path: 0)
    monkeypatch.setattr(dump, "_cron_summary", lambda path: "0")
    monkeypatch.setattr(dump, "_configured_platforms", lambda: [])

    dump.run_dump(type("Args", (), {"show_keys": False})())
    output = capsys.readouterr().out

    assert _PRIVATE_URI not in output
    assert "SYNTHETIC_PRIVATE_VAULT" not in output
