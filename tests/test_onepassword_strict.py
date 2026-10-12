"""Security contract for the strict 1Password selector used by TARS API auth.

All helpers and subprocesses are fakes.  This module must never invoke the real
``op`` CLI, a system bootstrap helper, or the network.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from types import SimpleNamespace

import pytest

from agent.secret_sources import onepassword as op_src


_SECRET = "SYNTHETIC_TARS_SECRET_VALUE_DO_NOT_DISCLOSE_91f2"
_BOOTSTRAP = "SYNTHETIC_BOOTSTRAP_TOKEN_DO_NOT_DISCLOSE_54a7"
_PRIVATE_ID = "SYNTHETIC_PRIVATE_ITEM_ID_777"
_PRIVATE_TITLE = "SYNTHETIC_PRIVATE_ITEM_TITLE_888"
_PRIVATE_FIELD = "SYNTHETIC_PRIVATE_FIELD_999"
_PRIVATE_URI = "op://SYNTHETIC_PRIVATE_VAULT/SYNTHETIC_PRIVATE_ITEM/SYNTHETIC_PRIVATE_FIELD"
_SELECTOR = "tars_api_server"


def _strict_cfg(**overrides):
    cfg = {
        "enabled": True,
        "mode": "strict_selector",
        "selector": _SELECTOR,
        "cache_ttl_seconds": 0,
        "override_existing": True,
    }
    cfg.update(overrides)
    return cfg


def _completed(*, returncode=0, payload=None, stderr=""):
    stdout = json.dumps(payload) if payload is not None else ""
    return subprocess.CompletedProcess(
        args=["fixed-helper"], returncode=returncode, stdout=stdout, stderr=stderr
    )


@pytest.fixture(autouse=True)
def _clean_sensitive_env(monkeypatch):
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    op_src._reset_cache_for_tests()


class TestStrictSelectorConfig:
    @pytest.mark.parametrize(
        "override",
        [
            {"selector": "not_allowlisted"},
            {"selector": _PRIVATE_URI},
            {"cache_ttl_seconds": 1},
            {"cache_ttl_seconds": "0"},
            {"env": {"API_SERVER_KEY": _PRIVATE_URI}},
            {"account": "private-account"},
            {"binary_path": "/private/path/to/op"},
            {"service_account_token_env": "PRIVATE_TOKEN_ENV"},
            {"item_id": _PRIVATE_ID},
            {"title": _PRIVATE_TITLE},
            {"field": _PRIVATE_FIELD},
        ],
    )
    def test_strict_config_rejects_non_allowlisted_or_private_metadata(self, override):
        with pytest.raises(op_src.StrictOnePasswordConfigError) as raised:
            op_src.validate_strict_onepassword_config(_strict_cfg(**override))

        rendered = str(raised.value)
        for forbidden in (
            _PRIVATE_URI,
            _PRIVATE_ID,
            _PRIVATE_TITLE,
            _PRIVATE_FIELD,
            "private-account",
            "/private/path/to/op",
            "PRIVATE_TOKEN_ENV",
        ):
            assert forbidden not in rendered

    def test_valid_strict_config_contains_only_non_sensitive_selector(self):
        assert op_src.validate_strict_onepassword_config(_strict_cfg()) == _SELECTOR

    def test_save_config_refuses_to_persist_invalid_strict_config(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.config import save_config

        config = {
            "secrets": {
                "onepassword": _strict_cfg(
                    env={"API_SERVER_KEY": _PRIVATE_URI}
                )
            }
        }
        with pytest.raises(op_src.StrictOnePasswordConfigError):
            save_config(config, strip_defaults=False)

        path = tmp_path / "config.yaml"
        assert not path.exists() or _PRIVATE_URI not in path.read_text()


class TestStrictHelperBoundary:
    def test_fixed_argv_shell_false_selector_only_stdin_and_secret_memory_only(
        self, monkeypatch, tmp_path, capsys, caplog
    ):
        monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", _BOOTSTRAP)
        calls = []

        def fake_run(argv, **kwargs):
            calls.append((argv, kwargs))
            return _completed(
                payload={
                    "matches": [
                        {
                            "value": _SECRET,
                            "metadata": {
                                "id": _PRIVATE_ID,
                                "title": _PRIVATE_TITLE,
                                "field": _PRIVATE_FIELD,
                                "reference": _PRIVATE_URI,
                            },
                        }
                    ]
                }
            )

        monkeypatch.setattr(op_src.subprocess, "run", fake_run)
        caplog.set_level(logging.DEBUG)

        result = op_src.OnePasswordSource().fetch(_strict_cfg(), tmp_path)

        assert result.ok
        assert result.secrets == {"API_SERVER_KEY": _SECRET}
        assert result.binary_path is None
        assert len(calls) == 1
        argv, kwargs = calls[0]
        assert tuple(argv) == op_src.STRICT_HELPER_ARGV
        assert kwargs["shell"] is False
        assert kwargs["check"] is False
        assert kwargs["input"] == json.dumps(
            {"version": 1, "selector": _SELECTOR}, separators=(",", ":")
        )
        assert kwargs["stdin"] is None
        assert kwargs["env"]["OP_SERVICE_ACCOUNT_TOKEN"] == _BOOTSTRAP

        boundary = repr(argv) + kwargs["input"]
        for forbidden in (
            _SECRET,
            _BOOTSTRAP,
            _PRIVATE_URI,
            _PRIVATE_ID,
            _PRIVATE_TITLE,
            _PRIVATE_FIELD,
        ):
            assert forbidden not in boundary

        observable = (
            capsys.readouterr().out
            + capsys.readouterr().err
            + caplog.text
            + repr(result)
        )
        for forbidden in (
            _SECRET,
            _BOOTSTRAP,
            _PRIVATE_URI,
            _PRIVATE_ID,
            _PRIVATE_TITLE,
            _PRIVATE_FIELD,
        ):
            assert forbidden not in observable

    def test_ttl_zero_strict_path_never_reads_or_writes_either_cache(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", _BOOTSTRAP)
        monkeypatch.setattr(
            op_src.subprocess,
            "run",
            lambda *a, **k: _completed(
                payload={"matches": [{"value": _SECRET, "metadata": {}}]}
            ),
        )
        monkeypatch.setattr(
            op_src._DISK_CACHE,
            "read",
            lambda *a, **k: pytest.fail("strict mode read the disk cache"),
        )
        monkeypatch.setattr(
            op_src._DISK_CACHE,
            "write",
            lambda *a, **k: pytest.fail("strict mode wrote the disk cache"),
        )
        monkeypatch.setattr(
            op_src._CACHE,
            "get",
            lambda *a, **k: pytest.fail("strict mode read the memory cache"),
        )

        result = op_src.OnePasswordSource().fetch(_strict_cfg(), tmp_path)

        assert result.secrets == {"API_SERVER_KEY": _SECRET}
        assert not list(tmp_path.rglob("*cache*"))

    @pytest.mark.parametrize(
        ("case", "runner", "expected_category"),
        [
            (
                "missing",
                lambda *a, **k: (_ for _ in ()).throw(
                    FileNotFoundError(f"missing /private/{_PRIVATE_ID}")
                ),
                "helper_missing",
            ),
            (
                "auth",
                lambda *a, **k: _completed(
                    returncode=20, stderr=f"auth failed {_SECRET} {_PRIVATE_URI}"
                ),
                "auth_unavailable",
            ),
            (
                "blocked",
                lambda *a, **k: _completed(
                    returncode=21, stderr=f"blocked {_SECRET} {_PRIVATE_ID}"
                ),
                "access_blocked",
            ),
            (
                "timeout",
                lambda *a, **k: (_ for _ in ()).throw(
                    subprocess.TimeoutExpired(
                        cmd=["helper", _PRIVATE_URI], timeout=10, stderr=_SECRET
                    )
                ),
                "timeout",
            ),
            (
                "malformed",
                lambda *a, **k: subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=f"{{bad:{_SECRET}", stderr=_PRIVATE_URI
                ),
                "malformed_response",
            ),
            (
                "zero",
                lambda *a, **k: _completed(
                    payload={"matches": [], "metadata": _PRIVATE_ID}
                ),
                "zero_matches",
            ),
            (
                "multiple",
                lambda *a, **k: _completed(
                    payload={
                        "matches": [
                            {"value": _SECRET, "metadata": _PRIVATE_ID},
                            {"value": "another-secret", "metadata": _PRIVATE_TITLE},
                        ]
                    }
                ),
                "multiple_matches",
            ),
            (
                "empty",
                lambda *a, **k: _completed(
                    payload={
                        "matches": [
                            {
                                "value": "   ",
                                "metadata": {
                                    "id": _PRIVATE_ID,
                                    "title": _PRIVATE_TITLE,
                                },
                            }
                        ]
                    }
                ),
                "empty_value",
            ),
        ],
    )
    def test_failures_are_closed_categories_with_no_payload_leak(
        self,
        case,
        runner,
        expected_category,
        monkeypatch,
        tmp_path,
        capsys,
        caplog,
    ):
        monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", _BOOTSTRAP)
        monkeypatch.setattr(op_src.subprocess, "run", runner)
        caplog.set_level(logging.DEBUG)

        result = op_src.OnePasswordSource().fetch(_strict_cfg(), tmp_path)

        assert not result.ok, case
        assert result.secrets == {}
        assert result.error == f"strict_onepassword:{expected_category}"
        assert result.error_kind is not None
        assert result.warnings == []
        observable = (
            str(result.error)
            + repr(result)
            + capsys.readouterr().out
            + capsys.readouterr().err
            + caplog.text
        )
        for forbidden in (
            _SECRET,
            _BOOTSTRAP,
            _PRIVATE_URI,
            _PRIVATE_ID,
            _PRIVATE_TITLE,
            _PRIVATE_FIELD,
            "another-secret",
        ):
            assert forbidden not in observable

    def test_missing_bootstrap_fails_without_spawning_helper(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            op_src.subprocess,
            "run",
            lambda *a, **k: pytest.fail("helper spawned without bootstrap"),
        )

        result = op_src.OnePasswordSource().fetch(_strict_cfg(), tmp_path)

        assert result.error == "strict_onepassword:bootstrap_unavailable"
        assert result.secrets == {}


class TestStrictBootstrapLoading:
    @staticmethod
    def _write_strict_home(home, *, env_token, op_env_token):
        home.mkdir(exist_ok=True)
        (home / "config.yaml").write_text(
            "secrets:\n"
            "  onepassword:\n"
            "    enabled: true\n"
            "    mode: strict_selector\n"
            f"    selector: {_SELECTOR}\n"
            "    cache_ttl_seconds: 0\n"
        )
        (home / ".env").write_text(
            f"OP_SERVICE_ACCOUNT_TOKEN={env_token}\n"
            "API_SERVER_KEY=PLAINTEXT_FILE_API_KEY_MUST_NOT_WIN\n"
        )
        (home / ".op.env").write_text(
            f"OP_SERVICE_ACCOUNT_TOKEN={op_env_token}\n"
        )

    def test_strict_mode_preserves_only_preexisting_system_bootstrap(
        self, tmp_path, monkeypatch
    ):
        from hermes_cli import env_loader

        home = tmp_path / "home"
        self._write_strict_home(
            home,
            env_token="SYNTHETIC_DOTENV_TOKEN_MUST_NOT_LOAD",
            op_env_token="SYNTHETIC_OP_DOTENV_TOKEN_MUST_NOT_LOAD",
        )
        monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", _BOOTSTRAP)
        seen = {}

        def fake_apply(path):
            seen["bootstrap"] = os.environ.get("OP_SERVICE_ACCOUNT_TOKEN")
            seen["api_key"] = os.environ.get("API_SERVER_KEY")

        monkeypatch.setattr(env_loader, "_apply_external_secret_sources", fake_apply)
        env_loader.load_hermes_dotenv(hermes_home=home)

        assert seen == {"bootstrap": _BOOTSTRAP, "api_key": None}

    def test_strict_mode_has_no_dotenv_bootstrap_fallback(
        self, tmp_path, monkeypatch
    ):
        from hermes_cli import env_loader

        home = tmp_path / "home"
        self._write_strict_home(
            home,
            env_token="SYNTHETIC_DOTENV_TOKEN_MUST_NOT_LOAD",
            op_env_token="SYNTHETIC_OP_DOTENV_TOKEN_MUST_NOT_LOAD",
        )
        seen = {}

        def fake_apply(path):
            seen["bootstrap"] = os.environ.get("OP_SERVICE_ACCOUNT_TOKEN")
            seen["api_key"] = os.environ.get("API_SERVER_KEY")

        monkeypatch.setattr(env_loader, "_apply_external_secret_sources", fake_apply)
        env_loader.load_hermes_dotenv(hermes_home=home)

        assert seen == {"bootstrap": None, "api_key": None}


class TestStrictOnePasswordCliSurfaces:
    def test_status_never_prints_legacy_reference(self, monkeypatch, capsys):
        from hermes_cli import onepassword_secrets_cli as cli

        monkeypatch.setattr(
            cli,
            "load_config",
            lambda: {
                "secrets": {
                    "onepassword": {
                        "enabled": True,
                        "env": {"API_SERVER_KEY": _PRIVATE_URI},
                        "cache_ttl_seconds": 0,
                    }
                }
            },
        )
        monkeypatch.setattr(cli.op_src, "find_op", lambda *a, **k: None)

        assert cli.cmd_status(argparse.Namespace()) == 0
        output = capsys.readouterr().out
        assert _PRIVATE_URI not in output
        assert "[REDACTED]" in output

    def test_legacy_set_does_not_echo_reference(self, monkeypatch, capsys):
        from hermes_cli import onepassword_secrets_cli as cli

        config = {"secrets": {"onepassword": {"enabled": True, "env": {}}}}
        monkeypatch.setattr(cli, "load_config", lambda: config)
        monkeypatch.setattr(cli, "save_config", lambda value: None)

        result = cli.cmd_set(
            argparse.Namespace(env_var="API_SERVER_KEY", reference=_PRIVATE_URI)
        )

        assert result == 0
        assert config["secrets"]["onepassword"]["env"]["API_SERVER_KEY"] == _PRIVATE_URI
        assert _PRIVATE_URI not in capsys.readouterr().out

    def test_strict_set_rejects_reference_without_persisting_or_echoing(
        self, monkeypatch, capsys
    ):
        from hermes_cli import onepassword_secrets_cli as cli

        config = {"secrets": {"onepassword": _strict_cfg()}}
        saved = []
        monkeypatch.setattr(cli, "load_config", lambda: config)
        monkeypatch.setattr(cli, "save_config", saved.append)

        result = cli.cmd_set(
            argparse.Namespace(env_var="API_SERVER_KEY", reference=_PRIVATE_URI)
        )

        assert result == 1
        assert saved == []
        assert _PRIVATE_URI not in capsys.readouterr().out
