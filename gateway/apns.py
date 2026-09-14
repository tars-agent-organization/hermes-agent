"""Private APNs bridge for first-party Hermes clients.

The API server owns device registration and calls :class:`APNSPushService`
after a run changes to a user-actionable state.  Push payloads deliberately do
not contain agent output: the authenticated client fetches the run after it is
woken, which keeps professional, personal, and health content off the lock
screen and out of Apple's notification transport.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


logger = logging.getLogger(__name__)

_DEVICE_TOKEN_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_INSTALLATION_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,128}$")
_APNS_ENVIRONMENTS = frozenset({"sandbox", "production"})
_APNS_HOSTS = {
    "sandbox": "https://api.sandbox.push.apple.com",
    "production": "https://api.push.apple.com",
}
_TERMINAL_INVALID_TOKEN_REASONS = frozenset({"BadDeviceToken", "DeviceTokenNotForTopic", "Unregistered"})


class APNSConfigurationError(RuntimeError):
    """Raised when the server cannot authenticate to APNs."""


class APNSDeliveryError(RuntimeError):
    """Raised when Apple rejects or cannot accept a push."""

    def __init__(self, reason: str, *, status_code: Optional[int] = None):
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


@dataclass(frozen=True)
class APNSDevice:
    installation_id: str
    token: str
    environment: str
    created_at: float
    updated_at: float
    enabled: bool = True


@dataclass(frozen=True)
class APNSSettings:
    enabled: bool
    team_id: str
    key_id: str
    topic: str
    auth_key: str

    @property
    def configured(self) -> bool:
        return bool(
            self.enabled
            and self.team_id.strip()
            and self.key_id.strip()
            and self.topic.strip()
            and self.auth_key.strip()
        )


class APNSDeviceStore:
    """Profile-local SQLite registry for APNs device tokens."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            from hermes_constants import get_hermes_home

            db_path = get_hermes_home() / "apns_devices.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._lock = threading.RLock()
        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(self._conn, db_label="apns_devices.db")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS devices (
                installation_id TEXT PRIMARY KEY,
                token TEXT NOT NULL UNIQUE,
                environment TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        self._conn.commit()
        self._tighten_permissions()

    @staticmethod
    def validate_installation_id(value: Any) -> str:
        installation_id = str(value or "").strip()
        if not _INSTALLATION_ID_RE.fullmatch(installation_id):
            raise ValueError("invalid_installation_id")
        return installation_id

    @staticmethod
    def validate_token(value: Any) -> str:
        token = str(value or "").strip().lower()
        if not _DEVICE_TOKEN_RE.fullmatch(token):
            raise ValueError("invalid_device_token")
        return token

    @staticmethod
    def validate_environment(value: Any) -> str:
        environment = str(value or "").strip().lower()
        if environment not in _APNS_ENVIRONMENTS:
            raise ValueError("invalid_apns_environment")
        return environment

    def register(self, installation_id: Any, token: Any, environment: Any) -> APNSDevice:
        installation_id = self.validate_installation_id(installation_id)
        token = self.validate_token(token)
        environment = self.validate_environment(environment)
        now = time.time()
        with self._lock:
            # APNs can rotate a token and an app reinstall can rotate its local
            # installation id.  A token must belong to exactly one installation.
            self._conn.execute(
                "DELETE FROM devices WHERE token = ? AND installation_id <> ?",
                (token, installation_id),
            )
            self._conn.execute(
                """INSERT INTO devices (
                       installation_id, token, environment, enabled,
                       created_at, updated_at
                   ) VALUES (?, ?, ?, 1, ?, ?)
                   ON CONFLICT(installation_id) DO UPDATE SET
                       token = excluded.token,
                       environment = excluded.environment,
                       enabled = 1,
                       updated_at = excluded.updated_at""",
                (installation_id, token, environment, now, now),
            )
            self._conn.commit()
        self._tighten_permissions()
        device = self.get(installation_id)
        if device is None:  # pragma: no cover - SQLite write/read invariant
            raise RuntimeError("device_registration_failed")
        return device

    def get(self, installation_id: Any) -> Optional[APNSDevice]:
        installation_id = self.validate_installation_id(installation_id)
        with self._lock:
            row = self._conn.execute(
                """SELECT installation_id, token, environment, created_at,
                          updated_at, enabled
                   FROM devices WHERE installation_id = ?""",
                (installation_id,),
            ).fetchone()
        if row is None:
            return None
        return APNSDevice(
            installation_id=row[0],
            token=row[1],
            environment=row[2],
            created_at=float(row[3]),
            updated_at=float(row[4]),
            enabled=bool(row[5]),
        )

    def delete(self, installation_id: Any) -> bool:
        installation_id = self.validate_installation_id(installation_id)
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM devices WHERE installation_id = ?", (installation_id,)
            )
            self._conn.commit()
            return bool(cursor.rowcount)

    def disable(self, installation_id: Any) -> None:
        installation_id = self.validate_installation_id(installation_id)
        with self._lock:
            self._conn.execute(
                "UPDATE devices SET enabled = 0, updated_at = ? WHERE installation_id = ?",
                (time.time(), installation_id),
            )
            self._conn.commit()

    def _tighten_permissions(self) -> None:
        for candidate in (
            self._db_path,
            Path(f"{self._db_path}-wal"),
            Path(f"{self._db_path}-shm"),
        ):
            try:
                if candidate.exists():
                    candidate.chmod(0o600)
            except OSError:
                logger.debug("Failed to restrict APNs registry permissions", exc_info=True)


def privacy_safe_payload(event: str, run_id: str) -> dict[str, Any]:
    """Build a bounded push containing no prompt, tool arguments, or output."""
    messages = {
        "run.completed": ("TARS respondeu", "Seu comando foi concluído."),
        "run.failed": ("TARS precisa de atenção", "Não foi possível concluir um comando."),
        "approval.request": ("TARS aguarda você", "Uma autorização é necessária para continuar."),
    }
    if event not in messages:
        raise ValueError("unsupported_push_event")
    title, body = messages[event]
    return {
        "aps": {
            "alert": {"title": title, "body": body},
            "sound": "default",
            "thread-id": "tars-runs",
        },
        "tars": {
            "event": event,
            "run_id": str(run_id)[:128],
        },
    }


class APNSPushService:
    """Token-authenticated HTTP/2 sender backed by a profile-local store."""

    def __init__(
        self,
        settings: APNSSettings,
        store: APNSDeviceStore,
        *,
        client_factory: Optional[Callable[..., Any]] = None,
    ):
        self.settings = settings
        self.store = store
        self._client_factory = client_factory
        self._provider_token: Optional[str] = None
        self._provider_token_issued_at = 0.0

    def _authorization_token(self) -> str:
        now = time.time()
        if self._provider_token and now - self._provider_token_issued_at < 45 * 60:
            return self._provider_token
        if not self.settings.configured:
            raise APNSConfigurationError("apns_not_configured")
        import jwt

        try:
            token = jwt.encode(
                {"iss": self.settings.team_id, "iat": int(now)},
                self.settings.auth_key,
                algorithm="ES256",
                headers={"kid": self.settings.key_id},
            )
        except Exception as exc:
            raise APNSConfigurationError("invalid_apns_auth_key") from exc
        self._provider_token = token
        self._provider_token_issued_at = now
        return token

    async def send(self, installation_id: str, event: str, run_id: str) -> str:
        device = self.store.get(installation_id)
        if device is None or not device.enabled:
            raise APNSDeliveryError("device_not_registered")

        payload = privacy_safe_payload(event, run_id)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        if len(encoded) > 4096:  # defensive: Apple rejects oversized alert payloads
            raise APNSDeliveryError("payload_too_large")

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - declared core dependency
            raise APNSConfigurationError("httpx_unavailable") from exc

        factory = self._client_factory or httpx.AsyncClient
        host = _APNS_HOSTS[device.environment]
        headers = {
            "authorization": f"bearer {self._authorization_token()}",
            "apns-topic": self.settings.topic,
            "apns-push-type": "alert",
            "apns-priority": "10",
            "apns-expiration": "0",
            "apns-collapse-id": f"tars-{run_id}"[:64],
        }
        try:
            async with factory(http2=True, timeout=10.0) as client:
                response = await client.post(
                    f"{host}/3/device/{device.token}",
                    content=encoded,
                    headers=headers,
                )
        except APNSDeliveryError:
            raise
        except Exception as exc:
            raise APNSDeliveryError("apns_unreachable") from exc

        if response.status_code == 200:
            return response.headers.get("apns-id", "")

        reason = "apns_rejected"
        try:
            body = response.json()
            if isinstance(body, Mapping) and body.get("reason"):
                reason = str(body["reason"])
        except Exception:
            pass
        if reason in _TERMINAL_INVALID_TOKEN_REASONS:
            self.store.disable(installation_id)
        raise APNSDeliveryError(reason, status_code=response.status_code)
