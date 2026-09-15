"""Durable transcript inbox for first-party meeting clients.

Clients keep raw audio locally, transcribe it on-device, and submit only the
transcript plus bounded routing metadata. A successful write is the client's
signal that the raw audio can be removed safely.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)

_RECORDING_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,128}$")
_SOURCES = frozenset({"in_person", "mac"})
_CONTEXTS = frozenset({"professional", "personal", "health"})
_DELIVERIES = frozenset({"app", "slack_dm", "whatsapp_dm"})
_MAX_TRANSCRIPT_CHARS = 2_000_000
_MAX_TITLE_CHARS = 200


@dataclass(frozen=True)
class MeetingRecording:
    recording_id: str
    transcript: str
    source: str
    context: str
    delivery: str
    started_at: str
    ended_at: str
    title: Optional[str]
    created_at: float


class MeetingRecordingStore:
    """Profile-local, idempotent SQLite inbox for meeting transcripts."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            from hermes_constants import get_hermes_home

            db_path = get_hermes_home() / "meeting_recordings.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._lock = threading.RLock()
        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(self._conn, db_label="meeting_recordings.db")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS meeting_recordings (
                recording_id TEXT PRIMARY KEY,
                transcript TEXT NOT NULL,
                source TEXT NOT NULL,
                context TEXT NOT NULL,
                delivery TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                title TEXT,
                created_at REAL NOT NULL
            )"""
        )
        self._conn.commit()
        self._tighten_permissions()

    @staticmethod
    def _bounded_text(value: Any, *, code: str, maximum: int) -> str:
        if not isinstance(value, str):
            raise ValueError(code)
        text = value.strip()
        if not text or len(text) > maximum:
            raise ValueError(code)
        return text

    @staticmethod
    def validate_recording_id(value: Any) -> str:
        recording_id = str(value or "").strip()
        if not _RECORDING_ID_RE.fullmatch(recording_id):
            raise ValueError("invalid_recording_id")
        return recording_id

    @staticmethod
    def _choice(value: Any, choices: frozenset[str], code: str) -> str:
        candidate = str(value or "").strip().lower()
        if candidate not in choices:
            raise ValueError(code)
        return candidate

    @classmethod
    def validate_payload(cls, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("invalid_meeting_payload")
        title = payload.get("title")
        if title is not None:
            title = cls._bounded_text(
                title,
                code="invalid_meeting_title",
                maximum=_MAX_TITLE_CHARS,
            )
        return {
            "recording_id": cls.validate_recording_id(payload.get("recording_id")),
            "transcript": cls._bounded_text(
                payload.get("transcript"),
                code="invalid_meeting_transcript",
                maximum=_MAX_TRANSCRIPT_CHARS,
            ),
            "source": cls._choice(
                payload.get("source"), _SOURCES, "invalid_meeting_source"
            ),
            "context": cls._choice(
                payload.get("context"), _CONTEXTS, "invalid_meeting_context"
            ),
            "delivery": cls._choice(
                payload.get("delivery"), _DELIVERIES, "invalid_meeting_delivery"
            ),
            "started_at": cls._bounded_text(
                payload.get("started_at"),
                code="invalid_meeting_started_at",
                maximum=64,
            ),
            "ended_at": cls._bounded_text(
                payload.get("ended_at"),
                code="invalid_meeting_ended_at",
                maximum=64,
            ),
            "title": title,
        }

    def save(self, payload: Any) -> tuple[MeetingRecording, bool]:
        values = self.validate_payload(payload)
        created_at = time.time()
        with self._lock:
            existing = self._get_unlocked(values["recording_id"])
            if existing is not None:
                expected = (
                    existing.transcript,
                    existing.source,
                    existing.context,
                    existing.delivery,
                    existing.started_at,
                    existing.ended_at,
                    existing.title,
                )
                submitted = (
                    values["transcript"],
                    values["source"],
                    values["context"],
                    values["delivery"],
                    values["started_at"],
                    values["ended_at"],
                    values["title"],
                )
                if expected != submitted:
                    raise ValueError("meeting_recording_conflict")
                return existing, False

            self._conn.execute(
                """INSERT INTO meeting_recordings (
                       recording_id, transcript, source, context, delivery,
                       started_at, ended_at, title, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    values["recording_id"],
                    values["transcript"],
                    values["source"],
                    values["context"],
                    values["delivery"],
                    values["started_at"],
                    values["ended_at"],
                    values["title"],
                    created_at,
                ),
            )
            self._conn.commit()
            result = self._get_unlocked(values["recording_id"])
        self._tighten_permissions()
        if result is None:  # pragma: no cover - SQLite write/read invariant
            raise RuntimeError("meeting_recording_write_failed")
        return result, True

    def get(self, recording_id: Any) -> Optional[MeetingRecording]:
        validated = self.validate_recording_id(recording_id)
        with self._lock:
            return self._get_unlocked(validated)

    def _get_unlocked(self, recording_id: str) -> Optional[MeetingRecording]:
        row = self._conn.execute(
            """SELECT recording_id, transcript, source, context, delivery,
                      started_at, ended_at, title, created_at
               FROM meeting_recordings WHERE recording_id = ?""",
            (recording_id,),
        ).fetchone()
        if row is None:
            return None
        return MeetingRecording(
            recording_id=row[0],
            transcript=row[1],
            source=row[2],
            context=row[3],
            delivery=row[4],
            started_at=row[5],
            ended_at=row[6],
            title=row[7],
            created_at=float(row[8]),
        )

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
                logger.debug(
                    "Failed to restrict meeting transcript permissions",
                    exc_info=True,
                )
