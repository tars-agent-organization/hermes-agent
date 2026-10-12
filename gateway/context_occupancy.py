"""Persistent, event-driven context occupancy threshold monitoring."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextThresholdAlert:
    threshold_percent: int
    prompt_tokens: int
    effective_context_length: int
    model: str
    provider: str

    @property
    def message(self) -> str:
        return (
            f"📊 Contexto em {self.threshold_percent}% · "
            f"{self.prompt_tokens:,} / {self.effective_context_length:,} tokens · "
            f"{self.model} ({self.provider})"
        ).replace(",", ".")


class ContextOccupancyMonitor:
    def __init__(self, session_db):
        self.session_db = session_db

    def observe(self, *, logical_session_id: str, prompt_tokens: int,
                effective_context_length: int, model: str,
                provider: str) -> list[ContextThresholdAlert]:
        if (
            not logical_session_id
            or not isinstance(prompt_tokens, int)
            or isinstance(prompt_tokens, bool)
            or prompt_tokens <= 0
            or not isinstance(effective_context_length, int)
            or isinstance(effective_context_length, bool)
            or effective_context_length <= 0
        ):
            return []
        claimed = self.session_db.claim_context_thresholds(
            logical_session_id,
            prompt_tokens=prompt_tokens,
            effective_context_length=effective_context_length,
        )
        return [ContextThresholdAlert(
            threshold_percent=threshold,
            prompt_tokens=prompt_tokens,
            effective_context_length=effective_context_length,
            model=model or "unknown",
            provider=provider or "unknown",
        ) for threshold in claimed]