"""Per-turn foreground tool-admission deadline for gateway sessions.

The guard is intentionally independent from tool dispatch.  It owns only the
monotonic deadline and the classification of tool calls made after that
deadline; executors remain responsible for turning denied decisions into tool
results.  Calls already admitted are never interrupted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import threading
import time
from typing import Any, Callable, Mapping


DEFAULT_STRICT_HANDOFF_ACK = (
    "I'm checking this in the background and will return here with the result."
)


@dataclass(frozen=True)
class ForegroundBudgetDecision:
    allowed: bool
    code: str = "allowed"
    message: str = ""
    is_handoff: bool = False


class ForegroundBudgetGuard:
    """Enforce one safe handoff after a gateway turn's deadline.

    A zero budget disables the guard, preserving the historical unbounded
    foreground behavior. Invalid, negative, and non-finite values are rejected
    instead of silently disabling enforcement. ``before_tool`` is thread-safe
    because a concurrent tool batch may race to claim the single handoff slot.
    """

    def __init__(
        self,
        budget_seconds: float | int,
        *,
        strict_background_handoff: bool = False,
        strict_handoff_ack: str = DEFAULT_STRICT_HANDOFF_ACK,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(budget_seconds, bool)
            or not isinstance(budget_seconds, Real)
            or not math.isfinite(budget_seconds)
            or budget_seconds < 0
        ):
            raise ValueError(
                "foreground budget must be a finite non-negative number"
            )
        self.budget_seconds = float(budget_seconds)
        self.strict_background_handoff = bool(strict_background_handoff)
        self.strict_handoff_ack = (
            str(strict_handoff_ack or "").strip() or DEFAULT_STRICT_HANDOFF_ACK
        )
        self._clock = clock
        self._turn_started_at: float | None = None
        self._handoff_started = False
        self._strict_handoff_ack_pending = False
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.budget_seconds > 0

    def start_turn(self) -> None:
        with self._lock:
            self._turn_started_at = self._clock()
            self._handoff_started = False
            self._strict_handoff_ack_pending = False

    def observe_tool_result(
        self,
        function_name: str,
        function_args: Mapping[str, Any] | None,
        function_result: Any,
    ) -> bool:
        """Latch a successful detached delegation regardless of the deadline.

        The foreground deadline is a fallback safety net. Strict handoff is a
        stronger contract: once ``delegate_task`` confirms that work was
        dispatched, the parent turn must stop immediately even when dispatch
        happened before the deadline.
        """
        if not self.strict_background_handoff or function_name != "delegate_task":
            return False

        payload = function_result
        if isinstance(payload, str):
            try:
                import json

                payload = json.loads(payload)
            except (TypeError, ValueError):
                return False
        if not isinstance(payload, Mapping) or payload.get("status") != "dispatched":
            return False

        with self._lock:
            self._handoff_started = True
            self._strict_handoff_ack_pending = True
        return True

    def consume_strict_handoff_ack(self) -> str | None:
        """Return the one deterministic acknowledgment for this turn."""
        with self._lock:
            if not self._strict_handoff_ack_pending:
                return None
            self._strict_handoff_ack_pending = False
            return self.strict_handoff_ack

    def before_tool(
        self, function_name: str, function_args: Mapping[str, Any] | None
    ) -> ForegroundBudgetDecision:
        with self._lock:
            if self.strict_background_handoff and self._handoff_started:
                return ForegroundBudgetDecision(
                    allowed=False,
                    code="foreground_handoff_already_started",
                    message=(
                        "A background delegation has already been dispatched. "
                        "Do not start another tool; return the configured "
                        "handoff confirmation to the user now."
                    ),
                )
            if not self.enabled or self._turn_started_at is None:
                return ForegroundBudgetDecision(allowed=True)
            if self._clock() - self._turn_started_at < self.budget_seconds:
                return ForegroundBudgetDecision(allowed=True)

            args = function_args if isinstance(function_args, Mapping) else {}
            if self._is_safe_handoff(function_name, args):
                if self._handoff_started:
                    return ForegroundBudgetDecision(
                        allowed=False,
                        code="foreground_handoff_already_started",
                        message=(
                            "The foreground budget is exhausted and one safe "
                            "handoff has already been accepted. Do not start "
                            "another tool; return one concise handoff "
                            "confirmation to the user now."
                        ),
                    )
                self._handoff_started = True
                return ForegroundBudgetDecision(
                    allowed=True,
                    code="foreground_handoff_started",
                    is_handoff=True,
                )

            return ForegroundBudgetDecision(
                allowed=False,
                code="foreground_budget_exhausted",
                message=(
                    "The gateway turn's foreground budget is exhausted. This "
                    "tool was not started. You may make exactly one safe "
                    "handoff using delegate_task, terminal with background=true, "
                    "or cronjob(action='create'); otherwise return a concise "
                    "status to the user."
                ),
            )

    @staticmethod
    def _is_safe_handoff(
        function_name: str, function_args: Mapping[str, Any]
    ) -> bool:
        if function_name == "delegate_task":
            return True
        if function_name == "terminal":
            return function_args.get("background") is True
        if function_name == "cronjob":
            return str(function_args.get("action") or "").strip().lower() == "create"
        return False


def foreground_budget_guard_from_config(
    config: Mapping[str, Any] | None,
    *,
    gateway_session_key: str | None,
    platform: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ForegroundBudgetGuard:
    """Build a guard from config, scoped strictly to gateway sessions."""
    budget: Any = 0
    strict_background_handoff = False
    strict_handoff_ack = DEFAULT_STRICT_HANDOFF_ACK
    if gateway_session_key and isinstance(config, Mapping):
        agent_config = config.get("agent")
        if isinstance(agent_config, Mapping):
            budget = agent_config.get("foreground_budget_seconds", 0)
            raw_platforms = agent_config.get("strict_background_handoff_platforms", [])
            if isinstance(raw_platforms, (list, tuple, set)):
                platform_key = str(platform or "").strip().lower()
                if not platform_key:
                    parts = str(gateway_session_key).split(":", 4)
                    platform_key = parts[2].strip().lower() if len(parts) >= 3 else ""
                strict_background_handoff = platform_key in {
                    str(item).strip().lower() for item in raw_platforms
                }
            strict_handoff_ack = agent_config.get(
                "strict_background_handoff_ack", DEFAULT_STRICT_HANDOFF_ACK
            )
    return ForegroundBudgetGuard(
        budget,
        strict_background_handoff=strict_background_handoff,
        strict_handoff_ack=strict_handoff_ack,
        clock=clock,
    )
