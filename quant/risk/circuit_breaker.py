"""Drawdown circuit breaker: halts trading when drawdown exceeds threshold."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DrawdownCircuitBreaker:
    """Monitors portfolio drawdown and halts new orders when breached.

    Attributes:
        max_drawdown_threshold: Maximum allowable drawdown as a positive
            fraction (e.g., 0.10 = halt at 10% drawdown from peak).
        reset_on_new_peak: If True, the breaker automatically resets when
            portfolio value recovers to a new peak.  If False, once tripped
            the breaker stays open until manually reset.
    """

    max_drawdown_threshold: float = 0.10
    reset_on_new_peak: bool = True

    _tripped: bool = False
    _peak_value: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < self.max_drawdown_threshold < 1.0):
            raise ValueError("max_drawdown_threshold must be in (0, 1)")

    def update(self, current_value: float) -> None:
        """Update the circuit breaker state with the latest portfolio value.

        Call this once per period (bar/tick) before evaluating new orders.

        Args:
            current_value: Current portfolio value (must be positive).
        """
        if current_value <= 0:
            return

        if current_value > self._peak_value:
            self._peak_value = current_value
            if self.reset_on_new_peak:
                self._tripped = False

        if self._peak_value > 0:
            drawdown = (self._peak_value - current_value) / self._peak_value
            if drawdown >= self.max_drawdown_threshold:
                self._tripped = True

    def is_tripped(self) -> bool:
        """Return True if the circuit breaker has halted trading."""
        return self._tripped

    def current_drawdown(self) -> float:
        """Return the current drawdown as a fraction of peak value."""
        if self._peak_value <= 0:
            return 0.0
        return 0.0 if self._peak_value == 0 else max(
            0.0,
            (self._peak_value - self._peak_value) / self._peak_value,
        )

    def check(self, current_value: float) -> tuple[bool, str]:
        """Convenience method: update state and return (approved, reason).

        Args:
            current_value: Current portfolio value.

        Returns:
            (True, "") if trading is allowed, (False, reason) if halted.
        """
        self.update(current_value)
        if self._tripped:
            dd = (
                (self._peak_value - current_value) / self._peak_value
                if self._peak_value > 0
                else 0.0
            )
            return (
                False,
                f"Drawdown circuit breaker tripped: current drawdown "
                f"{dd:.1%} >= threshold {self.max_drawdown_threshold:.1%}",
            )
        return True, ""

    def reset(self) -> None:
        """Manually reset the circuit breaker (re-enables trading)."""
        self._tripped = False
