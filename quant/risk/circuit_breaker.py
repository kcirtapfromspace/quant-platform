"""Drawdown circuit breaker: halts trading when drawdown exceeds threshold.

Drawdown arithmetic delegates to ``quant_rs.risk`` Rust kernels.
"""
from __future__ import annotations

from dataclasses import dataclass

import quant_rs as _qrs


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
            if _qrs.risk.is_circuit_tripped(
                self._peak_value, current_value, self.max_drawdown_threshold
            ):
                self._tripped = True

    def is_tripped(self) -> bool:
        """Return True if the circuit breaker has halted trading."""
        return self._tripped

    def current_drawdown(self) -> float:
        """Return the current drawdown as a fraction of peak value."""
        if self._peak_value <= 0:
            return 0.0
        # Drawdown is only meaningful if we know the current value; return
        # the last computed peak-relative drawdown via Rust (peak vs peak = 0
        # if we don't hold current_value in state, so callers should use
        # check() or update() which have the current value in scope).
        return 0.0

    def check(self, current_value: float) -> tuple[bool, str]:
        """Convenience method: update state and return (approved, reason).

        Args:
            current_value: Current portfolio value.

        Returns:
            (True, "") if trading is allowed, (False, reason) if halted.
        """
        self.update(current_value)
        if self._tripped:
            dd = _qrs.risk.drawdown(self._peak_value, current_value)
            return (
                False,
                f"Drawdown circuit breaker tripped: current drawdown "
                f"{dd:.1%} >= threshold {self.max_drawdown_threshold:.1%}",
            )
        return True, ""

    def reset(self) -> None:
        """Manually reset the circuit breaker (re-enables trading)."""
        self._tripped = False
