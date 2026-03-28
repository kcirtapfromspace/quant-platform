"""Drawdown-aware portfolio exposure scaler.

Continuously scales portfolio exposure as a function of current drawdown
depth, providing a graduated risk-off mechanism that complements the
binary circuit breaker.

Scaling function::

    scale = max(floor, 1 - ((dd / max_dd) ^ exponent))

Where ``dd`` is current drawdown (positive), ``max_dd`` is the maximum
drawdown threshold, ``floor`` is the minimum exposure, and ``exponent``
controls the aggressiveness of scaling (1 = linear, 2 = quadratic).

At zero drawdown the scale is 1.0 (full exposure).  As drawdown deepens
toward ``max_dd``, the scale falls toward ``floor``.

Key outputs:

  * **ScalerState** — current drawdown, scale factor, and peak.
  * **ScaledWeights** — adjusted portfolio weights after scaling.

Usage::

    from quant.risk.drawdown_scaler import (
        DrawdownScaler,
        DrawdownScalerConfig,
    )

    scaler = DrawdownScaler(DrawdownScalerConfig(
        max_drawdown=0.10,
        floor=0.25,
        exponent=1.5,
    ))
    scaler.update(portfolio_value)
    scaled = scaler.scale_weights(target_weights)
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DrawdownScalerConfig:
    """Configuration for drawdown-aware exposure scaling.

    Attributes:
        max_drawdown:   Maximum drawdown threshold — at this depth the
                        scale factor reaches ``floor``.
        floor:          Minimum exposure scale (0 = can go to zero,
                        0.25 = at least 25% exposure).
        exponent:       Controls scaling aggressiveness.  1.0 = linear,
                        >1 = convex (slow initial reduction, aggressive
                        later), <1 = concave (aggressive early).
        reset_on_peak:  If True, peak resets when portfolio reaches a
                        new high, restoring full exposure.
        warmup_periods: Number of update calls before scaling engages.
                        Prevents premature scaling on initialization.
    """

    max_drawdown: float = 0.10
    floor: float = 0.25
    exponent: float = 1.0
    reset_on_peak: bool = True
    warmup_periods: int = 0


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScalerState:
    """Current state of the drawdown scaler.

    Attributes:
        peak:           High-water mark portfolio value.
        current_value:  Most recent portfolio value.
        drawdown:       Current drawdown as a positive fraction (0 = at peak).
        scale_factor:   Exposure multiplier [floor, 1.0].
        n_updates:      Total number of update calls.
        is_active:      True if scaling is engaged (past warmup).
    """

    peak: float
    current_value: float
    drawdown: float
    scale_factor: float
    n_updates: int
    is_active: bool


@dataclass
class ScaledWeights:
    """Portfolio weights after drawdown scaling.

    Attributes:
        weights:        Scaled portfolio weights ``{symbol: weight}``.
        scale_factor:   Applied scale factor.
        drawdown:       Current drawdown.
        cash_weight:    Weight moved to cash (1 - sum of scaled weights).
    """

    weights: dict[str, float] = field(default_factory=dict)
    scale_factor: float = 1.0
    drawdown: float = 0.0
    cash_weight: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Drawdown Scaling (factor={self.scale_factor:.2f}, "
            f"dd={self.drawdown:.2%})",
            "=" * 60,
            "",
            f"Scale factor  : {self.scale_factor:.4f}",
            f"Drawdown      : {self.drawdown:.2%}",
            f"Cash weight   : {self.cash_weight:.2%}",
            f"N positions   : {len(self.weights)}",
        ]

        if self.weights:
            sorted_w = sorted(
                self.weights.items(), key=lambda x: abs(x[1]), reverse=True,
            )
            lines.append("")
            lines.append("Scaled weights (top 5):")
            for sym, w in sorted_w[:5]:
                lines.append(f"  {sym:<10s}: {w:+.4f}")
            if len(sorted_w) > 5:
                lines.append(f"  ... and {len(sorted_w) - 5} more")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------


class DrawdownScaler:
    """Continuously scales portfolio exposure based on drawdown depth.

    Call :meth:`update` once per period with the latest portfolio value,
    then :meth:`scale_weights` to get drawdown-adjusted weights.

    Args:
        config: Scaler configuration.
    """

    def __init__(self, config: DrawdownScalerConfig | None = None) -> None:
        self._config = config or DrawdownScalerConfig()
        self._peak = 0.0
        self._current = 0.0
        self._n_updates = 0

    @property
    def config(self) -> DrawdownScalerConfig:
        return self._config

    def update(self, portfolio_value: float) -> ScalerState:
        """Update scaler with latest portfolio value.

        Args:
            portfolio_value: Current portfolio value (must be positive).

        Returns:
            :class:`ScalerState` with current drawdown and scale factor.
        """
        if portfolio_value <= 0:
            return self.state

        self._n_updates += 1
        self._current = portfolio_value

        if portfolio_value > self._peak:
            self._peak = portfolio_value

        return self.state

    @property
    def state(self) -> ScalerState:
        """Current scaler state."""
        dd = self._current_drawdown()
        sf = self._compute_scale(dd)
        active = self._n_updates >= self._config.warmup_periods
        return ScalerState(
            peak=self._peak,
            current_value=self._current,
            drawdown=dd,
            scale_factor=sf if active else 1.0,
            n_updates=self._n_updates,
            is_active=active,
        )

    def scale_weights(
        self,
        target_weights: dict[str, float],
    ) -> ScaledWeights:
        """Apply drawdown scaling to target portfolio weights.

        Each weight is multiplied by the current scale factor.
        The freed-up weight is allocated to cash.

        Args:
            target_weights: Target portfolio weights ``{symbol: weight}``.

        Returns:
            :class:`ScaledWeights` with adjusted weights.
        """
        state = self.state
        sf = state.scale_factor

        scaled = {sym: w * sf for sym, w in target_weights.items()}
        total_w = sum(abs(w) for w in scaled.values())
        original_total = sum(abs(w) for w in target_weights.values())
        cash = max(original_total - total_w, 0.0)

        return ScaledWeights(
            weights=scaled,
            scale_factor=sf,
            drawdown=state.drawdown,
            cash_weight=cash,
        )

    def reset(self) -> None:
        """Reset the scaler to initial state."""
        self._peak = 0.0
        self._current = 0.0
        self._n_updates = 0

    # ── Internal ───────────────────────────────────────────────────

    def _current_drawdown(self) -> float:
        """Compute current drawdown as a positive fraction."""
        if self._peak <= 0 or self._current <= 0:
            return 0.0
        dd = 1.0 - (self._current / self._peak)
        return max(dd, 0.0)

    def _compute_scale(self, drawdown: float) -> float:
        """Compute the exposure scale factor for a given drawdown.

        scale = max(floor, 1 - (dd / max_dd) ^ exponent)
        """
        cfg = self._config

        if drawdown <= 0:
            return 1.0

        if cfg.max_drawdown <= 0:
            return cfg.floor

        ratio = min(drawdown / cfg.max_drawdown, 1.0)
        raw_scale = 1.0 - (ratio ** cfg.exponent)

        return max(raw_scale, cfg.floor)
