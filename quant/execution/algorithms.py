"""Execution algorithms — TWAP, VWAP, and market impact estimation.

Provides order-slicing algorithms that decompose a parent order into a
schedule of child orders to minimise market impact.  Each algorithm
produces an :class:`ExecutionSchedule` containing the ordered sequence of
:class:`OrderSlice` objects with timing, quantity, and optional limit
prices.

Supported algorithms:

  * **TWAP** (Time-Weighted Average Price): uniform slices at equal time
    intervals.  Best for low-urgency orders in liquid names.
  * **VWAP** (Volume-Weighted Average Price): sizes each slice
    proportional to a historical intraday volume profile so that
    execution tracks the market's natural volume curve.
  * **Market impact estimator**: square-root model for pre-trade cost
    estimation.

Usage::

    from quant.execution.algorithms import TWAPAlgorithm, VWAPAlgorithm

    algo = TWAPAlgorithm(n_slices=10)
    schedule = algo.schedule(
        symbol="AAPL",
        side="buy",
        quantity=10_000,
        duration_seconds=3600,  # execute over 1 hour
    )
    for s in schedule.slices:
        print(f"  t={s.scheduled_seconds:.0f}s  qty={s.quantity:.0f}")
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderSlice:
    """A single child order in an execution schedule.

    Attributes:
        sequence: 0-based index within the schedule.
        scheduled_seconds: Seconds from execution start when this slice
            should be submitted.
        quantity: Number of shares/units to trade in this slice.
        pct_of_parent: Fraction of the total parent order this slice
            represents (sums to ~1.0 across all slices).
        limit_offset_bps: Optional limit price offset from mid-price in
            basis points.  Positive = more aggressive (higher for buys,
            lower for sells).  ``None`` means market order.
    """

    sequence: int
    scheduled_seconds: float
    quantity: float
    pct_of_parent: float
    limit_offset_bps: float | None = None


@dataclass(frozen=True, slots=True)
class ExecutionSchedule:
    """Complete slicing plan for a parent order.

    Attributes:
        symbol: Asset identifier.
        side: ``"buy"`` or ``"sell"``.
        total_quantity: Total shares to execute.
        algorithm: Name of the algorithm that produced this schedule.
        duration_seconds: Total time window for execution.
        slices: Ordered list of child-order slices.
        n_slices: Number of slices.
        estimated_impact_bps: Pre-trade market impact estimate (basis
            points) if volume and volatility were provided, else 0.
    """

    symbol: str
    side: str
    total_quantity: float
    algorithm: str
    duration_seconds: float
    slices: list[OrderSlice]
    n_slices: int
    estimated_impact_bps: float = 0.0

    @property
    def avg_slice_quantity(self) -> float:
        """Mean quantity per slice."""
        return self.total_quantity / self.n_slices if self.n_slices > 0 else 0.0

    @property
    def slice_interval_seconds(self) -> float:
        """Average time between slices."""
        return (
            self.duration_seconds / self.n_slices if self.n_slices > 0 else 0.0
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ExecutionAlgorithm(abc.ABC):
    """Base class for execution algorithms.

    Subclasses implement :meth:`schedule` to produce an
    :class:`ExecutionSchedule` from order parameters.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Algorithm identifier."""

    @abc.abstractmethod
    def schedule(
        self,
        symbol: str,
        side: str,
        quantity: float,
        duration_seconds: float,
        *,
        daily_volume: float | None = None,
        volatility: float | None = None,
    ) -> ExecutionSchedule:
        """Generate a child-order schedule for a parent order.

        Args:
            symbol: Asset identifier.
            side: ``"buy"`` or ``"sell"``.
            quantity: Total shares/units to execute.
            duration_seconds: Time window over which to spread execution.
            daily_volume: Estimated average daily volume (for impact calc).
            volatility: Annualised volatility (for impact calc).

        Returns:
            :class:`ExecutionSchedule` with child slices.
        """


# ---------------------------------------------------------------------------
# TWAP
# ---------------------------------------------------------------------------


class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price execution algorithm.

    Splits the parent order into ``n_slices`` equal-sized child orders
    submitted at evenly spaced time intervals over ``duration_seconds``.

    Args:
        n_slices: Number of child orders (default 10).
        limit_offset_bps: Optional limit price offset per slice.
    """

    def __init__(
        self,
        n_slices: int = 10,
        limit_offset_bps: float | None = None,
    ) -> None:
        if n_slices < 1:
            raise ValueError("n_slices must be >= 1")
        self._n_slices = n_slices
        self._limit_offset_bps = limit_offset_bps

    @property
    def name(self) -> str:
        return "twap"

    def schedule(
        self,
        symbol: str,
        side: str,
        quantity: float,
        duration_seconds: float,
        *,
        daily_volume: float | None = None,
        volatility: float | None = None,
    ) -> ExecutionSchedule:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")

        n = min(self._n_slices, max(1, int(quantity)))  # no more slices than shares
        slice_qty = quantity / n
        interval = duration_seconds / n

        slices: list[OrderSlice] = []
        for i in range(n):
            # Last slice absorbs any rounding remainder
            qty = quantity - slice_qty * (n - 1) if i == n - 1 else slice_qty
            slices.append(
                OrderSlice(
                    sequence=i,
                    scheduled_seconds=i * interval,
                    quantity=qty,
                    pct_of_parent=qty / quantity,
                    limit_offset_bps=self._limit_offset_bps,
                )
            )

        impact = estimate_market_impact(
            quantity, daily_volume, volatility
        )

        return ExecutionSchedule(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            algorithm=self.name,
            duration_seconds=duration_seconds,
            slices=slices,
            n_slices=n,
            estimated_impact_bps=impact,
        )


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

# Default US equity intraday volume profile (30-minute buckets, 09:30–16:00)
# Fractions of daily volume per bucket — characteristic U-shape
_DEFAULT_VOLUME_PROFILE: list[float] = [
    0.10,  # 09:30–10:00  (opening)
    0.08,  # 10:00–10:30
    0.07,  # 10:30–11:00
    0.06,  # 11:00–11:30
    0.05,  # 11:30–12:00
    0.05,  # 12:00–12:30  (midday lull)
    0.05,  # 12:30–13:00
    0.06,  # 13:00–13:30
    0.07,  # 13:30–14:00
    0.08,  # 14:00–14:30
    0.09,  # 14:30–15:00
    0.12,  # 15:00–15:30
    0.12,  # 15:30–16:00  (closing)
]


class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume-Weighted Average Price execution algorithm.

    Sizes each child order proportional to a historical intraday volume
    profile so that the execution tracks the market's natural volume
    pattern.  This minimises temporary market impact because each slice
    represents a constant fraction of the market's volume rather than a
    constant share count.

    Args:
        n_slices: Number of child orders.  If ``volume_profile`` is
            provided, ``n_slices`` is capped at its length.
        volume_profile: List of floats representing the fraction of daily
            volume in each time bucket.  Need not sum to 1 — normalised
            internally.  If *None*, uses a default US equity intraday
            profile (13 half-hour buckets, 09:30–16:00).
        limit_offset_bps: Optional limit price offset per slice.
    """

    def __init__(
        self,
        n_slices: int = 10,
        volume_profile: list[float] | None = None,
        limit_offset_bps: float | None = None,
    ) -> None:
        if n_slices < 1:
            raise ValueError("n_slices must be >= 1")
        self._n_slices = n_slices
        self._raw_profile = volume_profile
        self._limit_offset_bps = limit_offset_bps

    @property
    def name(self) -> str:
        return "vwap"

    def schedule(
        self,
        symbol: str,
        side: str,
        quantity: float,
        duration_seconds: float,
        *,
        daily_volume: float | None = None,
        volatility: float | None = None,
    ) -> ExecutionSchedule:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")

        profile = self._resolve_profile()
        n = min(self._n_slices, len(profile), max(1, int(quantity)))

        # Normalise the profile slice we'll use
        profile_slice = profile[:n]
        total_vol = sum(profile_slice)
        weights = [1.0 / n] * n if total_vol <= 0 else [v / total_vol for v in profile_slice]

        interval = duration_seconds / n

        slices: list[OrderSlice] = []
        allocated = 0.0
        for i in range(n):
            if i == n - 1:
                qty = quantity - allocated  # remainder
            else:
                qty = round(quantity * weights[i], 6)
                allocated += qty

            slices.append(
                OrderSlice(
                    sequence=i,
                    scheduled_seconds=i * interval,
                    quantity=qty,
                    pct_of_parent=qty / quantity if quantity > 0 else 0.0,
                    limit_offset_bps=self._limit_offset_bps,
                )
            )

        impact = estimate_market_impact(
            quantity, daily_volume, volatility
        )

        return ExecutionSchedule(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            algorithm=self.name,
            duration_seconds=duration_seconds,
            slices=slices,
            n_slices=n,
            estimated_impact_bps=impact,
        )

    def _resolve_profile(self) -> list[float]:
        """Return the volume profile, using default if not configured."""
        if self._raw_profile is not None:
            return list(self._raw_profile)
        return list(_DEFAULT_VOLUME_PROFILE)


# ---------------------------------------------------------------------------
# Market impact estimation
# ---------------------------------------------------------------------------


def estimate_market_impact(
    quantity: float,
    daily_volume: float | None,
    volatility: float | None,
    eta: float = 0.5,
) -> float:
    """Estimate temporary market impact using the square-root model.

    The Almgren–Chriss square-root model estimates temporary price impact
    as::

        impact = eta * sigma * sqrt(q / V)

    where:
      - ``eta`` = market impact coefficient (default 0.5)
      - ``sigma`` = annualised volatility
      - ``q`` = order quantity (shares)
      - ``V`` = average daily volume (shares)

    Returns the estimated impact in **basis points** (1 bp = 0.01%).
    Returns 0.0 if volume or volatility data is missing.

    Args:
        quantity: Total order quantity.
        daily_volume: Estimated average daily volume.
        volatility: Annualised volatility (e.g. 0.25 for 25%).
        eta: Market impact coefficient. Default 0.5.

    Returns:
        Estimated impact in basis points.
    """
    if daily_volume is None or volatility is None:
        return 0.0
    if daily_volume <= 0 or volatility <= 0 or quantity <= 0:
        return 0.0

    # Daily vol = annual vol / sqrt(252)
    daily_vol = volatility / math.sqrt(252)

    # Square-root model
    participation = quantity / daily_volume
    impact_fraction = eta * daily_vol * math.sqrt(participation)

    # Convert to basis points
    return impact_fraction * 10_000
