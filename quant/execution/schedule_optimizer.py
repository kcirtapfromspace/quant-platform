"""Optimal execution scheduling (Almgren-Chriss).

Computes the optimal trade schedule that minimises expected implementation
shortfall (market impact + opportunity cost) over a discrete multi-period
horizon.

Model:

  * **Temporary impact**: proportional to trade rate.
  * **Permanent impact**: proportional to cumulative traded quantity.
  * **Risk penalty**: quadratic penalty on variance of shortfall.

The closed-form solution trades more aggressively at the start when
risk aversion is high, and evenly (TWAP) when risk aversion is zero.

Key outputs:

  * **Trade schedule**: fraction of total order to execute each period.
  * **Expected cost**: total expected implementation shortfall in bps.
  * **Cost breakdown**: temporary impact, permanent impact, risk cost.
  * **Efficient frontier**: cost vs risk trade-off for different urgency.

Usage::

    from quant.execution.schedule_optimizer import (
        ScheduleOptimizer,
        ScheduleConfig,
    )

    optimizer = ScheduleOptimizer(ScheduleConfig(
        n_periods=10,
        risk_aversion=1e-6,
    ))
    result = optimizer.optimize(
        total_shares=100_000,
        daily_volume=1_000_000,
        volatility_bps=150,
        spread_bps=5.0,
    )
    print(result.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ScheduleConfig:
    """Configuration for execution schedule optimisation.

    Attributes:
        n_periods:       Number of trading periods over which to execute.
        risk_aversion:   Risk-aversion parameter λ (higher = more front-loaded).
                         Set to 0 for pure cost minimisation (TWAP).
        temp_impact:     Temporary impact coefficient η (bps per unit
                         participation rate).
        perm_impact:     Permanent impact coefficient γ (bps per unit
                         participation rate).
    """

    n_periods: int = 10
    risk_aversion: float = 1e-6
    temp_impact: float = 0.1
    perm_impact: float = 0.05


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Cost decomposition of the execution schedule.

    Attributes:
        temporary_bps:   Expected temporary impact cost (bps).
        permanent_bps:   Expected permanent impact cost (bps).
        risk_bps:        Risk penalty (variance-based, bps).
        total_bps:       Total expected cost (bps).
    """

    temporary_bps: float
    permanent_bps: float
    risk_bps: float
    total_bps: float


@dataclass
class ScheduleResult:
    """Optimal execution schedule result.

    Attributes:
        trade_fractions:  Fraction of total order per period (sums to 1).
        trade_shares:     Shares to trade per period.
        remaining:        Remaining inventory after each period.
        cost:             Cost breakdown.
        participation:    Average participation rate across periods.
        n_periods:        Number of trading periods.
        total_shares:     Total shares to execute.
    """

    trade_fractions: np.ndarray
    trade_shares: np.ndarray
    remaining: np.ndarray
    cost: CostBreakdown
    participation: float
    n_periods: int
    total_shares: float

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Optimal Execution Schedule ({self.n_periods} periods)",
            "=" * 60,
            "",
            f"Total shares   : {self.total_shares:,.0f}",
            f"Avg particip.  : {self.participation:.2%}",
            "",
            "Cost breakdown:",
            f"  Temporary    : {self.cost.temporary_bps:.2f} bps",
            f"  Permanent    : {self.cost.permanent_bps:.2f} bps",
            f"  Risk penalty : {self.cost.risk_bps:.2f} bps",
            f"  Total        : {self.cost.total_bps:.2f} bps",
            "",
            "Schedule (period → trade fraction):",
        ]
        for i, frac in enumerate(self.trade_fractions):
            bar = "#" * int(frac * 40)
            lines.append(f"  {i + 1:>3d}: {frac:6.2%}  {bar}")
        return "\n".join(lines)


@dataclass
class FrontierPoint:
    """Single point on the cost-risk efficient frontier.

    Attributes:
        risk_aversion:  λ parameter used.
        expected_cost:  Expected shortfall cost (bps).
        risk:           Standard deviation of shortfall (bps).
    """

    risk_aversion: float
    expected_cost: float
    risk: float


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class ScheduleOptimizer:
    """Almgren-Chriss optimal execution schedule.

    Args:
        config: Schedule configuration.
    """

    def __init__(self, config: ScheduleConfig | None = None) -> None:
        self._config = config or ScheduleConfig()

    @property
    def config(self) -> ScheduleConfig:
        return self._config

    def optimize(
        self,
        total_shares: float,
        daily_volume: float,
        volatility_bps: float,
        spread_bps: float = 5.0,
    ) -> ScheduleResult:
        """Compute the optimal trade schedule.

        Args:
            total_shares:   Total shares to execute.
            daily_volume:   Average daily volume in shares.
            volatility_bps: Daily volatility of the asset in bps.
            spread_bps:     Half-spread cost in bps.

        Returns:
            :class:`ScheduleResult` with optimal trade fractions and costs.

        Raises:
            ValueError: If ``total_shares`` or ``daily_volume`` are non-positive.
        """
        if total_shares <= 0:
            raise ValueError(f"total_shares must be positive, got {total_shares}")
        if daily_volume <= 0:
            raise ValueError(f"daily_volume must be positive, got {daily_volume}")

        cfg = self._config
        n = cfg.n_periods
        lam = cfg.risk_aversion
        eta = cfg.temp_impact
        gamma = cfg.perm_impact
        sigma = volatility_bps

        # Almgren-Chriss: optimal trajectory
        # κ = sqrt(λσ² / η)  (urgency parameter)
        kappa = math.sqrt(lam * sigma ** 2 / eta) if (eta > 1e-15 and lam > 0 and sigma > 0) else 0.0

        # Inventory trajectory: x_j / X = sinh(κ(N-j)) / sinh(κN)
        # For κ→0 this reduces to (N-j)/N (TWAP)
        remaining_frac = np.zeros(n + 1)
        remaining_frac[0] = 1.0  # full inventory at start

        if kappa > 1e-10:
            kn = kappa * n
            if kn > 500:
                # Large κN: sinh ratio ≈ exp(-κj)
                for j in range(1, n + 1):
                    remaining_frac[j] = math.exp(-kappa * j)
            else:
                sinh_kn = math.sinh(kn)
                if abs(sinh_kn) > 1e-15:
                    for j in range(1, n + 1):
                        remaining_frac[j] = math.sinh(kappa * (n - j)) / sinh_kn
                else:
                    for j in range(1, n + 1):
                        remaining_frac[j] = (n - j) / n
        else:
            # TWAP: linear decay
            for j in range(1, n + 1):
                remaining_frac[j] = (n - j) / n

        # Trade fractions: n_j / X = x_{j-1} - x_j
        trade_frac = np.diff(-remaining_frac)  # positive trades
        trade_frac = np.maximum(trade_frac, 0.0)

        # Normalise to sum to 1 (handle floating point)
        total_frac = trade_frac.sum()
        if total_frac > 1e-15:
            trade_frac /= total_frac

        trade_shares = trade_frac * total_shares
        remaining_shares = remaining_frac[1:] * total_shares

        # Average participation rate
        avg_participation = total_shares / (daily_volume * n)

        # Cost estimation
        cost = self._estimate_cost(
            trade_frac, total_shares, daily_volume, sigma, spread_bps,
            eta, gamma, lam, n,
        )

        return ScheduleResult(
            trade_fractions=trade_frac,
            trade_shares=trade_shares,
            remaining=remaining_shares,
            cost=cost,
            participation=avg_participation,
            n_periods=n,
            total_shares=total_shares,
        )

    def efficient_frontier(
        self,
        total_shares: float,
        daily_volume: float,
        volatility_bps: float,
        spread_bps: float = 5.0,
        n_points: int = 20,
    ) -> list[FrontierPoint]:
        """Compute the cost-risk efficient frontier.

        Sweeps risk aversion from 0 (TWAP) to a high value (aggressive
        front-loading) and returns the cost-risk pairs.

        Args:
            total_shares:   Total shares to execute.
            daily_volume:   Average daily volume.
            volatility_bps: Daily asset volatility in bps.
            spread_bps:     Half-spread in bps.
            n_points:       Number of frontier points.

        Returns:
            List of :class:`FrontierPoint` ordered by risk aversion.
        """
        cfg = self._config
        n = cfg.n_periods
        eta = cfg.temp_impact
        sigma = volatility_bps

        # Log-spaced risk aversion levels
        lam_min = 1e-10
        lam_max = 10.0 * eta / max(sigma ** 2, 1e-10)
        if lam_max <= lam_min:
            lam_max = 1.0

        lam_values = np.logspace(
            math.log10(lam_min), math.log10(lam_max), n_points,
        )

        points = []
        for lam in lam_values:
            opt = ScheduleOptimizer(ScheduleConfig(
                n_periods=n,
                risk_aversion=float(lam),
                temp_impact=eta,
                perm_impact=cfg.perm_impact,
            ))
            result = opt.optimize(
                total_shares, daily_volume, volatility_bps, spread_bps,
            )
            # Risk: σ of shortfall ≈ σ · sqrt(Σ x_j²) where x_j = remaining
            # Using the remaining inventory
            remaining_frac = np.concatenate(
                [[1.0], result.remaining / total_shares],
            )
            risk_var = sigma ** 2 * np.sum(remaining_frac[1:] ** 2)
            risk_std = math.sqrt(max(risk_var, 0.0))

            points.append(FrontierPoint(
                risk_aversion=float(lam),
                expected_cost=result.cost.total_bps,
                risk=risk_std,
            ))

        return points

    # ── Internal ──────────────────────────────────────────────────

    @staticmethod
    def _estimate_cost(
        trade_frac: np.ndarray,
        total_shares: float,
        daily_volume: float,
        sigma: float,
        spread_bps: float,
        eta: float,
        gamma: float,
        lam: float,
        n: int,
    ) -> CostBreakdown:
        """Estimate execution cost breakdown."""
        # Participation rates per period
        shares_per_period = trade_frac * total_shares
        participation = shares_per_period / daily_volume

        # Temporary impact: η · Σ participation_j²  (bps, weighted by trade size)
        temp_cost = eta * float(np.sum(participation * participation))
        # Scale to bps of total order
        temp_bps = temp_cost * 10000 if total_shares > 0 else 0.0

        # Simplify: temporary impact ≈ η * avg_participation * 10000
        temp_bps = eta * float(np.sum(participation ** 2)) * 10000

        # Permanent impact: γ · total_participation · 10000
        total_participation = total_shares / daily_volume
        perm_bps = gamma * total_participation * 10000

        # Spread cost
        spread_total = spread_bps

        # Risk cost: λ · σ² · Σ remaining_frac²
        remaining = 1.0 - np.cumsum(trade_frac)
        risk_cost = lam * sigma ** 2 * float(np.sum(remaining ** 2))

        total_bps = temp_bps + perm_bps + spread_total + risk_cost

        return CostBreakdown(
            temporary_bps=temp_bps,
            permanent_bps=perm_bps + spread_total,
            risk_bps=risk_cost,
            total_bps=total_bps,
        )
