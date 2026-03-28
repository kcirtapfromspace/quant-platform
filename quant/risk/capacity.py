"""Strategy capacity estimation.

Estimates the maximum AUM a strategy can support before market impact
costs erode the expected alpha.  Uses the square-root market impact model
(``cost ~ η * σ * sqrt(participation_rate)``) to project how total
execution costs scale with portfolio size.

The **capacity frontier** is the AUM at which net-of-cost alpha drops
to zero — the strategy's break-even size.

Key outputs:

  * **Capacity estimate**: AUM at which expected net alpha = 0.
  * **Impact curve**: projected total impact cost vs. AUM.
  * **Net alpha curve**: alpha minus impact cost vs. AUM.
  * **Utilisation**: current AUM / capacity as a percentage.

Usage::

    from quant.risk.capacity import CapacityEstimator, CapacityConfig

    estimator = CapacityEstimator()
    result = estimator.estimate(
        gross_alpha_bps=50.0,
        avg_turnover=0.30,
        n_assets=20,
        avg_daily_volume=50_000_000,
        current_aum=100_000_000,
    )
    print(result.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CapacityConfig:
    """Configuration for capacity estimation.

    Attributes:
        impact_coefficient: Market impact coefficient (η).  Typical 0.1
                            for liquid US large-cap equities.
        impact_exponent:    Participation rate exponent (β).  Square-root
                            model uses 0.5.
        annualised_vol:     Assumed annualised asset volatility (default 20%).
        spread_bps:         Average half-spread in basis points.
        commission_bps:     Commission cost in basis points.
        rebalances_per_year: Number of rebalance events per year.
        aum_grid_points:    Number of points to evaluate on the AUM grid.
        max_aum_multiple:   Maximum AUM to evaluate as multiple of current.
    """

    impact_coefficient: float = 0.10
    impact_exponent: float = 0.50
    annualised_vol: float = 0.20
    spread_bps: float = 5.0
    commission_bps: float = 2.0
    rebalances_per_year: int = 12
    aum_grid_points: int = 100
    max_aum_multiple: float = 20.0


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CapacityPoint:
    """One point on the capacity curve."""

    aum: float
    impact_cost_bps: float
    spread_cost_bps: float
    commission_cost_bps: float
    total_cost_bps: float
    net_alpha_bps: float
    participation_rate: float


@dataclass
class CapacityResult:
    """Strategy capacity estimation results."""

    # Inputs
    gross_alpha_bps: float
    avg_turnover: float
    n_assets: int
    avg_daily_volume: float
    current_aum: float

    # Capacity estimate
    capacity_aum: float  # AUM where net_alpha = 0
    capacity_utilisation: float  # current_aum / capacity_aum
    current_net_alpha_bps: float  # Net alpha at current AUM
    current_total_cost_bps: float  # Total cost at current AUM

    # Curves
    capacity_curve: list[CapacityPoint] = field(repr=False)

    def summary(self) -> str:
        """Return a human-readable summary."""
        cap_str = f"${self.capacity_aum:,.0f}" if math.isfinite(self.capacity_aum) else "unlimited"
        lines = [
            "Strategy Capacity Estimate",
            "=" * 55,
            "",
            f"Gross alpha (ann.)    : {self.gross_alpha_bps:.1f} bps",
            f"Avg turnover / rebal  : {self.avg_turnover:.1%}",
            f"N assets              : {self.n_assets}",
            f"Avg daily volume      : ${self.avg_daily_volume:,.0f}",
            "",
            f"Current AUM           : ${self.current_aum:,.0f}",
            f"Current cost (ann.)   : {self.current_total_cost_bps:.1f} bps",
            f"Current net alpha     : {self.current_net_alpha_bps:+.1f} bps",
            "",
            f"Capacity (break-even) : {cap_str}",
            f"Utilisation           : {self.capacity_utilisation:.1%}",
        ]

        # Show a few capacity curve points
        if self.capacity_curve:
            lines.extend(["", "Capacity Curve (sample):", "-" * 55])
            lines.append(
                f"  {'AUM ($M)':>10s} {'Impact':>8s} {'Spread':>8s} "
                f"{'Total':>8s} {'Net α':>8s} {'Part%':>7s}"
            )
            step = max(1, len(self.capacity_curve) // 8)
            for i in range(0, len(self.capacity_curve), step):
                pt = self.capacity_curve[i]
                lines.append(
                    f"  {pt.aum / 1e6:>10.1f} {pt.impact_cost_bps:>7.1f} "
                    f"{pt.spread_cost_bps:>7.1f} {pt.total_cost_bps:>7.1f} "
                    f"{pt.net_alpha_bps:>+7.1f} {pt.participation_rate:>6.2%}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class CapacityEstimator:
    """Strategy capacity estimator using the square-root impact model."""

    def __init__(self, config: CapacityConfig | None = None) -> None:
        self._config = config or CapacityConfig()

    def estimate(
        self,
        gross_alpha_bps: float,
        avg_turnover: float,
        n_assets: int,
        avg_daily_volume: float,
        current_aum: float,
    ) -> CapacityResult:
        """Estimate strategy capacity.

        Args:
            gross_alpha_bps:   Expected gross alpha in annualised basis points.
            avg_turnover:      Average one-way turnover per rebalance (0-1).
            n_assets:          Number of assets in the portfolio.
            avg_daily_volume:  Average daily dollar volume per asset.
            current_aum:       Current AUM in dollars.

        Returns:
            :class:`CapacityResult` with capacity estimate and curves.
        """
        cfg = self._config

        if n_assets <= 0 or avg_daily_volume <= 0 or avg_turnover <= 0:
            return self._empty_result(
                gross_alpha_bps, avg_turnover, n_assets,
                avg_daily_volume, current_aum,
            )

        # Build AUM grid
        max_aum = max(current_aum * cfg.max_aum_multiple, current_aum + 1)
        aum_grid = np.linspace(
            max(current_aum * 0.1, 1_000_000),
            max_aum,
            cfg.aum_grid_points,
        )

        curve: list[CapacityPoint] = []
        capacity_aum = float("inf")
        current_pt: CapacityPoint | None = None

        for aum in aum_grid:
            pt = self._evaluate_at_aum(
                aum, gross_alpha_bps, avg_turnover, n_assets,
                avg_daily_volume, cfg,
            )
            curve.append(pt)

            if pt.net_alpha_bps <= 0 and capacity_aum == float("inf"):
                # Interpolate to find break-even
                if len(curve) >= 2 and curve[-2].net_alpha_bps > 0:
                    prev = curve[-2]
                    frac = prev.net_alpha_bps / (prev.net_alpha_bps - pt.net_alpha_bps)
                    capacity_aum = prev.aum + frac * (pt.aum - prev.aum)
                else:
                    capacity_aum = aum

        # Evaluate at current AUM
        current_pt = self._evaluate_at_aum(
            current_aum, gross_alpha_bps, avg_turnover, n_assets,
            avg_daily_volume, cfg,
        )

        utilisation = current_aum / capacity_aum if math.isfinite(capacity_aum) and capacity_aum > 0 else 0.0

        return CapacityResult(
            gross_alpha_bps=gross_alpha_bps,
            avg_turnover=avg_turnover,
            n_assets=n_assets,
            avg_daily_volume=avg_daily_volume,
            current_aum=current_aum,
            capacity_aum=capacity_aum,
            capacity_utilisation=utilisation,
            current_net_alpha_bps=current_pt.net_alpha_bps,
            current_total_cost_bps=current_pt.total_cost_bps,
            capacity_curve=curve,
        )

    @staticmethod
    def _evaluate_at_aum(
        aum: float,
        gross_alpha_bps: float,
        avg_turnover: float,
        n_assets: int,
        avg_daily_volume: float,
        cfg: CapacityConfig,
    ) -> CapacityPoint:
        """Compute costs and net alpha for a given AUM."""
        # Per-asset trade notional per rebalance
        trade_per_asset = (aum * avg_turnover) / n_assets

        # Participation rate: what fraction of daily volume is our trade?
        participation = trade_per_asset / avg_daily_volume if avg_daily_volume > 0 else 0.0

        # Market impact (annualised bps):
        # impact_per_rebal = η * σ_daily * sqrt(participation) in return units
        # σ_daily = σ_annual / sqrt(252)
        daily_vol = cfg.annualised_vol / math.sqrt(252)
        impact_per_rebal = (
            cfg.impact_coefficient * daily_vol * (participation ** cfg.impact_exponent)
        )
        # Convert to annualised bps
        impact_bps = impact_per_rebal * cfg.rebalances_per_year * 10_000

        # Spread cost (annualised bps)
        spread_bps = cfg.spread_bps * avg_turnover * cfg.rebalances_per_year

        # Commission cost (annualised bps)
        commission_bps = cfg.commission_bps * avg_turnover * cfg.rebalances_per_year

        total_cost = impact_bps + spread_bps + commission_bps
        net_alpha = gross_alpha_bps - total_cost

        return CapacityPoint(
            aum=aum,
            impact_cost_bps=impact_bps,
            spread_cost_bps=spread_bps,
            commission_cost_bps=commission_bps,
            total_cost_bps=total_cost,
            net_alpha_bps=net_alpha,
            participation_rate=participation,
        )

    @staticmethod
    def _empty_result(
        gross_alpha_bps: float,
        avg_turnover: float,
        n_assets: int,
        avg_daily_volume: float,
        current_aum: float,
    ) -> CapacityResult:
        """Return an empty result when inputs are invalid."""
        return CapacityResult(
            gross_alpha_bps=gross_alpha_bps,
            avg_turnover=avg_turnover,
            n_assets=n_assets,
            avg_daily_volume=avg_daily_volume,
            current_aum=current_aum,
            capacity_aum=0.0,
            capacity_utilisation=0.0,
            current_net_alpha_bps=0.0,
            current_total_cost_bps=0.0,
            capacity_curve=[],
        )
