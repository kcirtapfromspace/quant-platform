"""Liquidity risk monitoring and days-to-liquidate analysis.

Measures portfolio liquidity risk across multiple dimensions to help the
CIO understand tail-event liquidation exposure:

  * **Days to liquidate (DTL)**: How many trading days to exit each position
    at a given participation rate limit.
  * **Liquidity-at-Risk (LaR)**: Expected market-impact cost of liquidating
    the portfolio under normal *and* stressed volume conditions.
  * **Liquidity score**: Per-position and portfolio-level score (0-100) based
    on position size relative to available liquidity.
  * **Concentration risk**: Identifies positions that dominate the illiquidity
    budget.

Usage::

    from quant.risk.liquidity import LiquidityMonitor, LiquidityConfig

    monitor = LiquidityMonitor(LiquidityConfig(
        max_participation_rate=0.10,
        stress_volume_haircut=0.50,
    ))
    result = monitor.analyze(
        positions={"AAPL": 5_000_000, "ILLIQ": 2_000_000},
        adv={"AAPL": 100_000_000, "ILLIQ": 1_000_000},
        volatility={"AAPL": 0.25, "ILLIQ": 0.45},
    )
    print(result.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LiquidityConfig:
    """Configuration for liquidity risk monitoring.

    Attributes:
        max_participation_rate: Maximum fraction of ADV to trade per day.
        impact_coefficient:     Market impact coefficient (η).
        impact_exponent:        Participation rate exponent (β = 0.5 for
                                square-root model).
        stress_volume_haircut:  Fraction of normal ADV assumed available in
                                a stress scenario (e.g. 0.50 = 50% of normal).
        dtl_warning_days:       Positions taking longer than this to liquidate
                                are flagged.
        liquidity_score_cap:    Maximum DTL that maps to score 0.
    """

    max_participation_rate: float = 0.10
    impact_coefficient: float = 0.10
    impact_exponent: float = 0.50
    stress_volume_haircut: float = 0.50
    dtl_warning_days: float = 5.0
    liquidity_score_cap: float = 20.0


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionLiquidity:
    """Liquidity metrics for a single position.

    Attributes:
        symbol:             Asset identifier.
        position_value:     Dollar value of the position.
        adv:                Average daily volume in dollars.
        volatility:         Annualised volatility.
        days_to_liquidate:  Trading days to fully exit at max participation.
        days_to_liquidate_stressed: DTL under stressed volume.
        impact_cost_bps:    Expected market impact in basis points.
        impact_cost_stressed_bps: Impact under stressed volume.
        liquidity_score:    Score from 100 (most liquid) to 0 (illiquid).
        is_warning:         True if DTL exceeds warning threshold.
        pct_of_adv:         Position value as fraction of one day's ADV.
    """

    symbol: str
    position_value: float
    adv: float
    volatility: float
    days_to_liquidate: float
    days_to_liquidate_stressed: float
    impact_cost_bps: float
    impact_cost_stressed_bps: float
    liquidity_score: float
    is_warning: bool
    pct_of_adv: float


@dataclass
class LiquidityResult:
    """Portfolio-level liquidity risk analysis.

    Attributes:
        positions:                  Per-position liquidity details.
        portfolio_value:            Total portfolio value.
        portfolio_dtl:              Weighted-average DTL (normal).
        portfolio_dtl_stressed:     Weighted-average DTL (stressed).
        portfolio_lar_bps:          Liquidity-at-Risk in bps (normal).
        portfolio_lar_stressed_bps: LaR under stressed volume.
        portfolio_lar_dollars:      LaR in dollars (normal).
        portfolio_lar_stressed_dollars: LaR in dollars (stressed).
        portfolio_liquidity_score:  Weighted-average liquidity score.
        n_warnings:                 Positions exceeding DTL warning.
        worst_position:             Symbol with highest DTL.
        concentration:              Fraction of total illiquidity from top position.
    """

    positions: list[PositionLiquidity] = field(repr=False)
    portfolio_value: float = 0.0
    portfolio_dtl: float = 0.0
    portfolio_dtl_stressed: float = 0.0
    portfolio_lar_bps: float = 0.0
    portfolio_lar_stressed_bps: float = 0.0
    portfolio_lar_dollars: float = 0.0
    portfolio_lar_stressed_dollars: float = 0.0
    portfolio_liquidity_score: float = 0.0
    n_warnings: int = 0
    worst_position: str = ""
    concentration: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Liquidity Risk Monitor ({len(self.positions)} positions)",
            "=" * 65,
            "",
            f"Portfolio value           : ${self.portfolio_value:,.0f}",
            f"Portfolio DTL (normal)    : {self.portfolio_dtl:.1f} days",
            f"Portfolio DTL (stressed)  : {self.portfolio_dtl_stressed:.1f} days",
            "",
            f"Liquidity-at-Risk (normal): {self.portfolio_lar_bps:.1f} bps"
            f"  (${self.portfolio_lar_dollars:,.0f})",
            f"Liquidity-at-Risk (stress): {self.portfolio_lar_stressed_bps:.1f} bps"
            f"  (${self.portfolio_lar_stressed_dollars:,.0f})",
            "",
            f"Portfolio liquidity score : {self.portfolio_liquidity_score:.0f}/100",
            f"Warnings (DTL > threshold): {self.n_warnings}",
            f"Worst position            : {self.worst_position}",
            f"Top-position concentration: {self.concentration:.1%}",
        ]

        if self.positions:
            lines.extend(["", "Position Details:", "-" * 65])
            lines.append(
                f"  {'Symbol':<10s} {'Value ($)':>12s} {'DTL':>6s} "
                f"{'DTL-S':>6s} {'LaR':>6s} {'Score':>6s} {'Flag':>5s}"
            )
            for p in sorted(self.positions, key=lambda x: -x.days_to_liquidate):
                flag = "!!" if p.is_warning else ""
                lines.append(
                    f"  {p.symbol:<10s} {p.position_value:>12,.0f} "
                    f"{p.days_to_liquidate:>6.1f} "
                    f"{p.days_to_liquidate_stressed:>6.1f} "
                    f"{p.impact_cost_bps:>5.1f} "
                    f"{p.liquidity_score:>6.0f} {flag:>5s}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class LiquidityMonitor:
    """Portfolio liquidity risk monitor.

    Args:
        config: Liquidity monitoring parameters.
    """

    def __init__(self, config: LiquidityConfig | None = None) -> None:
        self._config = config or LiquidityConfig()

    @property
    def config(self) -> LiquidityConfig:
        return self._config

    def analyze(
        self,
        positions: dict[str, float],
        adv: dict[str, float],
        volatility: dict[str, float] | None = None,
    ) -> LiquidityResult:
        """Analyze liquidity risk for a portfolio.

        Args:
            positions:  ``{symbol: dollar_value}`` of current positions.
            adv:        ``{symbol: avg_daily_volume_dollars}`` per asset.
            volatility: ``{symbol: annualised_vol}`` per asset.  If not
                        provided, a default of 0.20 is used.

        Returns:
            :class:`LiquidityResult` with position and portfolio metrics.

        Raises:
            ValueError: If positions and ADV have no common symbols.
        """
        cfg = self._config
        volatility = volatility or {}

        if not positions:
            return LiquidityResult(positions=[])

        common = set(positions) & set(adv)
        if not common:
            raise ValueError(
                "No common symbols between positions and ADV data"
            )

        pos_details: list[PositionLiquidity] = []
        total_value = sum(abs(positions[s]) for s in common)

        for sym in sorted(common):
            pos_val = abs(positions[sym])
            sym_adv = max(adv[sym], 1.0)  # Guard against zero
            sym_vol = volatility.get(sym, 0.20)

            pl = self._analyze_position(pos_val, sym_adv, sym_vol, sym, cfg)
            pos_details.append(pl)

        # Portfolio aggregates (value-weighted)
        if total_value > 0:
            port_dtl = sum(
                p.days_to_liquidate * p.position_value / total_value
                for p in pos_details
            )
            port_dtl_stressed = sum(
                p.days_to_liquidate_stressed * p.position_value / total_value
                for p in pos_details
            )
            port_lar_dollars = sum(
                p.impact_cost_bps / 10_000 * p.position_value
                for p in pos_details
            )
            port_lar_stressed_dollars = sum(
                p.impact_cost_stressed_bps / 10_000 * p.position_value
                for p in pos_details
            )
            port_lar_bps = port_lar_dollars / total_value * 10_000
            port_lar_stressed_bps = port_lar_stressed_dollars / total_value * 10_000
            port_score = sum(
                p.liquidity_score * p.position_value / total_value
                for p in pos_details
            )
        else:
            port_dtl = port_dtl_stressed = 0.0
            port_lar_bps = port_lar_stressed_bps = 0.0
            port_lar_dollars = port_lar_stressed_dollars = 0.0
            port_score = 100.0

        n_warnings = sum(1 for p in pos_details if p.is_warning)
        worst = max(pos_details, key=lambda p: p.days_to_liquidate)

        # Concentration: fraction of total DTL-weighted illiquidity from worst
        total_dtl_weighted = sum(
            p.days_to_liquidate * p.position_value for p in pos_details
        )
        concentration = (
            worst.days_to_liquidate * worst.position_value / total_dtl_weighted
            if total_dtl_weighted > 0
            else 0.0
        )

        return LiquidityResult(
            positions=pos_details,
            portfolio_value=total_value,
            portfolio_dtl=port_dtl,
            portfolio_dtl_stressed=port_dtl_stressed,
            portfolio_lar_bps=port_lar_bps,
            portfolio_lar_stressed_bps=port_lar_stressed_bps,
            portfolio_lar_dollars=port_lar_dollars,
            portfolio_lar_stressed_dollars=port_lar_stressed_dollars,
            portfolio_liquidity_score=port_score,
            n_warnings=n_warnings,
            worst_position=worst.symbol,
            concentration=concentration,
        )

    @staticmethod
    def _analyze_position(
        position_value: float,
        adv: float,
        volatility: float,
        symbol: str,
        cfg: LiquidityConfig,
    ) -> PositionLiquidity:
        """Compute liquidity metrics for a single position."""
        # Daily tradable amount at max participation
        daily_capacity = adv * cfg.max_participation_rate
        stressed_adv = adv * cfg.stress_volume_haircut
        daily_capacity_stressed = stressed_adv * cfg.max_participation_rate

        # Days to liquidate
        dtl = position_value / daily_capacity if daily_capacity > 0 else float("inf")
        dtl_stressed = (
            position_value / daily_capacity_stressed
            if daily_capacity_stressed > 0
            else float("inf")
        )

        # Market impact: η * σ_daily * participation^β * 10000 (bps)
        daily_vol = volatility / math.sqrt(252)
        participation = min(position_value / adv, 1.0) if adv > 0 else 1.0
        impact_bps = (
            cfg.impact_coefficient
            * daily_vol
            * (participation ** cfg.impact_exponent)
            * 10_000
        )

        stressed_participation = (
            min(position_value / stressed_adv, 1.0) if stressed_adv > 0 else 1.0
        )
        impact_stressed_bps = (
            cfg.impact_coefficient
            * daily_vol
            * (stressed_participation ** cfg.impact_exponent)
            * 10_000
        )

        # Liquidity score: 100 = instant, 0 = takes >= cap days
        score = 0.0 if math.isinf(dtl) else max(0.0, 100.0 * (1.0 - dtl / cfg.liquidity_score_cap))

        pct_adv = position_value / adv if adv > 0 else float("inf")

        return PositionLiquidity(
            symbol=symbol,
            position_value=position_value,
            adv=adv,
            volatility=volatility,
            days_to_liquidate=dtl,
            days_to_liquidate_stressed=dtl_stressed,
            impact_cost_bps=impact_bps,
            impact_cost_stressed_bps=impact_stressed_bps,
            liquidity_score=score,
            is_warning=dtl >= cfg.dtl_warning_days,
            pct_of_adv=pct_adv,
        )
