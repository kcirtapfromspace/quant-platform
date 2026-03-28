"""Execution quality analysis (Transaction Cost Analysis).

Measures how well trades were executed by comparing actual fill prices
against standard benchmarks.  Helps the CIO evaluate broker performance,
detect execution leakage, and optimise algo selection.

Benchmarks:

  * **Arrival price**: Price at the time the order was placed.
  * **VWAP**: Volume-weighted average price over the execution window.
  * **TWAP**: Time-weighted average price over the execution window.
  * **Implementation shortfall**: Full round-trip cost including market
    impact, timing, and opportunity cost.

Usage::

    from quant.execution.quality import (
        ExecutionAnalyzer,
        Fill,
    )

    analyzer = ExecutionAnalyzer()
    result = analyzer.analyze([
        Fill(symbol="AAPL", side="BUY", quantity=1000,
             fill_price=150.20, arrival_price=150.00,
             vwap=150.10, notional=150200),
    ])
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fill:
    """A single trade fill for TCA analysis.

    Attributes:
        symbol:        Asset identifier.
        side:          "BUY" or "SELL".
        quantity:      Number of shares filled.
        fill_price:    Average fill price.
        arrival_price: Price at order submission (decision price).
        vwap:          Volume-weighted average price during execution window.
        notional:      Total fill notional (quantity × fill_price).
        twap:          Time-weighted average price (optional, defaults to VWAP).
        broker:        Broker or algo identifier (optional).
    """

    symbol: str
    side: str
    quantity: float
    fill_price: float
    arrival_price: float
    vwap: float
    notional: float
    twap: float | None = None
    broker: str | None = None


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FillAnalysis:
    """Per-fill execution quality metrics.

    Attributes:
        symbol:             Asset.
        side:               BUY or SELL.
        notional:           Fill notional.
        slippage_bps:       Slippage vs arrival price (positive = cost).
        vs_vwap_bps:        Performance vs VWAP (positive = worse than VWAP).
        vs_twap_bps:        Performance vs TWAP.
        impl_shortfall_bps: Implementation shortfall in bps.
        broker:             Broker/algo identifier.
    """

    symbol: str
    side: str
    notional: float
    slippage_bps: float
    vs_vwap_bps: float
    vs_twap_bps: float
    impl_shortfall_bps: float
    broker: str | None


@dataclass
class BrokerStats:
    """Aggregate execution stats per broker.

    Attributes:
        broker:               Broker name.
        n_fills:              Number of fills.
        total_notional:       Total fill notional.
        avg_slippage_bps:     Notional-weighted avg slippage vs arrival.
        avg_vs_vwap_bps:      Notional-weighted avg performance vs VWAP.
        avg_impl_shortfall:   Notional-weighted avg implementation shortfall.
    """

    broker: str
    n_fills: int
    total_notional: float
    avg_slippage_bps: float
    avg_vs_vwap_bps: float
    avg_impl_shortfall: float


@dataclass
class ExecutionQualityResult:
    """Complete execution quality analysis.

    Attributes:
        fill_analyses:        Per-fill metrics.
        broker_stats:         Per-broker aggregates.
        n_fills:              Total number of fills.
        total_notional:       Total notional across all fills.
        avg_slippage_bps:     Notional-weighted average slippage.
        avg_vs_vwap_bps:      Notional-weighted average vs VWAP.
        avg_impl_shortfall:   Notional-weighted average implementation shortfall.
        total_cost_dollars:   Total execution cost in dollars.
        pct_fills_beat_vwap:  Fraction of fills that beat VWAP.
    """

    fill_analyses: list[FillAnalysis] = field(repr=False)
    broker_stats: list[BrokerStats] = field(default_factory=list)
    n_fills: int = 0
    total_notional: float = 0.0
    avg_slippage_bps: float = 0.0
    avg_vs_vwap_bps: float = 0.0
    avg_impl_shortfall: float = 0.0
    total_cost_dollars: float = 0.0
    pct_fills_beat_vwap: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Execution Quality Report ({self.n_fills} fills)",
            "=" * 65,
            "",
            f"Total notional          : ${self.total_notional:,.0f}",
            f"Total cost              : ${self.total_cost_dollars:,.0f}",
            "",
            f"Avg slippage (arrival)  : {self.avg_slippage_bps:+.2f} bps",
            f"Avg vs VWAP             : {self.avg_vs_vwap_bps:+.2f} bps",
            f"Avg impl shortfall      : {self.avg_impl_shortfall:+.2f} bps",
            f"Fills beating VWAP      : {self.pct_fills_beat_vwap:.1%}",
        ]

        if self.broker_stats:
            lines.extend(["", "Broker Comparison:", "-" * 65])
            lines.append(
                f"  {'Broker':<15s} {'Fills':>6s} {'Notional':>14s} "
                f"{'Slip':>7s} {'vsVWAP':>7s} {'IS':>7s}"
            )
            for bs in sorted(self.broker_stats, key=lambda x: x.avg_slippage_bps):
                lines.append(
                    f"  {bs.broker:<15s} {bs.n_fills:>6d} "
                    f"${bs.total_notional:>13,.0f} "
                    f"{bs.avg_slippage_bps:>+6.1f} "
                    f"{bs.avg_vs_vwap_bps:>+6.1f} "
                    f"{bs.avg_impl_shortfall:>+6.1f}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class ExecutionAnalyzer:
    """Execution quality / TCA analyzer."""

    def analyze(self, fills: list[Fill]) -> ExecutionQualityResult:
        """Analyze execution quality for a set of fills.

        Args:
            fills: List of trade fills to analyze.

        Returns:
            :class:`ExecutionQualityResult` with per-fill and aggregate metrics.
        """
        if not fills:
            return ExecutionQualityResult(fill_analyses=[])

        analyses: list[FillAnalysis] = []
        for fill in fills:
            analyses.append(self._analyze_fill(fill))

        total_notional = sum(fa.notional for fa in analyses)

        # Notional-weighted averages
        if total_notional > 0:
            avg_slip = sum(
                fa.slippage_bps * fa.notional for fa in analyses
            ) / total_notional
            avg_vwap = sum(
                fa.vs_vwap_bps * fa.notional for fa in analyses
            ) / total_notional
            avg_is = sum(
                fa.impl_shortfall_bps * fa.notional for fa in analyses
            ) / total_notional
        else:
            avg_slip = avg_vwap = avg_is = 0.0

        total_cost = avg_slip / 10_000 * total_notional

        # VWAP beat rate
        n_beat = sum(1 for fa in analyses if fa.vs_vwap_bps < 0)
        pct_beat = n_beat / len(analyses) if analyses else 0.0

        # Broker aggregation
        broker_map: dict[str, list[FillAnalysis]] = {}
        for fa in analyses:
            key = fa.broker or "unknown"
            broker_map.setdefault(key, []).append(fa)

        broker_stats: list[BrokerStats] = []
        for broker, fas in broker_map.items():
            b_notional = sum(f.notional for f in fas)
            if b_notional > 0:
                b_slip = sum(f.slippage_bps * f.notional for f in fas) / b_notional
                b_vwap = sum(f.vs_vwap_bps * f.notional for f in fas) / b_notional
                b_is = sum(f.impl_shortfall_bps * f.notional for f in fas) / b_notional
            else:
                b_slip = b_vwap = b_is = 0.0
            broker_stats.append(
                BrokerStats(
                    broker=broker,
                    n_fills=len(fas),
                    total_notional=b_notional,
                    avg_slippage_bps=b_slip,
                    avg_vs_vwap_bps=b_vwap,
                    avg_impl_shortfall=b_is,
                )
            )

        return ExecutionQualityResult(
            fill_analyses=analyses,
            broker_stats=broker_stats,
            n_fills=len(analyses),
            total_notional=total_notional,
            avg_slippage_bps=avg_slip,
            avg_vs_vwap_bps=avg_vwap,
            avg_impl_shortfall=avg_is,
            total_cost_dollars=total_cost,
            pct_fills_beat_vwap=pct_beat,
        )

    @staticmethod
    def _analyze_fill(fill: Fill) -> FillAnalysis:
        """Compute execution quality metrics for a single fill."""
        # Direction sign: BUY => positive cost when fill > benchmark
        #                 SELL => positive cost when fill < benchmark
        sign = 1.0 if fill.side.upper() == "BUY" else -1.0

        # Slippage vs arrival price
        if fill.arrival_price > 0:
            slippage_bps = sign * (fill.fill_price - fill.arrival_price) / fill.arrival_price * 10_000
        else:
            slippage_bps = 0.0

        # Performance vs VWAP
        if fill.vwap > 0:
            vs_vwap_bps = sign * (fill.fill_price - fill.vwap) / fill.vwap * 10_000
        else:
            vs_vwap_bps = 0.0

        # Performance vs TWAP
        twap = fill.twap if fill.twap is not None else fill.vwap
        vs_twap_bps = sign * (fill.fill_price - twap) / twap * 10_000 if twap > 0 else 0.0

        # Implementation shortfall: slippage is the main component
        # In a full model this would include delay cost and opportunity cost;
        # here we use arrival-price slippage as the shortfall proxy.
        impl_shortfall_bps = slippage_bps

        return FillAnalysis(
            symbol=fill.symbol,
            side=fill.side,
            notional=fill.notional,
            slippage_bps=slippage_bps,
            vs_vwap_bps=vs_vwap_bps,
            vs_twap_bps=vs_twap_bps,
            impl_shortfall_bps=impl_shortfall_bps,
            broker=fill.broker,
        )
