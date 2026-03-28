"""Turnover-optimized rebalancing with no-trade bands and priority scheduling.

Standard rebalancing recomputes target weights and trades the full delta
every period, regardless of how small the drift or how costly the trade.
This module adds three optimisations that directly improve net P&L:

  1. **No-trade bands** — only rebalance a position when its drift from
     target exceeds a configurable tolerance.  Small drifts are noise;
     trading on them is pure cost.
  2. **Priority-weighted trade selection** — when the turnover budget is
     binding, rank candidate trades by their expected value (alpha-to-cost
     ratio, drift magnitude, or equal) and greedily include the best ones.
  3. **Partial rebalancing** — move a configurable fraction toward target
     instead of snapping fully, reducing impact cost while still reducing
     tracking error.

The optimizer sits *after* portfolio construction produces target weights
and *before* the rebalance engine generates executable orders.

Usage::

    from quant.portfolio.turnover_optimizer import (
        TurnoverOptimizer,
        TurnoverConfig,
    )

    optimizer = TurnoverOptimizer(TurnoverConfig(
        no_trade_band=0.02,
        max_turnover=0.30,
        priority_method="alpha_cost_ratio",
    ))

    result = optimizer.optimize(
        current_weights={"AAPL": 0.28, "GOOG": 0.22, "MSFT": 0.50},
        target_weights={"AAPL": 0.30, "GOOG": 0.30, "MSFT": 0.40},
        expected_alpha={"AAPL": 10, "GOOG": 25, "MSFT": 5},
        cost_estimates={"AAPL": 4, "GOOG": 3, "MSFT": 6},
    )
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PriorityMethod(Enum):
    """How to rank trades when turnover budget is binding."""

    ALPHA_COST_RATIO = "alpha_cost_ratio"
    LARGEST_DRIFT = "largest_drift"
    EQUAL = "equal"


@dataclass
class TurnoverConfig:
    """Configuration for turnover-optimized rebalancing.

    Attributes:
        no_trade_band:          Minimum absolute drift from target before a
                                position is eligible for rebalancing.
        max_turnover:           Maximum one-way turnover per rebalance.  Set to
                                ``None`` for unlimited.
        priority_method:        How to rank trades when budget is binding.
        partial_rebalance_frac: Fraction of the drift to close (1.0 = full
                                rebalance, 0.5 = move halfway to target).
        min_trade_weight:       Dead-band — trades smaller than this are dropped.
    """

    no_trade_band: float = 0.02
    max_turnover: float | None = 0.30
    priority_method: PriorityMethod = PriorityMethod.LARGEST_DRIFT
    partial_rebalance_frac: float = 1.0
    min_trade_weight: float = 0.001


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SymbolDrift:
    """Drift information for a single symbol."""

    symbol: str
    current_weight: float
    target_weight: float
    drift: float  # target - current (signed)
    abs_drift: float
    exceeds_band: bool


@dataclass
class DriftReport:
    """Portfolio-level drift analysis.

    Attributes:
        symbol_drifts:    Per-symbol drift details.
        total_drift:      Sum of absolute drifts (one-way turnover to fully rebalance).
        max_drift:        Largest absolute drift.
        max_drift_symbol: Symbol with largest drift.
        needs_rebalance:  True if any symbol exceeds the no-trade band.
        n_drifted:        Number of symbols exceeding band.
    """

    symbol_drifts: list[SymbolDrift]
    total_drift: float
    max_drift: float
    max_drift_symbol: str
    needs_rebalance: bool
    n_drifted: int


@dataclass(frozen=True, slots=True)
class OptimizedTrade:
    """One trade in the optimized rebalance plan.

    Attributes:
        symbol:           Asset symbol.
        current_weight:   Weight before rebalancing.
        target_weight:    Ideal target weight from portfolio construction.
        optimized_weight: Weight after optimized (possibly partial) rebalance.
        trade_weight:     Weight delta to execute (signed).
        priority_score:   Score used for ranking (higher = more urgent).
        included:         True if this trade was included within turnover budget.
    """

    symbol: str
    current_weight: float
    target_weight: float
    optimized_weight: float
    trade_weight: float
    priority_score: float
    included: bool


@dataclass
class TurnoverOptimizationResult:
    """Complete turnover optimization output.

    Attributes:
        drift_report:          Pre-optimisation drift analysis.
        trades:                All candidate trades with inclusion status.
        optimized_weights:     Final weights after optimized rebalancing.
        total_turnover:        Actual one-way turnover of included trades.
        naive_turnover:        Turnover that a full rebalance would have incurred.
        turnover_saved:        ``naive_turnover - total_turnover``.
        n_trades_executed:     Number of included trades.
        n_trades_skipped:      Trades excluded (band, budget, or dead-band).
        rebalance_efficiency:  Drift reduction per unit turnover.
    """

    drift_report: DriftReport
    trades: list[OptimizedTrade]
    optimized_weights: dict[str, float] = field(repr=False)
    total_turnover: float = 0.0
    naive_turnover: float = 0.0
    turnover_saved: float = 0.0
    n_trades_executed: int = 0
    n_trades_skipped: int = 0
    rebalance_efficiency: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "Turnover-Optimized Rebalance",
            "=" * 60,
            "",
            f"  Drift (pre-rebal)     : {self.drift_report.total_drift:.4f}",
            f"  Max drift             : {self.drift_report.max_drift:.4f}"
            f"  ({self.drift_report.max_drift_symbol})",
            f"  Positions drifted     : {self.drift_report.n_drifted}",
            "",
            f"  Naive turnover        : {self.naive_turnover:.4f}",
            f"  Optimized turnover    : {self.total_turnover:.4f}",
            f"  Turnover saved        : {self.turnover_saved:.4f}",
            f"  Rebalance efficiency  : {self.rebalance_efficiency:.2f}",
            "",
            f"  Trades executed       : {self.n_trades_executed}",
            f"  Trades skipped        : {self.n_trades_skipped}",
        ]

        if self.trades:
            lines.extend(["", "  Trade Schedule:", "  " + "-" * 56])
            lines.append(
                f"  {'Symbol':<8s} {'Curr':>7s} {'Target':>7s} "
                f"{'Optim':>7s} {'Delta':>7s} {'Score':>7s} {'Status':>8s}"
            )
            for t in sorted(self.trades, key=lambda x: -x.priority_score):
                status = "EXEC" if t.included else "SKIP"
                lines.append(
                    f"  {t.symbol:<8s} {t.current_weight:>+7.4f} "
                    f"{t.target_weight:>+7.4f} {t.optimized_weight:>+7.4f} "
                    f"{t.trade_weight:>+7.4f} {t.priority_score:>7.2f} "
                    f"{status:>8s}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class TurnoverOptimizer:
    """Turnover-optimized rebalancing engine.

    Args:
        config: Optimisation parameters.
    """

    def __init__(self, config: TurnoverConfig | None = None) -> None:
        self._config = config or TurnoverConfig()

    @property
    def config(self) -> TurnoverConfig:
        return self._config

    def compute_drift(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> DriftReport:
        """Compute per-symbol drift from target weights.

        Args:
            current_weights: ``{symbol: weight}`` of current portfolio.
            target_weights:  ``{symbol: weight}`` of desired portfolio.

        Returns:
            :class:`DriftReport` with per-symbol and aggregate drift metrics.
        """
        band = self._config.no_trade_band
        all_symbols = sorted(set(current_weights) | set(target_weights))

        drifts: list[SymbolDrift] = []
        for sym in all_symbols:
            curr = current_weights.get(sym, 0.0)
            tgt = target_weights.get(sym, 0.0)
            drift = tgt - curr
            abs_drift = abs(drift)
            drifts.append(
                SymbolDrift(
                    symbol=sym,
                    current_weight=curr,
                    target_weight=tgt,
                    drift=drift,
                    abs_drift=abs_drift,
                    exceeds_band=abs_drift >= band,
                )
            )

        total = sum(d.abs_drift for d in drifts)
        if drifts:
            worst = max(drifts, key=lambda d: d.abs_drift)
            max_drift = worst.abs_drift
            max_sym = worst.symbol
        else:
            max_drift = 0.0
            max_sym = ""

        n_drifted = sum(1 for d in drifts if d.exceeds_band)

        return DriftReport(
            symbol_drifts=drifts,
            total_drift=total,
            max_drift=max_drift,
            max_drift_symbol=max_sym,
            needs_rebalance=n_drifted > 0,
            n_drifted=n_drifted,
        )

    def optimize(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        expected_alpha: dict[str, float] | None = None,
        cost_estimates: dict[str, float] | None = None,
    ) -> TurnoverOptimizationResult:
        """Produce a turnover-optimized rebalance plan.

        Args:
            current_weights: ``{symbol: weight}`` of current portfolio.
            target_weights:  ``{symbol: weight}`` of desired portfolio.
            expected_alpha:  ``{symbol: expected_alpha_bps}`` per position.
                             Used when ``priority_method`` is ``ALPHA_COST_RATIO``.
            cost_estimates:  ``{symbol: estimated_cost_bps}`` per position.
                             Used when ``priority_method`` is ``ALPHA_COST_RATIO``.

        Returns:
            :class:`TurnoverOptimizationResult` with optimized trade schedule.
        """
        cfg = self._config

        # Step 1: Compute drift
        drift_report = self.compute_drift(current_weights, target_weights)
        naive_turnover = drift_report.total_drift

        # Step 2: Build candidate list — only symbols exceeding band
        candidates: list[dict] = []
        for sd in drift_report.symbol_drifts:
            if not sd.exceeds_band:
                continue
            # Apply partial rebalance fraction
            partial_drift = sd.drift * cfg.partial_rebalance_frac
            if abs(partial_drift) < cfg.min_trade_weight:
                continue
            candidates.append({
                "symbol": sd.symbol,
                "current": sd.current_weight,
                "target": sd.target_weight,
                "partial_drift": partial_drift,
                "abs_partial": abs(partial_drift),
                "full_drift": sd.drift,
            })

        # Step 3: Score candidates by priority method
        for c in candidates:
            c["score"] = self._score_candidate(
                c, expected_alpha, cost_estimates, cfg.priority_method,
            )

        # Sort descending by score
        candidates.sort(key=lambda c: -c["score"])

        # Step 4: Greedily include trades within turnover budget
        budget = cfg.max_turnover if cfg.max_turnover is not None else float("inf")
        used_turnover = 0.0
        included_trades: list[OptimizedTrade] = []
        skipped_trades: list[OptimizedTrade] = []

        for c in candidates:
            trade_turnover = c["abs_partial"]
            if used_turnover + trade_turnover <= budget + 1e-10:
                # Include full partial trade
                optimized_w = c["current"] + c["partial_drift"]
                included_trades.append(
                    OptimizedTrade(
                        symbol=c["symbol"],
                        current_weight=c["current"],
                        target_weight=c["target"],
                        optimized_weight=optimized_w,
                        trade_weight=c["partial_drift"],
                        priority_score=c["score"],
                        included=True,
                    )
                )
                used_turnover += trade_turnover
            elif budget - used_turnover > cfg.min_trade_weight:
                # Partial inclusion: use remaining budget
                remaining = budget - used_turnover
                fraction = remaining / trade_turnover
                scaled_drift = c["partial_drift"] * fraction
                optimized_w = c["current"] + scaled_drift
                included_trades.append(
                    OptimizedTrade(
                        symbol=c["symbol"],
                        current_weight=c["current"],
                        target_weight=c["target"],
                        optimized_weight=optimized_w,
                        trade_weight=scaled_drift,
                        priority_score=c["score"],
                        included=True,
                    )
                )
                used_turnover += abs(scaled_drift)
            else:
                # Budget exhausted
                skipped_trades.append(
                    OptimizedTrade(
                        symbol=c["symbol"],
                        current_weight=c["current"],
                        target_weight=c["target"],
                        optimized_weight=c["current"],  # No trade
                        trade_weight=0.0,
                        priority_score=c["score"],
                        included=False,
                    )
                )

        # Also record symbols within band as skipped (no trade needed)
        band_skipped: list[OptimizedTrade] = []
        for sd in drift_report.symbol_drifts:
            is_candidate = any(
                c["symbol"] == sd.symbol for c in candidates
            )
            if not is_candidate and sd.abs_drift > 1e-10:
                band_skipped.append(
                    OptimizedTrade(
                        symbol=sd.symbol,
                        current_weight=sd.current_weight,
                        target_weight=sd.target_weight,
                        optimized_weight=sd.current_weight,
                        trade_weight=0.0,
                        priority_score=0.0,
                        included=False,
                    )
                )

        all_trades = included_trades + skipped_trades + band_skipped

        # Build final weight map
        optimized_weights = dict(current_weights)
        for t in included_trades:
            optimized_weights[t.symbol] = t.optimized_weight
        # Add new positions (in target but not in current)
        for sym in target_weights:
            if sym not in optimized_weights:
                optimized_weights[sym] = 0.0
        # Clean near-zero
        optimized_weights = {
            s: w for s, w in optimized_weights.items() if abs(w) > 1e-10
        }

        # Compute efficiency: drift reduction per unit turnover
        residual_drift = sum(
            abs(target_weights.get(s, 0.0) - optimized_weights.get(s, 0.0))
            for s in set(target_weights) | set(optimized_weights)
        )
        drift_reduced = naive_turnover - residual_drift
        efficiency = drift_reduced / used_turnover if used_turnover > 1e-10 else 0.0

        n_exec = len(included_trades)
        n_skip = len(skipped_trades) + len(band_skipped)

        return TurnoverOptimizationResult(
            drift_report=drift_report,
            trades=all_trades,
            optimized_weights=optimized_weights,
            total_turnover=used_turnover,
            naive_turnover=naive_turnover,
            turnover_saved=naive_turnover - used_turnover,
            n_trades_executed=n_exec,
            n_trades_skipped=n_skip,
            rebalance_efficiency=efficiency,
        )

    @staticmethod
    def _score_candidate(
        candidate: dict,
        expected_alpha: dict[str, float] | None,
        cost_estimates: dict[str, float] | None,
        method: PriorityMethod,
    ) -> float:
        """Compute priority score for a candidate trade."""
        sym = candidate["symbol"]

        if method == PriorityMethod.ALPHA_COST_RATIO:
            alpha = (expected_alpha or {}).get(sym, 0.0)
            cost = (cost_estimates or {}).get(sym, 1.0)
            # Avoid division by zero; scale by drift magnitude as tiebreaker
            cost = max(cost, 0.01)
            return (alpha / cost) * candidate["abs_partial"]

        if method == PriorityMethod.LARGEST_DRIFT:
            return candidate["abs_partial"]

        # EQUAL: all trades get the same score
        return 1.0
