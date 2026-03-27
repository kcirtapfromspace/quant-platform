"""Pre-trade validation and adjustment pipeline.

Sits between portfolio construction (target weights) and order execution,
applying a composable chain of checks and adjustments:

  1. **Risk limit enforcement** — clamp weights that breach position, sector,
     leverage, or concentration limits.
  2. **Cost-aware filtering** — drop trades whose estimated cost exceeds the
     expected alpha (break-even test).
  3. **Minimum trade filter** — skip trades below a dollar or weight threshold
     to avoid microrebalancing noise.

The pipeline returns adjusted weights and a detailed audit of what changed.

Usage::

    from quant.portfolio.pre_trade import PreTradePipeline, PreTradeConfig

    pipeline = PreTradePipeline(PreTradeConfig(
        limit_checker=my_limit_checker,
        cost_model=my_cost_model,
        min_trade_weight=0.005,
    ))

    result = pipeline.process(
        target_weights=target,
        current_weights=current,
        portfolio_value=1_000_000,
        sector_map=sectors,
    )
    final_weights = result.adjusted_weights
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from quant.execution.cost_model import TransactionCostModel
from quant.risk.limit_checker import LimitCheckReport, RiskLimitChecker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PreTradeConfig:
    """Configuration for the pre-trade pipeline.

    Attributes:
        limit_checker:      Risk limit checker (None to skip).
        cost_model:         Transaction cost model (None to skip cost filter).
        min_trade_weight:   Minimum absolute weight change to execute.
                            Trades below this are dropped.
        min_trade_dollars:  Minimum dollar notional per trade.
        cost_alpha_ratio:   Maximum cost-to-alpha ratio.  Trades where
                            estimated round-trip cost exceeds this fraction
                            of expected alpha are dropped.  Set to None
                            to disable.
        enforce_limits:     If True, clamp weights on limit breach.
                            If False, only report breaches.
    """

    limit_checker: RiskLimitChecker | None = None
    cost_model: TransactionCostModel | None = None
    min_trade_weight: float = 0.005
    min_trade_dollars: float = 500.0
    cost_alpha_ratio: float | None = None
    enforce_limits: bool = True


# ---------------------------------------------------------------------------
# Adjustment record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeAdjustment:
    """Record of a single weight adjustment made by the pipeline.

    Attributes:
        symbol:         Affected symbol.
        stage:          Pipeline stage that made the adjustment.
        original_weight: Weight before adjustment.
        adjusted_weight: Weight after adjustment.
        reason:         Why the adjustment was made.
    """

    symbol: str
    stage: str
    original_weight: float
    adjusted_weight: float
    reason: str


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PreTradeResult:
    """Output of the pre-trade pipeline.

    Attributes:
        adjusted_weights:   Final weights after all pipeline stages.
        original_weights:   Input target weights (unchanged).
        adjustments:        List of individual adjustments.
        limit_report:       Risk limit check report (None if skipped).
        trades_filtered:    Number of trades dropped by cost/min filters.
        trades_remaining:   Number of trades that passed all filters.
        timestamp:          When the pipeline was run.
    """

    adjusted_weights: dict[str, float] = field(default_factory=dict)
    original_weights: dict[str, float] = field(default_factory=dict)
    adjustments: list[TradeAdjustment] = field(default_factory=list)
    limit_report: LimitCheckReport | None = None
    trades_filtered: int = 0
    trades_remaining: int = 0
    timestamp: datetime | None = None

    @property
    def n_adjustments(self) -> int:
        return len(self.adjustments)

    @property
    def was_modified(self) -> bool:
        return self.n_adjustments > 0

    def summary(self) -> str:
        """Human-readable pipeline summary."""
        lines = [
            "Pre-Trade Pipeline",
            "=" * 60,
            f"  Trades remaining: {self.trades_remaining}",
            f"  Trades filtered:  {self.trades_filtered}",
            f"  Adjustments:      {self.n_adjustments}",
        ]

        if self.limit_report and self.limit_report.has_any_breach:
            lines.append(f"  Limit breaches:   {len(self.limit_report.breaches)}")

        if self.adjustments:
            lines.append("")
            lines.append(f"  {'Symbol':<10}{'Stage':<20}{'From':>10}{'To':>10}{'Reason'}")
            lines.append("-" * 60)
            for adj in self.adjustments:
                lines.append(
                    f"  {adj.symbol:<10}{adj.stage:<20}"
                    f"{adj.original_weight:>+10.4f}{adj.adjusted_weight:>+10.4f}"
                    f"  {adj.reason}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PreTradePipeline:
    """Pre-trade validation and adjustment pipeline.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: PreTradeConfig | None = None) -> None:
        self._config = config or PreTradeConfig()

    @property
    def config(self) -> PreTradeConfig:
        return self._config

    def process(
        self,
        target_weights: dict[str, float],
        current_weights: dict[str, float] | None = None,
        portfolio_value: float = 1_000_000,
        sector_map: dict[str, str] | None = None,
        expected_alpha_bps: dict[str, float] | None = None,
        adv: dict[str, float] | None = None,
        volatility: dict[str, float] | None = None,
    ) -> PreTradeResult:
        """Run the full pre-trade pipeline.

        Args:
            target_weights:     ``{symbol: target_weight}`` from portfolio
                                construction.
            current_weights:    ``{symbol: current_weight}`` (defaults to
                                all zeros).
            portfolio_value:    Current portfolio value in dollars.
            sector_map:         ``{symbol: sector}`` for limit checks.
            expected_alpha_bps: ``{symbol: expected_alpha}`` in bps for
                                cost-alpha filtering.
            adv:                ``{symbol: average_daily_volume}`` for
                                cost estimation.
            volatility:         ``{symbol: annualised_vol}`` for cost
                                estimation.

        Returns:
            :class:`PreTradeResult` with adjusted weights and audit trail.
        """
        now = datetime.now(timezone.utc)
        current_weights = current_weights or {}
        adjustments: list[TradeAdjustment] = []
        limit_report: LimitCheckReport | None = None

        weights = dict(target_weights)

        # ── Stage 1: Risk limit enforcement ───────────────────────
        checker = self._config.limit_checker
        if checker is not None:
            limit_report = checker.check(weights, sector_map, timestamp=now)

            if limit_report.has_any_breach and self._config.enforce_limits:
                enforced = checker.enforce(weights, sector_map)
                for sym in weights:
                    if abs(enforced.get(sym, 0) - weights[sym]) > 1e-10:
                        adjustments.append(
                            TradeAdjustment(
                                symbol=sym,
                                stage="limit_enforce",
                                original_weight=weights[sym],
                                adjusted_weight=enforced.get(sym, 0),
                                reason="limit breach",
                            )
                        )
                weights = enforced

        # ── Stage 2: Minimum trade filter ─────────────────────────
        trades_filtered = 0
        for sym in list(weights.keys()):
            dw = abs(weights[sym] - current_weights.get(sym, 0.0))
            if dw < 1e-10:
                continue  # no trade needed

            # Skip if weight change too small
            if dw < self._config.min_trade_weight:
                adjustments.append(
                    TradeAdjustment(
                        symbol=sym,
                        stage="min_weight",
                        original_weight=weights[sym],
                        adjusted_weight=current_weights.get(sym, 0.0),
                        reason=f"dw={dw:.4f} < {self._config.min_trade_weight}",
                    )
                )
                weights[sym] = current_weights.get(sym, 0.0)
                trades_filtered += 1
                continue

            # Skip if dollar value too small
            notional = dw * portfolio_value
            if notional < self._config.min_trade_dollars:
                adjustments.append(
                    TradeAdjustment(
                        symbol=sym,
                        stage="min_dollars",
                        original_weight=weights[sym],
                        adjusted_weight=current_weights.get(sym, 0.0),
                        reason=f"${notional:.0f} < ${self._config.min_trade_dollars:.0f}",
                    )
                )
                weights[sym] = current_weights.get(sym, 0.0)
                trades_filtered += 1

        # ── Stage 3: Cost-aware filtering ─────────────────────────
        cost_model = self._config.cost_model
        if (
            cost_model is not None
            and expected_alpha_bps is not None
            and self._config.cost_alpha_ratio is not None
        ):
            for sym in list(weights.keys()):
                dw = abs(weights[sym] - current_weights.get(sym, 0.0))
                if dw < 1e-10:
                    continue

                notional = dw * portfolio_value
                est = cost_model.estimate_order_cost(
                    symbol=sym,
                    notional=notional,
                    adv=adv.get(sym) if adv else None,
                    volatility=volatility.get(sym) if volatility else None,
                )

                alpha_bps = expected_alpha_bps.get(sym, 0.0)
                if alpha_bps > 0:
                    # Round-trip cost vs expected alpha
                    round_trip = est.total_bps * 2
                    ratio = round_trip / alpha_bps
                    if ratio > self._config.cost_alpha_ratio:
                        adjustments.append(
                            TradeAdjustment(
                                symbol=sym,
                                stage="cost_filter",
                                original_weight=weights[sym],
                                adjusted_weight=current_weights.get(sym, 0.0),
                                reason=f"cost/alpha={ratio:.2f} > {self._config.cost_alpha_ratio}",
                            )
                        )
                        weights[sym] = current_weights.get(sym, 0.0)
                        trades_filtered += 1

        # Count remaining trades
        trades_remaining = 0
        for sym in weights:
            dw = abs(weights[sym] - current_weights.get(sym, 0.0))
            if dw > 1e-10:
                trades_remaining += 1

        return PreTradeResult(
            adjusted_weights=weights,
            original_weights=dict(target_weights),
            adjustments=adjustments,
            limit_report=limit_report,
            trades_filtered=trades_filtered,
            trades_remaining=trades_remaining,
            timestamp=now,
        )
