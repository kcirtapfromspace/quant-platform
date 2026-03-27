"""Multi-strategy portfolio backtester.

Simulates the full orchestrator pipeline over historical data:

  N sleeves × (Signals → Alpha → Construction) → Capital-weighted
  combination → Lifecycle evaluation → Reallocation → Costs → P&L

Each sleeve runs independently with its own signal set and portfolio
config, then target weights are combined proportional to sleeve capital
weights.  Lifecycle evaluation runs at every rebalance point and, when
``apply_lifecycle_realloc`` is enabled, adjusts sleeve capital weights
for the next period.

Usage::

    from quant.backtest.multi_strategy import (
        MultiStrategyBacktestEngine,
        MultiStrategyConfig,
        SleeveConfig,
    )
    from quant.signals.factors import MomentumSignal, MeanReversionSignal

    config = MultiStrategyConfig(
        sleeves=[
            SleeveConfig(
                name="momentum",
                signals=[MomentumSignal()],
                capital_weight=0.6,
            ),
            SleeveConfig(
                name="mean_rev",
                signals=[MeanReversionSignal()],
                capital_weight=0.4,
            ),
        ],
        rebalance_frequency=21,
    )
    engine = MultiStrategyBacktestEngine()
    report = engine.run(daily_returns_df, config)
    print(report.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger

from quant.backtest import metrics as m
from quant.execution.cost_model import TransactionCostModel
from quant.portfolio.alpha import AlphaCombiner, CombinationMethod
from quant.portfolio.engine import PortfolioConfig, PortfolioEngine
from quant.portfolio.lifecycle import (
    LifecycleConfig,
    LifecycleManager,
    LifecycleReport,
    StrategySnapshot,
)
from quant.portfolio.position_scaler import PositionScaler, ScalingConfig
from quant.portfolio.pre_trade import PreTradeConfig, PreTradePipeline
from quant.signals.adaptive_combiner import AdaptiveCombinerConfig, AdaptiveSignalCombiner
from quant.signals.base import BaseSignal, SignalOutput

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SleeveConfig:
    """Configuration for one strategy sleeve within the multi-strategy backtest.

    Attributes:
        name:               Unique sleeve identifier.
        signals:            Signal instances for this sleeve.
        capital_weight:     Fraction of total capital allocated (0–1).
        portfolio_config:   Portfolio construction settings for this sleeve.
        combination_method: How to combine this sleeve's signal outputs.
        signal_weights:     Weights for STATIC_WEIGHT combination.
        adaptive_combiner_config: IC-adaptive signal combination config.
        scaling_config:     Position scaling config (conviction / vol / Kelly).
    """

    name: str
    signals: list[BaseSignal] = field(default_factory=list)
    capital_weight: float = 1.0
    portfolio_config: PortfolioConfig = field(default_factory=PortfolioConfig)
    combination_method: CombinationMethod = CombinationMethod.EQUAL_WEIGHT
    signal_weights: dict[str, float] | None = None
    adaptive_combiner_config: AdaptiveCombinerConfig | None = None
    scaling_config: ScalingConfig | None = None


@dataclass
class MultiStrategyConfig:
    """Configuration for the multi-strategy backtester.

    Attributes:
        sleeves:                Strategy sleeve configs.
        rebalance_frequency:    Trading days between rebalances.
        commission_bps:         Flat transaction cost in basis points.
        initial_capital:        Starting portfolio value.
        cost_model:             Full cost model (overrides commission_bps).
        pre_trade_config:       Pre-trade pipeline config.
        sector_map:             {symbol: sector} for constraint checks.
        lifecycle_config:       Lifecycle evaluation config.
        apply_lifecycle_realloc: Apply lifecycle recommendations to next period.
        min_history:            Min trading days before first trade.
        name:                   Label for the backtest.
    """

    sleeves: list[SleeveConfig] = field(default_factory=list)
    rebalance_frequency: int = 21
    commission_bps: float = 10.0
    initial_capital: float = 1_000_000.0
    cost_model: TransactionCostModel | None = None
    pre_trade_config: PreTradeConfig | None = None
    sector_map: dict[str, str] = field(default_factory=dict)
    lifecycle_config: LifecycleConfig | None = None
    apply_lifecycle_realloc: bool = False
    min_history: int = 60
    name: str = "multi_strategy_backtest"


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SleeveSnapshot:
    """Per-sleeve state at a rebalance point."""

    name: str
    capital_weight: float
    target_weights: dict[str, float]
    n_assets: int


@dataclass(frozen=True, slots=True)
class MultiRebalanceSnapshot:
    """Record of one multi-strategy rebalance event."""

    date: date
    combined_weights: dict[str, float]
    sleeve_snapshots: list[SleeveSnapshot]
    portfolio_value: float
    turnover: float
    transaction_costs: float
    n_assets: int
    lifecycle_report: LifecycleReport | None = None


@dataclass
class MultiStrategyBacktestReport:
    """Full results from a multi-strategy backtest."""

    name: str
    start_date: date
    end_date: date
    initial_capital: float
    final_value: float

    # Return metrics
    total_return: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float

    # Execution metrics
    avg_turnover: float
    total_costs: float
    n_rebalances: int
    n_trading_days: int
    n_sleeves: int

    # Time series
    equity_curve: pd.Series = field(repr=False)
    returns_series: pd.Series = field(repr=False)
    weights_history: pd.DataFrame = field(repr=False)
    rebalances: list[MultiRebalanceSnapshot] = field(repr=False)

    # Per-sleeve capital weight history
    capital_weight_history: pd.DataFrame = field(repr=False)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Multi-Strategy Backtest : {self.name}",
            f"Period                  : {self.start_date} -> {self.end_date}",
            f"Trading days            : {self.n_trading_days}",
            f"Sleeves                 : {self.n_sleeves}",
            "-" * 50,
            f"Initial capital         : ${self.initial_capital:,.0f}",
            f"Final value             : ${self.final_value:,.0f}",
            f"Total return            : {self.total_return:+.2%}",
            f"CAGR                    : {self.cagr:+.2%}",
            f"Sharpe ratio            : {self.sharpe_ratio:.2f}",
            f"Volatility (ann.)       : {self.volatility:.2%}",
            f"Max drawdown            : {self.max_drawdown:.2%}",
            "-" * 50,
            f"Rebalances              : {self.n_rebalances}",
            f"Avg turnover            : {self.avg_turnover:.2%}",
            f"Total costs             : ${self.total_costs:,.0f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MultiStrategyBacktestEngine:
    """Multi-strategy backtesting engine.

    Runs N strategy sleeves in parallel at each rebalance point, combines
    their target weights proportional to capital allocation, and optionally
    applies lifecycle-based capital reallocation between periods.
    """

    def run(
        self,
        returns: pd.DataFrame,
        config: MultiStrategyConfig,
    ) -> MultiStrategyBacktestReport:
        """Run a multi-strategy backtest.

        Args:
            returns: DataFrame of daily returns (DatetimeIndex × symbols).
            config:  Multi-strategy configuration.

        Returns:
            :class:`MultiStrategyBacktestReport` with full results.
        """
        returns = self._validate(returns)
        symbols = list(returns.columns)
        dates = returns.index
        n_days = len(dates)

        if n_days < config.min_history + 1:
            raise ValueError(
                f"Insufficient data: {n_days} days, need {config.min_history + 1}"
            )

        self._validate_sleeve_weights(config.sleeves)

        # NaN-safe returns
        filled = returns.fillna(0.0)
        commission_rate = config.commission_bps / 10_000

        # Per-sleeve pipeline components
        sleeve_combiners: dict[str, AlphaCombiner] = {}
        sleeve_adaptive: dict[str, AdaptiveSignalCombiner] = {}
        sleeve_engines: dict[str, PortfolioEngine] = {}

        for sc in config.sleeves:
            sleeve_combiners[sc.name] = AlphaCombiner(
                method=sc.combination_method,
                weights=sc.signal_weights,
            )
            if sc.adaptive_combiner_config is not None:
                sleeve_adaptive[sc.name] = AdaptiveSignalCombiner(
                    sc.adaptive_combiner_config
                )
            sleeve_engines[sc.name] = PortfolioEngine(sc.portfolio_config)

        # Lifecycle manager
        lifecycle_mgr: LifecycleManager | None = None
        if config.lifecycle_config is not None:
            lifecycle_mgr = LifecycleManager(config.lifecycle_config)

        # State
        portfolio_value = config.initial_capital
        weights: dict[str, float] = {}
        capital_weights = {sc.name: sc.capital_weight for sc in config.sleeves}
        total_costs = 0.0
        days_since_rebalance = config.rebalance_frequency

        # Collectors
        equity_values: list[float] = []
        daily_rets: list[float] = []
        weights_records: list[dict[str, float]] = []
        cap_weight_records: list[dict[str, float]] = []
        rebalances: list[MultiRebalanceSnapshot] = []

        for i in range(n_days):
            dt = dates[i]
            day_return = 0.0

            # ── 1. Earn today's return on existing weights ──────────────
            if weights and i > 0:
                day_return = sum(
                    weights.get(sym, 0.0) * float(filled.iat[i, j])
                    for j, sym in enumerate(symbols)
                )
                portfolio_value *= 1 + day_return

                # Drift weights
                if abs(1 + day_return) > 1e-12:
                    drifted: dict[str, float] = {}
                    for sym, w in weights.items():
                        j = symbols.index(sym) if sym in symbols else -1
                        if j < 0:
                            continue
                        r_sym = float(filled.iat[i, j])
                        new_w = w * (1 + r_sym) / (1 + day_return)
                        if abs(new_w) > 1e-12:
                            drifted[sym] = new_w
                    weights = drifted

            # ── 2. Rebalance ───────────────────────────────────────────
            days_since_rebalance += 1

            if (
                days_since_rebalance >= config.rebalance_frequency
                and i >= config.min_history
            ):
                visible = returns.iloc[: i + 1]
                ts = pd.Timestamp(dt).to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                # ── 2a. Run each sleeve ────────────────────────────────
                sleeve_snapshots: list[SleeveSnapshot] = []
                combined: dict[str, float] = {}

                for sc in config.sleeves:
                    sleeve_weights = self._run_sleeve(
                        sc=sc,
                        visible=visible,
                        symbols=symbols,
                        timestamp=ts,
                        combiner=sleeve_combiners[sc.name],
                        adaptive=sleeve_adaptive.get(sc.name),
                        engine=sleeve_engines[sc.name],
                        current_weights=weights,
                        portfolio_value=portfolio_value,
                        min_history=config.min_history,
                    )

                    if sleeve_weights is not None:
                        cap_w = capital_weights[sc.name]
                        scaled = {s: w * cap_w for s, w in sleeve_weights.items()}
                        for s, w in scaled.items():
                            combined[s] = combined.get(s, 0.0) + w

                        sleeve_snapshots.append(
                            SleeveSnapshot(
                                name=sc.name,
                                capital_weight=cap_w,
                                target_weights=scaled,
                                n_assets=sum(1 for v in scaled.values() if abs(v) > 1e-9),
                            )
                        )

                if combined:
                    # ── 2b. Pre-trade pipeline ─────────────────────────
                    if config.pre_trade_config is not None:
                        pipeline = PreTradePipeline(config.pre_trade_config)
                        pt_result = pipeline.process(
                            target_weights=combined,
                            current_weights=weights,
                            portfolio_value=portfolio_value,
                            sector_map=config.sector_map or None,
                        )
                        combined = pt_result.adjusted_weights

                    # ── 2c. Costs ──────────────────────────────────────
                    all_syms = set(combined) | set(weights)
                    turnover = sum(
                        abs(combined.get(s, 0.0) - weights.get(s, 0.0))
                        for s in all_syms
                    )
                    if config.cost_model is not None:
                        costs = self._estimate_rebalance_cost(
                            config.cost_model, combined, weights, portfolio_value
                        )
                    else:
                        costs = turnover * portfolio_value * commission_rate
                    portfolio_value -= costs
                    total_costs += costs

                    # ── 2d. Lifecycle evaluation ───────────────────────
                    lifecycle_report: LifecycleReport | None = None
                    if lifecycle_mgr is not None:
                        for snap in sleeve_snapshots:
                            strategy_returns = self._compute_sleeve_returns(
                                snap.target_weights, visible
                            )
                            lifecycle_mgr.update(
                                StrategySnapshot(
                                    name=snap.name,
                                    returns_series=strategy_returns,
                                    current_weight=snap.capital_weight,
                                )
                            )
                        lifecycle_report = lifecycle_mgr.evaluate()

                        # Apply reallocation for next period
                        if (
                            config.apply_lifecycle_realloc
                            and lifecycle_report.recommendations
                        ):
                            for rec in lifecycle_report.recommendations:
                                if rec.strategy in capital_weights:
                                    capital_weights[rec.strategy] = rec.recommended_weight

                    dt_date = dt.date() if hasattr(dt, "date") else dt
                    rebalances.append(
                        MultiRebalanceSnapshot(
                            date=dt_date,
                            combined_weights=dict(combined),
                            sleeve_snapshots=sleeve_snapshots,
                            portfolio_value=portfolio_value,
                            turnover=turnover,
                            transaction_costs=costs,
                            n_assets=sum(1 for v in combined.values() if abs(v) > 1e-9),
                            lifecycle_report=lifecycle_report,
                        )
                    )

                    weights = combined
                    days_since_rebalance = 0

            equity_values.append(portfolio_value)
            daily_rets.append(day_return)
            weights_records.append(dict(weights))
            cap_weight_records.append(dict(capital_weights))

        # ── Build report ──────────────────────────────────────────────
        equity_curve = pd.Series(equity_values, index=dates, name="portfolio_value")
        returns_series = pd.Series(daily_rets, index=dates, name="returns")
        weights_history = pd.DataFrame(weights_records, index=dates).fillna(0.0)
        cap_weight_history = pd.DataFrame(cap_weight_records, index=dates).fillna(0.0)

        eval_start = config.min_history
        eval_returns = returns_series.iloc[eval_start:]
        eval_equity = equity_curve.iloc[eval_start:]
        eval_rebased = (
            eval_equity / eval_equity.iloc[0]
            if not eval_equity.empty and eval_equity.iloc[0] > 0
            else eval_equity
        )

        sharpe = m.sharpe_ratio(eval_returns)
        mdd = m.max_drawdown(eval_rebased)
        total_ret = (portfolio_value / config.initial_capital) - 1
        n_years = n_days / 252
        cagr_val = (
            (portfolio_value / config.initial_capital) ** (1 / n_years) - 1
            if n_years > 0 and portfolio_value > 0
            else 0.0
        )
        vol = (
            float(eval_returns.std() * math.sqrt(252))
            if len(eval_returns) > 1
            else 0.0
        )
        avg_turnover = (
            float(np.mean([r.turnover for r in rebalances]))
            if rebalances
            else 0.0
        )

        start_date = dates[0].date() if hasattr(dates[0], "date") else dates[0]
        end_date = dates[-1].date() if hasattr(dates[-1], "date") else dates[-1]

        report = MultiStrategyBacktestReport(
            name=config.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=config.initial_capital,
            final_value=portfolio_value,
            total_return=total_ret,
            cagr=cagr_val,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            volatility=vol,
            avg_turnover=avg_turnover,
            total_costs=total_costs,
            n_rebalances=len(rebalances),
            n_trading_days=n_days,
            n_sleeves=len(config.sleeves),
            equity_curve=equity_curve,
            returns_series=returns_series,
            weights_history=weights_history,
            rebalances=rebalances,
            capital_weight_history=cap_weight_history,
        )

        logger.info(
            "Multi-strategy backtest: {} | {} sleeves | Sharpe={:.2f} | "
            "MaxDD={:.1%} | CAGR={:.1%} | {} rebalances",
            config.name,
            len(config.sleeves),
            sharpe,
            mdd,
            cagr_val,
            len(rebalances),
        )

        return report

    # ── Sleeve execution ──────────────────────────────────────────────

    @staticmethod
    def _run_sleeve(
        sc: SleeveConfig,
        visible: pd.DataFrame,
        symbols: list[str],
        timestamp: datetime,
        combiner: AlphaCombiner,
        adaptive: AdaptiveSignalCombiner | None,
        engine: PortfolioEngine,
        current_weights: dict[str, float],
        portfolio_value: float,
        min_history: int,
    ) -> dict[str, float] | None:
        """Run one sleeve's signal→alpha→construction pipeline.

        Returns target weights (unscaled by capital weight), or None on failure.
        """
        # Compute signals
        universe_signals: dict[str, list[SignalOutput]] = {}
        for sym in symbols:
            sym_returns = visible[sym].dropna()
            if len(sym_returns) < min_history:
                continue

            sym_signals: list[SignalOutput] = []
            for signal in sc.signals:
                try:
                    output = signal.compute(
                        sym, {"returns": sym_returns}, timestamp
                    )
                    tagged = SignalOutput(
                        symbol=output.symbol,
                        timestamp=output.timestamp,
                        score=output.score,
                        confidence=output.confidence,
                        target_position=output.target_position,
                        metadata={**output.metadata, "signal_name": signal.name},
                    )
                    sym_signals.append(tagged)
                except Exception:
                    pass
            if sym_signals:
                universe_signals[sym] = sym_signals

        if not universe_signals:
            return None

        # Combine into alpha
        if adaptive is not None:
            last_returns = visible.iloc[-1]
            signal_scores: dict[str, pd.Series] = {}
            for sym, sig_list in universe_signals.items():
                for sig in sig_list:
                    sig_name = sig.metadata.get("signal_name", "unknown")
                    if sig_name not in signal_scores:
                        signal_scores[sig_name] = pd.Series(dtype=float)
                    signal_scores[sig_name] = pd.concat([
                        signal_scores[sig_name],
                        pd.Series([sig.score], index=[sym]),
                    ])
            adaptive.update(signal_scores, last_returns, timestamp)
            alpha_scores = adaptive.combine_universe(timestamp, universe_signals)
        else:
            alpha_scores = combiner.combine_universe(timestamp, universe_signals)

        # Position scaling
        if sc.scaling_config is not None:
            scaler = PositionScaler(sc.scaling_config)
            alpha_scores = scaler.scale_to_alpha_dict(alpha_scores, visible)

        # Portfolio construction
        try:
            construction = engine.construct(
                alpha_scores=alpha_scores,
                returns_history=visible,
                current_weights=current_weights,
                portfolio_value=portfolio_value,
            )
        except Exception:
            return None

        if not construction.rebalance_triggered:
            return None

        return construction.optimization.weights

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_sleeve_returns(
        target_weights: dict[str, float], returns_df: pd.DataFrame
    ) -> pd.Series:
        """Approximate strategy returns from asset returns × weights."""
        if returns_df.empty or not target_weights:
            return pd.Series(dtype=float)
        common = [s for s in target_weights if s in returns_df.columns]
        if not common:
            return pd.Series(dtype=float)
        w = pd.Series({s: target_weights[s] for s in common})
        return returns_df[common].mul(w).sum(axis=1)

    @staticmethod
    def _estimate_rebalance_cost(
        cost_model: TransactionCostModel,
        new_weights: dict[str, float],
        old_weights: dict[str, float],
        portfolio_value: float,
    ) -> float:
        """Estimate total dollar cost of a rebalance."""
        total_cost = 0.0
        for sym in set(new_weights) | set(old_weights):
            dw = abs(new_weights.get(sym, 0.0) - old_weights.get(sym, 0.0))
            if dw < 1e-10:
                continue
            notional = dw * portfolio_value
            est = cost_model.estimate_order_cost(symbol=sym, notional=notional)
            total_cost += est.total_dollars
        return total_cost

    @staticmethod
    def _validate(returns: pd.DataFrame) -> pd.DataFrame:
        if returns.empty:
            raise ValueError("returns DataFrame is empty")
        df = returns.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    @staticmethod
    def _validate_sleeve_weights(sleeves: list[SleeveConfig]) -> None:
        total = sum(s.capital_weight for s in sleeves)
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"Sleeve capital weights sum to {total:.4f}, must be <= 1.0"
            )
