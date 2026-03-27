"""Multi-asset portfolio backtesting engine.

Simulates the full investment pipeline over historical data:

  Signals → Alpha Combination → Portfolio Optimisation → Rebalancing →
  Transaction Costs → Performance Measurement

Key properties:

  * **No lookahead bias**: at each rebalance date the engine sees only data
    up to and including that date.  New weights take effect on the *next*
    bar so signal computation cannot peek at the return it will earn.
  * **Realistic costs**: proportional transaction costs applied on turnover
    at every rebalance.
  * **Weight drift**: between rebalances portfolio weights drift with
    realised returns — no implicit daily rebalancing.
  * **Configurable frequency**: rebalance every N trading days.
  * **Benchmark comparison**: optional benchmark column for tracking error,
    information ratio, and active return attribution.

Usage::

    from quant.backtest.portfolio_backtest import (
        PortfolioBacktestConfig,
        PortfolioBacktestEngine,
    )
    from quant.signals.factors import VolatilitySignal, ReturnQualitySignal

    engine = PortfolioBacktestEngine()
    report = engine.run(
        returns=daily_returns_df,
        signals=[VolatilitySignal(), ReturnQualitySignal()],
        config=PortfolioBacktestConfig(
            rebalance_frequency=21,
            commission_bps=10.0,
        ),
    )
    print(report.summary())
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger

from quant.backtest import metrics as m
from quant.portfolio.alpha import AlphaCombiner, CombinationMethod
from quant.portfolio.attribution import AttributionReport, PerformanceAttributor
from quant.portfolio.engine import PortfolioConfig, PortfolioEngine
from quant.portfolio.factor_attribution import FactorAttributionReport, FactorAttributor
from quant.signals.base import BaseSignal, SignalOutput

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PortfolioBacktestConfig:
    """Configuration for a multi-asset portfolio backtest.

    Attributes:
        rebalance_frequency: Number of trading days between rebalances.
        commission_bps: One-way transaction cost in basis points
            (e.g. 10.0 = 10 bps = 0.10%).  Applied to the absolute
            weight change at each rebalance.
        initial_capital: Starting portfolio value in dollars.
        portfolio_config: Portfolio construction / optimisation settings.
        combination_method: How to combine signal outputs into alpha.
        signal_weights: Static weights for STATIC_WEIGHT combination.
        sector_map: ``{symbol: sector}`` for constraint checks.
        min_history: Minimum trading days of data before the first trade.
        benchmark: Column name in the returns DataFrame to use as benchmark.
            If set, the report includes active return and tracking error.
        name: Label for the backtest (appears in reports).
    """

    rebalance_frequency: int = 21
    commission_bps: float = 10.0
    initial_capital: float = 1_000_000.0
    portfolio_config: PortfolioConfig = field(default_factory=PortfolioConfig)
    combination_method: CombinationMethod = CombinationMethod.EQUAL_WEIGHT
    signal_weights: dict[str, float] | None = None
    sector_map: dict[str, str] = field(default_factory=dict)
    min_history: int = 60
    benchmark: str | None = None
    name: str = "portfolio_backtest"


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RebalanceSnapshot:
    """Record of a single rebalance event.

    Attributes:
        date:              Calendar date of the rebalance.
        weights:           Target weights set at this rebalance.
        portfolio_value:   Portfolio value *after* costs are deducted.
        turnover:          Sum of absolute weight changes.
        transaction_costs: Dollar costs incurred at this rebalance.
        n_assets:          Number of non-zero positions after rebalance.
    """

    date: date
    weights: dict[str, float]
    portfolio_value: float
    turnover: float
    transaction_costs: float
    n_assets: int


@dataclass
class PortfolioBacktestReport:
    """Full results from a portfolio backtest.

    Contains scalar metrics, time-series outputs, and per-rebalance
    snapshots sufficient for detailed performance analysis.
    """

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

    # Benchmark comparison
    benchmark_return: float
    benchmark_sharpe: float
    active_return: float
    information_ratio: float
    tracking_error: float

    # Time series
    equity_curve: pd.Series = field(repr=False)
    returns_series: pd.Series = field(repr=False)
    weights_history: pd.DataFrame = field(repr=False)
    rebalances: list[RebalanceSnapshot] = field(repr=False)

    # Attribution (populated when asset returns are available)
    attribution: AttributionReport | None = field(default=None, repr=False)
    factor_attribution: FactorAttributionReport | None = field(default=None, repr=False)

    def summary(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = [
            f"Portfolio Backtest : {self.name}",
            f"Period             : {self.start_date} -> {self.end_date}",
            f"Trading days       : {self.n_trading_days}",
            "-" * 45,
            f"Initial capital    : ${self.initial_capital:,.0f}",
            f"Final value        : ${self.final_value:,.0f}",
            f"Total return       : {self.total_return:+.2%}",
            f"CAGR               : {self.cagr:+.2%}",
            f"Sharpe ratio       : {self.sharpe_ratio:.2f}",
            f"Volatility (ann.)  : {self.volatility:.2%}",
            f"Max drawdown       : {self.max_drawdown:.2%}",
            "-" * 45,
            f"Rebalances         : {self.n_rebalances}",
            f"Avg turnover       : {self.avg_turnover:.2%}",
            f"Total costs        : ${self.total_costs:,.0f}",
        ]
        if self.benchmark_return != 0.0 or self.tracking_error > 0:
            lines += [
                "-" * 45,
                f"Benchmark return   : {self.benchmark_return:+.2%}",
                f"Active return      : {self.active_return:+.2%}",
                f"Tracking error     : {self.tracking_error:.2%}",
                f"Information ratio  : {self.information_ratio:.2f}",
            ]
        if self.factor_attribution is not None:
            fa = self.factor_attribution
            lines += [
                "-" * 45,
                f"Factor R²          : {fa.r_squared:.2%}",
                f"Alpha (ann.)       : {fa.alpha:+.2%}",
                f"Residual vol       : {fa.residual_vol:.2%}",
            ]
            for fc in fa.factor_contributions:
                lines.append(
                    f"  {fc.factor_name:18s}: β={fc.beta:+.3f}  "
                    f"contrib={fc.contribution:+.4f}  "
                    f"risk={fc.risk_contribution:.1%}"
                )
        return "\n".join(lines)


# Type alias for custom feature providers
FeatureProviderFn = Callable[[str, pd.Series], dict[str, pd.Series]]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PortfolioBacktestEngine:
    """Multi-asset portfolio backtesting engine.

    Orchestrates the full pipeline at each rebalance point and tracks
    portfolio value, weights, and costs between rebalances.

    The engine does not require OHLCV data — it operates on a DataFrame of
    daily returns.  Signals are evaluated via their standard ``compute()``
    interface; by default the engine supplies ``{"returns": ...}`` as
    features.  For signals requiring richer inputs, pass a custom
    ``feature_provider``.
    """

    def run(
        self,
        returns: pd.DataFrame,
        signals: list[BaseSignal],
        config: PortfolioBacktestConfig | None = None,
        feature_provider: FeatureProviderFn | None = None,
    ) -> PortfolioBacktestReport:
        """Run a multi-asset portfolio backtest.

        Args:
            returns: DataFrame of daily returns (DatetimeIndex x symbols).
                Each cell = ``(close_t / close_{t-1}) - 1``.
            signals: Signal instances to evaluate at each rebalance.
            config:  Backtest configuration.
            feature_provider: Optional ``(symbol, returns_series) -> features``
                callback.  If *None*, signals receive ``{"returns": series}``.

        Returns:
            :class:`PortfolioBacktestReport` with full results.

        Raises:
            ValueError: If *returns* is empty or has insufficient history.
        """
        if config is None:
            config = PortfolioBacktestConfig()

        returns = self._validate(returns)

        # Separate benchmark column
        benchmark_returns: pd.Series | None = None
        asset_returns = returns
        if config.benchmark and config.benchmark in returns.columns:
            benchmark_returns = returns[config.benchmark].copy()
            asset_returns = returns.drop(columns=[config.benchmark])

        symbols = list(asset_returns.columns)
        dates = asset_returns.index
        n_days = len(dates)

        if n_days < config.min_history + 1:
            raise ValueError(
                f"Insufficient data: {n_days} days, need at least "
                f"{config.min_history + 1}"
            )

        # NaN-safe returns for P&L (treat missing as flat)
        filled_returns = asset_returns.fillna(0.0)

        # Pipeline components
        combiner = AlphaCombiner(
            method=config.combination_method,
            weights=config.signal_weights,
        )
        portfolio_engine = PortfolioEngine(config.portfolio_config)
        commission_rate = config.commission_bps / 10_000

        # State
        portfolio_value = config.initial_capital
        weights: dict[str, float] = {}
        total_costs = 0.0
        days_since_rebalance = config.rebalance_frequency  # trigger first eligible day

        # Collectors
        equity_values: list[float] = []
        daily_rets: list[float] = []
        weights_records: list[dict[str, float]] = []
        rebalances: list[RebalanceSnapshot] = []

        for i in range(n_days):
            dt = dates[i]
            day_return = 0.0

            # ── 1. Earn today's return on existing weights ──────────────
            if weights and i > 0:
                day_return = sum(
                    weights.get(sym, 0.0) * float(filled_returns.iat[i, j])
                    for j, sym in enumerate(symbols)
                )
                portfolio_value *= 1 + day_return

                # Drift weights proportionally to per-asset returns
                if abs(1 + day_return) > 1e-12:
                    drifted: dict[str, float] = {}
                    for sym, w in weights.items():
                        j = symbols.index(sym) if sym in symbols else -1
                        if j < 0:
                            continue
                        r_sym = float(filled_returns.iat[i, j])
                        new_w = w * (1 + r_sym) / (1 + day_return)
                        if abs(new_w) > 1e-12:
                            drifted[sym] = new_w
                    weights = drifted

            # ── 2. Rebalance at end of day (new weights apply tomorrow) ─
            days_since_rebalance += 1

            if (
                days_since_rebalance >= config.rebalance_frequency
                and i >= config.min_history
            ):
                new_weights = self._rebalance(
                    asset_returns=asset_returns,
                    symbols=symbols,
                    date_idx=i,
                    timestamp=dt,
                    signals=signals,
                    combiner=combiner,
                    portfolio_engine=portfolio_engine,
                    current_weights=weights,
                    portfolio_value=portfolio_value,
                    config=config,
                    feature_provider=feature_provider,
                )

                if new_weights is not None:
                    all_syms = set(new_weights) | set(weights)
                    turnover = sum(
                        abs(new_weights.get(s, 0.0) - weights.get(s, 0.0))
                        for s in all_syms
                    )
                    costs = turnover * portfolio_value * commission_rate
                    portfolio_value -= costs
                    total_costs += costs

                    dt_date = (
                        dt.date() if hasattr(dt, "date") else dt
                    )
                    rebalances.append(
                        RebalanceSnapshot(
                            date=dt_date,
                            weights=dict(new_weights),
                            portfolio_value=portfolio_value,
                            turnover=turnover,
                            transaction_costs=costs,
                            n_assets=sum(
                                1 for v in new_weights.values() if abs(v) > 1e-9
                            ),
                        )
                    )

                    weights = new_weights
                    days_since_rebalance = 0

            equity_values.append(portfolio_value)
            daily_rets.append(day_return)
            weights_records.append(dict(weights))

        # ── Build report ────────────────────────────────────────────────
        equity_curve = pd.Series(
            equity_values, index=dates, name="portfolio_value"
        )
        returns_series = pd.Series(daily_rets, index=dates, name="returns")
        weights_history = pd.DataFrame(
            weights_records, index=dates
        ).fillna(0.0)

        # Evaluate from first possible trade onwards
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

        # Benchmark
        bench_ret = 0.0
        bench_sharpe = 0.0
        active_ret = 0.0
        ir = 0.0
        te = 0.0
        if benchmark_returns is not None:
            bench_eval = benchmark_returns.iloc[eval_start:]
            bench_ret = float((1 + bench_eval).prod() - 1)
            bench_sharpe = m.sharpe_ratio(bench_eval)
            active_ret = total_ret - bench_ret
            active_daily = (
                eval_returns
                - bench_eval.reindex(eval_returns.index).fillna(0.0)
            )
            te_std = active_daily.std()
            te = (
                float(te_std * math.sqrt(252))
                if len(active_daily) > 1 and not math.isnan(te_std)
                else 0.0
            )
            ir = (
                float(
                    active_daily.mean()
                    / active_daily.std()
                    * math.sqrt(252)
                )
                if te > 1e-12
                else 0.0
            )

        start_date = (
            dates[0].date()
            if hasattr(dates[0], "date")
            else dates[0]
        )
        end_date = (
            dates[-1].date()
            if hasattr(dates[-1], "date")
            else dates[-1]
        )

        # ── Attribution ────────────────────────────────────────────────
        attribution_report = None
        factor_attr_report = None
        try:
            perf_attr = PerformanceAttributor()
            attribution_report = perf_attr.attribute(
                portfolio_returns=eval_returns,
                benchmark_returns=(
                    benchmark_returns.iloc[eval_start:]
                    if benchmark_returns is not None
                    else None
                ),
                weights_history=weights_history.iloc[eval_start:],
                sector_map=config.sector_map or None,
                asset_returns=asset_returns.iloc[eval_start:],
            )
        except Exception:
            logger.debug("Attribution computation failed — skipping")

        try:
            fa = FactorAttributor(min_observations=20)
            factor_attr_report = fa.attribute(
                portfolio_returns=eval_returns,
                asset_returns=asset_returns.iloc[eval_start:],
            )
        except Exception:
            logger.debug("Factor attribution failed — skipping")

        report = PortfolioBacktestReport(
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
            benchmark_return=bench_ret,
            benchmark_sharpe=bench_sharpe,
            active_return=active_ret,
            information_ratio=ir,
            tracking_error=te,
            equity_curve=equity_curve,
            returns_series=returns_series,
            weights_history=weights_history,
            rebalances=rebalances,
            attribution=attribution_report,
            factor_attribution=factor_attr_report,
        )

        logger.info(
            "Portfolio backtest: {} | Sharpe={:.2f} | "
            "MaxDD={:.1%} | CAGR={:.1%} | {} rebalances",
            config.name,
            sharpe,
            mdd,
            cagr_val,
            len(rebalances),
        )

        return report

    # ── Private: rebalance logic ──────────────────────────────────────

    def _rebalance(
        self,
        asset_returns: pd.DataFrame,
        symbols: list[str],
        date_idx: int,
        timestamp: datetime,
        signals: list[BaseSignal],
        combiner: AlphaCombiner,
        portfolio_engine: PortfolioEngine,
        current_weights: dict[str, float],
        portfolio_value: float,
        config: PortfolioBacktestConfig,
        feature_provider: FeatureProviderFn | None,
    ) -> dict[str, float] | None:
        """Compute target weights at a rebalance point.

        Returns *None* if construction fails or the rebalance threshold
        is not met.
        """
        visible = asset_returns.iloc[: date_idx + 1]
        ts = pd.Timestamp(timestamp).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # ── Compute signals ────────────────────────────────────────────
        universe_signals: dict[str, list[SignalOutput]] = {}

        for sym in symbols:
            sym_returns = visible[sym].dropna()
            if len(sym_returns) < config.min_history:
                continue

            sym_signals: list[SignalOutput] = []
            for signal in signals:
                features = (
                    feature_provider(sym, sym_returns)
                    if feature_provider is not None
                    else {"returns": sym_returns}
                )
                try:
                    output = signal.compute(sym, features, ts)
                    tagged = SignalOutput(
                        symbol=output.symbol,
                        timestamp=output.timestamp,
                        score=output.score,
                        confidence=output.confidence,
                        target_position=output.target_position,
                        metadata={
                            **output.metadata,
                            "signal_name": signal.name,
                        },
                    )
                    sym_signals.append(tagged)
                except Exception:
                    logger.debug(
                        "Signal {} failed for {} at {} — skipping",
                        signal.name,
                        sym,
                        timestamp,
                    )

            if sym_signals:
                universe_signals[sym] = sym_signals

        if not universe_signals:
            return None

        # ── Combine into alpha ─────────────────────────────────────────
        alpha_scores = combiner.combine_universe(ts, universe_signals)

        # ── Portfolio construction ─────────────────────────────────────
        try:
            construction = portfolio_engine.construct(
                alpha_scores=alpha_scores,
                returns_history=visible,
                current_weights=current_weights,
                portfolio_value=portfolio_value,
            )
        except Exception:
            logger.debug(
                "Portfolio construction failed at {} — holding",
                timestamp,
            )
            return None

        if not construction.rebalance_triggered:
            return None

        return construction.optimization.weights

    # ── Private: validation ───────────────────────────────────────────

    @staticmethod
    def _validate(returns: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalise the returns DataFrame."""
        if returns.empty:
            raise ValueError("returns DataFrame is empty")
        df = returns.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
