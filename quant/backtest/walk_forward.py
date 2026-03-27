"""Walk-forward analysis for overfitting detection and strategy validation.

Walk-forward analysis (WFA) rolls through time, alternating between
in-sample (IS) and out-of-sample (OOS) windows.  At each step the
strategy is evaluated on the IS window, then its performance is measured
on the subsequent OOS window using the IS-calibrated parameters.  By
comparing IS and OOS metrics the analyst can detect overfitting and
estimate realistic live performance.

Key outputs:

  * **Walk-forward efficiency (WFE)**: ``OOS_Sharpe / IS_Sharpe``.
    Values near 1.0 indicate robust parameter choices; values << 1
    suggest overfitting.
  * **Aggregated OOS equity curve**: the concatenation of all OOS
    segments gives a continuous equity curve that was *never* optimised
    against — the closest proxy for live performance.
  * **Per-fold diagnostics**: IS and OOS metrics for each fold.

Two window modes are supported:

  * **Rolling** (default): both IS and OOS windows slide forward by
    ``step_size`` at each fold.  IS window size is constant.
  * **Expanding**: the IS window grows over time (anchored at the start),
    while OOS window size and step remain fixed.

Usage::

    from quant.backtest.walk_forward import (
        WalkForwardAnalyzer,
        WalkForwardConfig,
    )
    from quant.signals.factors import VolatilitySignal

    analyzer = WalkForwardAnalyzer()
    result = analyzer.run(
        returns=daily_returns_df,
        signals=[VolatilitySignal()],
        config=WalkForwardConfig(
            is_window=252,
            oos_window=63,
            step_size=63,
        ),
    )
    print(result.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from quant.backtest import metrics as m
from quant.backtest.portfolio_backtest import (
    FeatureProviderFn,
    PortfolioBacktestConfig,
    PortfolioBacktestEngine,
    PortfolioBacktestReport,
)
from quant.portfolio.alpha import CombinationMethod
from quant.portfolio.engine import PortfolioConfig
from quant.signals.base import BaseSignal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis.

    Attributes:
        is_window: Number of trading days in the in-sample window.
        oos_window: Number of trading days in the out-of-sample window.
        step_size: Number of trading days to advance between folds.
            Typically equal to ``oos_window`` for non-overlapping OOS.
        expanding: If True, the IS window grows from the start of data
            (anchored).  If False (default), IS window is a fixed rolling
            window of ``is_window`` days.
        portfolio_config: Portfolio construction settings used in each fold.
        rebalance_frequency: Trading days between rebalances within each fold.
        commission_bps: One-way transaction cost in basis points.
        initial_capital: Starting capital for each fold.
        combination_method: How to combine multiple signal outputs.
        signal_weights: Static weights for STATIC_WEIGHT combination.
        min_history: Minimum trading days before first trade within a fold.
        name: Label for reports.
    """

    is_window: int = 252
    oos_window: int = 63
    step_size: int = 63
    expanding: bool = False
    portfolio_config: PortfolioConfig = field(default_factory=PortfolioConfig)
    rebalance_frequency: int = 21
    commission_bps: float = 10.0
    initial_capital: float = 1_000_000.0
    combination_method: CombinationMethod = CombinationMethod.EQUAL_WEIGHT
    signal_weights: dict[str, float] | None = None
    min_history: int = 60
    name: str = "walk_forward"


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Metrics for a single walk-forward fold.

    Attributes:
        fold_index: 0-based fold number.
        is_start: First date of the in-sample window.
        is_end: Last date of the in-sample window.
        oos_start: First date of the out-of-sample window.
        oos_end: Last date of the out-of-sample window.
        is_sharpe: Annualised Sharpe on the IS window.
        is_return: Total return on the IS window.
        is_max_drawdown: Max drawdown on the IS window.
        oos_sharpe: Annualised Sharpe on the OOS window.
        oos_return: Total return on the OOS window.
        oos_max_drawdown: Max drawdown on the OOS window.
        wfe: Walk-forward efficiency (OOS Sharpe / IS Sharpe).
    """

    fold_index: int
    is_start: date
    is_end: date
    oos_start: date
    oos_end: date
    is_sharpe: float
    is_return: float
    is_max_drawdown: float
    oos_sharpe: float
    oos_return: float
    oos_max_drawdown: float
    wfe: float


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward analysis results.

    Contains per-fold diagnostics, aggregated OOS metrics, and
    overfitting detection indicators.
    """

    name: str
    n_folds: int
    start_date: date
    end_date: date

    # Aggregated OOS metrics
    oos_total_return: float
    oos_sharpe: float
    oos_max_drawdown: float
    oos_volatility: float

    # Aggregated IS metrics (for comparison)
    is_mean_sharpe: float
    is_mean_return: float

    # Walk-forward efficiency
    mean_wfe: float
    median_wfe: float

    # Overfitting indicator
    oos_vs_is_sharpe_ratio: float

    # Per-fold details
    folds: list[FoldResult] = field(repr=False)

    # OOS equity curve (concatenated)
    oos_equity_curve: pd.Series = field(repr=False)
    oos_returns: pd.Series = field(repr=False)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Walk-Forward Analysis : {self.name}",
            f"Period                : {self.start_date} -> {self.end_date}",
            f"Folds                 : {self.n_folds}",
            "-" * 50,
            f"OOS Total Return      : {self.oos_total_return:+.2%}",
            f"OOS Sharpe            : {self.oos_sharpe:.2f}",
            f"OOS Volatility        : {self.oos_volatility:.2%}",
            f"OOS Max Drawdown      : {self.oos_max_drawdown:.2%}",
            "-" * 50,
            f"IS Mean Sharpe        : {self.is_mean_sharpe:.2f}",
            f"IS Mean Return        : {self.is_mean_return:+.2%}",
            "-" * 50,
            f"Mean WFE              : {self.mean_wfe:.2f}",
            f"Median WFE            : {self.median_wfe:.2f}",
            f"OOS/IS Sharpe Ratio   : {self.oos_vs_is_sharpe_ratio:.2f}",
        ]

        # Overfitting assessment
        if self.mean_wfe >= 0.5:
            assessment = "LOW overfitting risk"
        elif self.mean_wfe >= 0.25:
            assessment = "MODERATE overfitting risk"
        else:
            assessment = "HIGH overfitting risk"
        lines.append(f"Assessment            : {assessment}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class WalkForwardAnalyzer:
    """Walk-forward analysis engine.

    Splits historical data into rolling or expanding IS/OOS folds,
    runs a portfolio backtest on each fold, and aggregates the OOS
    results to produce overfitting-adjusted performance estimates.
    """

    def run(
        self,
        returns: pd.DataFrame,
        signals: list[BaseSignal],
        config: WalkForwardConfig | None = None,
        feature_provider: FeatureProviderFn | None = None,
    ) -> WalkForwardResult:
        """Run walk-forward analysis.

        Args:
            returns: DataFrame of daily returns (DatetimeIndex x symbols).
            signals: Signal instances to evaluate.
            config: Walk-forward configuration.
            feature_provider: Optional feature callback for signals.

        Returns:
            :class:`WalkForwardResult` with per-fold and aggregate metrics.

        Raises:
            ValueError: If data is insufficient for at least one fold.
        """
        if config is None:
            config = WalkForwardConfig()

        returns = self._validate(returns)
        n_days = len(returns)
        min_required = config.is_window + config.oos_window
        if n_days < min_required:
            raise ValueError(
                f"Need at least {min_required} days for one fold, "
                f"got {n_days}"
            )

        # Generate fold boundaries
        folds_spec = self._generate_folds(n_days, config)
        if not folds_spec:
            raise ValueError("Could not generate any folds with given config")

        backtest_engine = PortfolioBacktestEngine()
        fold_results: list[FoldResult] = []
        oos_equity_segments: list[pd.Series] = []
        oos_return_segments: list[pd.Series] = []

        for fold_idx, (is_start, is_end, oos_start, oos_end) in enumerate(
            folds_spec
        ):
            is_data = returns.iloc[is_start:is_end]
            oos_data = returns.iloc[oos_start:oos_end]

            # ── Run IS backtest ────────────────────────────────────────
            is_report = self._run_fold(
                is_data, signals, config, backtest_engine, feature_provider
            )

            # ── Run OOS backtest ───────────────────────────────────────
            oos_report = self._run_fold(
                oos_data, signals, config, backtest_engine, feature_provider
            )

            # WFE: OOS Sharpe / IS Sharpe (cap at 2.0, floor at -2.0)
            if abs(is_report.sharpe_ratio) > 0.01:
                wfe = oos_report.sharpe_ratio / is_report.sharpe_ratio
                wfe = max(-2.0, min(2.0, wfe))
            else:
                wfe = 0.0

            fold_results.append(
                FoldResult(
                    fold_index=fold_idx,
                    is_start=is_data.index[0].date(),
                    is_end=is_data.index[-1].date(),
                    oos_start=oos_data.index[0].date(),
                    oos_end=oos_data.index[-1].date(),
                    is_sharpe=is_report.sharpe_ratio,
                    is_return=is_report.total_return,
                    is_max_drawdown=is_report.max_drawdown,
                    oos_sharpe=oos_report.sharpe_ratio,
                    oos_return=oos_report.total_return,
                    oos_max_drawdown=oos_report.max_drawdown,
                    wfe=wfe,
                )
            )

            # Collect OOS segments for aggregated curve
            oos_equity_segments.append(oos_report.equity_curve)
            oos_return_segments.append(oos_report.returns_series)

        # ── Aggregate results ──────────────────────────────────────────
        return self._aggregate(
            fold_results,
            oos_equity_segments,
            oos_return_segments,
            returns,
            config,
        )

    # ── Private: fold generation ──────────────────────────────────────

    @staticmethod
    def _generate_folds(
        n_days: int, config: WalkForwardConfig
    ) -> list[tuple[int, int, int, int]]:
        """Generate (is_start, is_end, oos_start, oos_end) index tuples.

        Indices are half-open: ``data.iloc[start:end]``.
        """
        folds: list[tuple[int, int, int, int]] = []
        pos = 0

        while True:
            if config.expanding:
                is_start = 0
                is_end = config.is_window + pos
            else:
                is_start = pos
                is_end = pos + config.is_window

            oos_start = is_end
            oos_end = oos_start + config.oos_window

            if oos_end > n_days:
                break

            folds.append((is_start, is_end, oos_start, oos_end))
            pos += config.step_size

        return folds

    # ── Private: run a single fold ────────────────────────────────────

    @staticmethod
    def _run_fold(
        data: pd.DataFrame,
        signals: list[BaseSignal],
        config: WalkForwardConfig,
        engine: PortfolioBacktestEngine,
        feature_provider: FeatureProviderFn | None,
    ) -> PortfolioBacktestReport:
        """Run a portfolio backtest on a data slice."""
        fold_config = PortfolioBacktestConfig(
            rebalance_frequency=config.rebalance_frequency,
            commission_bps=config.commission_bps,
            initial_capital=config.initial_capital,
            portfolio_config=config.portfolio_config,
            combination_method=config.combination_method,
            signal_weights=config.signal_weights,
            min_history=min(config.min_history, len(data) // 3),
            name=config.name,
        )
        return engine.run(data, signals, fold_config, feature_provider)

    # ── Private: aggregate fold results ───────────────────────────────

    @staticmethod
    def _aggregate(
        folds: list[FoldResult],
        oos_equity_segments: list[pd.Series],
        oos_return_segments: list[pd.Series],
        full_returns: pd.DataFrame,
        config: WalkForwardConfig,
    ) -> WalkForwardResult:
        """Build aggregated WalkForwardResult from per-fold data."""
        n_folds = len(folds)

        # Concatenate OOS return segments (skip overlapping dates)
        all_oos_returns: list[pd.Series] = []
        seen_dates: set = set()
        for seg in oos_return_segments:
            mask = ~seg.index.isin(seen_dates)
            filtered = seg[mask]
            all_oos_returns.append(filtered)
            seen_dates.update(filtered.index)

        if all_oos_returns:
            oos_returns = pd.concat(all_oos_returns).sort_index()
        else:
            oos_returns = pd.Series(dtype=float)

        # Build OOS equity curve from concatenated returns
        if not oos_returns.empty:
            oos_cumulative = (1 + oos_returns).cumprod()
            oos_equity = oos_cumulative * config.initial_capital
        else:
            oos_equity = pd.Series(dtype=float)

        # OOS aggregate metrics
        oos_total_return = (
            float((1 + oos_returns).prod() - 1)
            if not oos_returns.empty
            else 0.0
        )
        oos_sharpe = (
            m.sharpe_ratio(oos_returns) if len(oos_returns) > 1 else 0.0
        )
        oos_mdd = (
            m.max_drawdown(oos_cumulative) if not oos_returns.empty else 0.0
        )
        oos_vol = (
            float(oos_returns.std() * math.sqrt(252))
            if len(oos_returns) > 1
            else 0.0
        )

        # IS aggregate metrics
        is_sharpes = [f.is_sharpe for f in folds]
        is_returns = [f.is_return for f in folds]
        is_mean_sharpe = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        is_mean_return = float(np.mean(is_returns)) if is_returns else 0.0

        # WFE
        wfes = [f.wfe for f in folds]
        mean_wfe = float(np.mean(wfes)) if wfes else 0.0
        median_wfe = float(np.median(wfes)) if wfes else 0.0

        # OOS vs IS Sharpe ratio
        oos_vs_is = (
            oos_sharpe / is_mean_sharpe
            if abs(is_mean_sharpe) > 0.01
            else 0.0
        )

        start_date = (
            full_returns.index[0].date()
            if hasattr(full_returns.index[0], "date")
            else full_returns.index[0]
        )
        end_date = (
            full_returns.index[-1].date()
            if hasattr(full_returns.index[-1], "date")
            else full_returns.index[-1]
        )

        return WalkForwardResult(
            name=config.name,
            n_folds=n_folds,
            start_date=start_date,
            end_date=end_date,
            oos_total_return=oos_total_return,
            oos_sharpe=oos_sharpe,
            oos_max_drawdown=oos_mdd,
            oos_volatility=oos_vol,
            is_mean_sharpe=is_mean_sharpe,
            is_mean_return=is_mean_return,
            mean_wfe=mean_wfe,
            median_wfe=median_wfe,
            oos_vs_is_sharpe_ratio=oos_vs_is,
            folds=folds,
            oos_equity_curve=oos_equity,
            oos_returns=oos_returns,
        )

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
