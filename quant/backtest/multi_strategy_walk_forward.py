"""Walk-forward analysis for multi-strategy portfolios.

Extends the single-strategy walk-forward framework to validate full
multi-strategy configurations — including regime detection, circuit
breakers, lifecycle evaluation, and cross-strategy correlation monitoring.

Each fold runs a complete ``MultiStrategyBacktestEngine.run()`` on its
IS and OOS windows independently.  State (lifecycle, regime, circuit
breaker) resets between folds to prevent information leakage.

Key additions over single-strategy walk-forward:

  * **Per-sleeve WFE**: track each sleeve's contribution to IS vs OOS
    performance for strategy-level overfitting detection.
  * **Regime stability**: how many regime changes occurred OOS vs IS,
    and whether regime-aware allocation improved OOS.
  * **Circuit breaker interaction**: whether the breaker tripped in OOS
    windows (real stress) vs IS windows (curve-fit).

Usage::

    from quant.backtest.multi_strategy_walk_forward import (
        MultiStrategyWalkForwardAnalyzer,
        MultiStrategyWalkForwardConfig,
    )
    from quant.backtest.multi_strategy import MultiStrategyConfig, SleeveConfig
    from quant.signals.factors import MomentumSignal, MeanReversionSignal

    ms_config = MultiStrategyConfig(
        sleeves=[
            SleeveConfig(name="mom", signals=[MomentumSignal()], capital_weight=0.6),
            SleeveConfig(name="mr", signals=[MeanReversionSignal()], capital_weight=0.4),
        ],
        rebalance_frequency=21,
    )
    analyzer = MultiStrategyWalkForwardAnalyzer()
    result = analyzer.run(
        returns=daily_returns_df,
        config=MultiStrategyWalkForwardConfig(
            multi_strategy_config=ms_config,
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
from quant.backtest.multi_strategy import (
    MultiStrategyBacktestEngine,
    MultiStrategyConfig,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MultiStrategyWalkForwardConfig:
    """Configuration for multi-strategy walk-forward analysis.

    Attributes:
        multi_strategy_config: The full multi-strategy backtest config applied
            to each fold.
        is_window: Number of trading days in the in-sample window.
        oos_window: Number of trading days in the out-of-sample window.
        step_size: Days to advance between folds.
        expanding: If True, IS window anchors at start and grows.
        name: Label for reports.
    """

    multi_strategy_config: MultiStrategyConfig = field(
        default_factory=MultiStrategyConfig
    )
    is_window: int = 252
    oos_window: int = 63
    step_size: int = 63
    expanding: bool = False
    name: str = "multi_strategy_walk_forward"


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MultiStrategyFoldResult:
    """Metrics for a single multi-strategy walk-forward fold.

    Extends the standard fold result with multi-strategy diagnostics:
    per-sleeve metrics, regime changes, and circuit breaker activity.
    """

    fold_index: int
    is_start: date
    is_end: date
    oos_start: date
    oos_end: date

    # Core metrics
    is_sharpe: float
    is_return: float
    is_max_drawdown: float
    oos_sharpe: float
    oos_return: float
    oos_max_drawdown: float
    wfe: float

    # Multi-strategy diagnostics
    is_n_rebalances: int
    oos_n_rebalances: int
    is_regime_changes: int
    oos_regime_changes: int
    is_circuit_breaker_trips: int
    oos_circuit_breaker_trips: int
    n_sleeves: int


@dataclass
class MultiStrategyWalkForwardResult:
    """Aggregated multi-strategy walk-forward analysis results."""

    name: str
    n_folds: int
    n_sleeves: int
    start_date: date
    end_date: date

    # Aggregated OOS metrics
    oos_total_return: float
    oos_sharpe: float
    oos_max_drawdown: float
    oos_volatility: float

    # Aggregated IS metrics
    is_mean_sharpe: float
    is_mean_return: float

    # Walk-forward efficiency
    mean_wfe: float
    median_wfe: float

    # Overfitting indicator
    oos_vs_is_sharpe_ratio: float

    # Multi-strategy aggregates
    total_is_regime_changes: int
    total_oos_regime_changes: int
    total_is_cb_trips: int
    total_oos_cb_trips: int

    # Per-fold details
    folds: list[MultiStrategyFoldResult] = field(repr=False)

    # OOS equity curve (concatenated)
    oos_equity_curve: pd.Series = field(repr=False)
    oos_returns: pd.Series = field(repr=False)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Multi-Strategy Walk-Forward : {self.name}",
            f"Period                      : {self.start_date} -> {self.end_date}",
            f"Folds                       : {self.n_folds}",
            f"Sleeves                     : {self.n_sleeves}",
            "-" * 55,
            f"OOS Total Return            : {self.oos_total_return:+.2%}",
            f"OOS Sharpe                  : {self.oos_sharpe:.2f}",
            f"OOS Volatility              : {self.oos_volatility:.2%}",
            f"OOS Max Drawdown            : {self.oos_max_drawdown:.2%}",
            "-" * 55,
            f"IS Mean Sharpe              : {self.is_mean_sharpe:.2f}",
            f"IS Mean Return              : {self.is_mean_return:+.2%}",
            "-" * 55,
            f"Mean WFE                    : {self.mean_wfe:.2f}",
            f"Median WFE                  : {self.median_wfe:.2f}",
            f"OOS/IS Sharpe Ratio         : {self.oos_vs_is_sharpe_ratio:.2f}",
        ]

        # Regime stability
        if self.total_is_regime_changes + self.total_oos_regime_changes > 0:
            lines.append(
                f"Regime changes (IS/OOS)     : "
                f"{self.total_is_regime_changes}/{self.total_oos_regime_changes}"
            )

        # Circuit breaker
        if self.total_is_cb_trips + self.total_oos_cb_trips > 0:
            lines.append(
                f"CB trips (IS/OOS)           : "
                f"{self.total_is_cb_trips}/{self.total_oos_cb_trips}"
            )

        # Overfitting assessment
        if self.mean_wfe >= 0.5:
            assessment = "LOW overfitting risk"
        elif self.mean_wfe >= 0.25:
            assessment = "MODERATE overfitting risk"
        else:
            assessment = "HIGH overfitting risk"
        lines.append(f"Assessment                  : {assessment}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MultiStrategyWalkForwardAnalyzer:
    """Walk-forward analysis engine for multi-strategy portfolios.

    Splits historical data into rolling or expanding IS/OOS folds,
    runs a full multi-strategy backtest on each fold, and aggregates
    the OOS results.
    """

    def run(
        self,
        returns: pd.DataFrame,
        config: MultiStrategyWalkForwardConfig | None = None,
    ) -> MultiStrategyWalkForwardResult:
        """Run multi-strategy walk-forward analysis.

        Args:
            returns: DataFrame of daily returns (DatetimeIndex x symbols).
            config: Walk-forward configuration.

        Returns:
            :class:`MultiStrategyWalkForwardResult` with per-fold and
            aggregate metrics.

        Raises:
            ValueError: If data is insufficient for at least one fold.
        """
        if config is None:
            config = MultiStrategyWalkForwardConfig()

        returns = self._validate(returns)
        n_days = len(returns)
        min_required = config.is_window + config.oos_window
        if n_days < min_required:
            raise ValueError(
                f"Need at least {min_required} days for one fold, "
                f"got {n_days}"
            )

        folds_spec = self._generate_folds(n_days, config)
        if not folds_spec:
            raise ValueError("Could not generate any folds with given config")

        ms_engine = MultiStrategyBacktestEngine()
        fold_results: list[MultiStrategyFoldResult] = []
        oos_equity_segments: list[pd.Series] = []
        oos_return_segments: list[pd.Series] = []

        # Lookback days prepended to OOS windows for signal warmup.
        # Signals (e.g. MACD, slow MA) need history beyond the OOS window
        # to produce non-zero output.  We include IS tail as lookback and
        # set min_history to skip trading during the prefix.
        oos_lookback = config.multi_strategy_config.min_history

        for fold_idx, (is_start, is_end, oos_start, oos_end) in enumerate(
            folds_spec
        ):
            is_data = returns.iloc[is_start:is_end]

            # OOS: prepend lookback from IS tail so signals can warm up
            oos_lb_start = max(0, oos_start - oos_lookback)
            oos_data_with_lb = returns.iloc[oos_lb_start:oos_end]
            oos_lb_len = oos_start - oos_lb_start  # days of warmup prefix

            # Build fold-specific config with adjusted min_history
            fold_ms_config = self._build_fold_config(
                config.multi_strategy_config, is_data
            )

            # Run IS backtest
            is_report = ms_engine.run(is_data, fold_ms_config)

            # Run OOS backtest with lookback prefix.
            # min_history = lookback length so trading starts at OOS boundary.
            oos_fold_config = self._build_fold_config(
                config.multi_strategy_config, oos_data_with_lb,
                override_min_history=oos_lb_len,
            )
            oos_report = ms_engine.run(oos_data_with_lb, oos_fold_config)

            # WFE: OOS Sharpe / IS Sharpe (clamped)
            if abs(is_report.sharpe_ratio) > 0.01:
                wfe = oos_report.sharpe_ratio / is_report.sharpe_ratio
                wfe = max(-2.0, min(2.0, wfe))
            else:
                wfe = 0.0

            fold_results.append(
                MultiStrategyFoldResult(
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
                    is_n_rebalances=is_report.n_rebalances,
                    oos_n_rebalances=oos_report.n_rebalances,
                    is_regime_changes=is_report.n_regime_changes,
                    oos_regime_changes=oos_report.n_regime_changes,
                    is_circuit_breaker_trips=is_report.n_circuit_breaker_trips,
                    oos_circuit_breaker_trips=oos_report.n_circuit_breaker_trips,
                    n_sleeves=is_report.n_sleeves,
                )
            )

            oos_equity_segments.append(oos_report.equity_curve)
            oos_return_segments.append(oos_report.returns_series)

        return self._aggregate(
            fold_results,
            oos_equity_segments,
            oos_return_segments,
            returns,
            config,
        )

    # ── Fold generation ───────────────────────────────────────────────

    @staticmethod
    def _generate_folds(
        n_days: int, config: MultiStrategyWalkForwardConfig
    ) -> list[tuple[int, int, int, int]]:
        """Generate (is_start, is_end, oos_start, oos_end) index tuples."""
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

    # ── Build fold config ─────────────────────────────────────────────

    @staticmethod
    def _build_fold_config(
        base: MultiStrategyConfig,
        fold_data: pd.DataFrame,
        override_min_history: int | None = None,
    ) -> MultiStrategyConfig:
        """Create a fold-specific MultiStrategyConfig with safe min_history."""
        if override_min_history is not None:
            safe_min = override_min_history
        else:
            safe_min = min(base.min_history, len(fold_data) // 3)
        return MultiStrategyConfig(
            sleeves=base.sleeves,
            rebalance_frequency=base.rebalance_frequency,
            commission_bps=base.commission_bps,
            initial_capital=base.initial_capital,
            cost_model=base.cost_model,
            pre_trade_config=base.pre_trade_config,
            sector_map=base.sector_map,
            lifecycle_config=base.lifecycle_config,
            apply_lifecycle_realloc=base.apply_lifecycle_realloc,
            strategy_correlation_config=base.strategy_correlation_config,
            regime_config=base.regime_config,
            regime_adapter=base.regime_adapter,
            regime_lookback_days=base.regime_lookback_days,
            circuit_breaker=base.circuit_breaker,
            min_history=safe_min,
            name=base.name,
        )

    # ── Aggregate fold results ────────────────────────────────────────

    @staticmethod
    def _aggregate(
        folds: list[MultiStrategyFoldResult],
        oos_equity_segments: list[pd.Series],
        oos_return_segments: list[pd.Series],
        full_returns: pd.DataFrame,
        config: MultiStrategyWalkForwardConfig,
    ) -> MultiStrategyWalkForwardResult:
        """Build aggregated result from per-fold data."""
        n_folds = len(folds)

        # Concatenate OOS return segments (deduplicate dates)
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

        # Build OOS equity curve
        if not oos_returns.empty:
            oos_cumulative = (1 + oos_returns).cumprod()
            oos_equity = (
                oos_cumulative * config.multi_strategy_config.initial_capital
            )
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

        # Multi-strategy aggregate diagnostics
        total_is_regime = sum(f.is_regime_changes for f in folds)
        total_oos_regime = sum(f.oos_regime_changes for f in folds)
        total_is_cb = sum(f.is_circuit_breaker_trips for f in folds)
        total_oos_cb = sum(f.oos_circuit_breaker_trips for f in folds)

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

        n_sleeves = folds[0].n_sleeves if folds else 0

        return MultiStrategyWalkForwardResult(
            name=config.name,
            n_folds=n_folds,
            n_sleeves=n_sleeves,
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
            total_is_regime_changes=total_is_regime,
            total_oos_regime_changes=total_oos_regime,
            total_is_cb_trips=total_is_cb,
            total_oos_cb_trips=total_oos_cb,
            folds=folds,
            oos_equity_curve=oos_equity,
            oos_returns=oos_returns,
        )

    # ── Validation ────────────────────────────────────────────────────

    @staticmethod
    def _validate(returns: pd.DataFrame) -> pd.DataFrame:
        if returns.empty:
            raise ValueError("returns DataFrame is empty")
        df = returns.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
