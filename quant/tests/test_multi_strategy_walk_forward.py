"""Tests for multi-strategy walk-forward analysis (QUA-74)."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quant.backtest.multi_strategy import (
    MultiStrategyConfig,
    SleeveConfig,
)
from quant.backtest.multi_strategy_walk_forward import (
    MultiStrategyFoldResult,
    MultiStrategyWalkForwardAnalyzer,
    MultiStrategyWalkForwardConfig,
    MultiStrategyWalkForwardResult,
)
from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.engine import PortfolioConfig
from quant.portfolio.lifecycle import LifecycleConfig
from quant.portfolio.optimizers import OptimizationMethod
from quant.signals.base import BaseSignal, SignalOutput
from quant.signals.regime import RegimeConfig, RegimeWeightAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOLS = ["AAPL", "GOOG", "MSFT"]


class _StubSignal(BaseSignal):
    """Deterministic stub signal."""

    def __init__(self, name: str = "stub", offset: float = 0.0) -> None:
        self._name = name
        self._offset = offset

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> list[str]:
        return []

    def compute(
        self, symbol: str, features: dict[str, pd.Series], timestamp: datetime
    ) -> SignalOutput:
        score = 0.5 + self._offset
        return SignalOutput(
            symbol=symbol,
            timestamp=timestamp,
            score=score,
            confidence=0.8,
            target_position=score * 0.8,
            metadata={"signal_name": self._name},
        )


def _make_returns(n_days: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic daily returns with enough data for walk-forward."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0005, 0.01, size=n_days)
    idio = rng.normal(0, 0.015, size=(n_days, len(SYMBOLS)))
    betas = rng.uniform(0.5, 1.5, size=len(SYMBOLS))
    data = factor[:, None] * betas[None, :] + idio
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(data, index=dates, columns=SYMBOLS)


def _make_ms_config(
    n_sleeves: int = 2,
    lifecycle: bool = False,
    regime: bool = False,
) -> MultiStrategyConfig:
    """Build a multi-strategy config for testing."""
    names = ["momentum", "mean_rev", "value", "carry"]
    sleeves = []
    for i in range(n_sleeves):
        sleeves.append(
            SleeveConfig(
                name=names[i % len(names)],
                signals=[_StubSignal(names[i % len(names)], offset=i * 0.1)],
                capital_weight=1.0 / n_sleeves,
                strategy_type=names[i % len(names)],
                portfolio_config=PortfolioConfig(
                    optimization_method=OptimizationMethod.RISK_PARITY,
                    constraints=PortfolioConstraints(
                        long_only=True, max_weight=0.5, max_gross_exposure=1.0
                    ),
                ),
            )
        )

    lc = None
    if lifecycle:
        lc = LifecycleConfig(
            drawdown_watch=0.15,
            drawdown_degraded=0.25,
            drawdown_critical=0.40,
            eval_window=63,
        )

    rc = None
    ra = None
    if regime:
        rc = RegimeConfig()
        ra = RegimeWeightAdapter()

    return MultiStrategyConfig(
        sleeves=sleeves,
        rebalance_frequency=21,
        commission_bps=10.0,
        min_history=60,
        lifecycle_config=lc,
        apply_lifecycle_realloc=lifecycle,
        regime_config=rc,
        regime_adapter=ra,
    )


def _make_wf_config(
    ms_config: MultiStrategyConfig | None = None,
    is_window: int = 200,
    oos_window: int = 63,
    step_size: int = 63,
    expanding: bool = False,
) -> MultiStrategyWalkForwardConfig:
    return MultiStrategyWalkForwardConfig(
        multi_strategy_config=ms_config or _make_ms_config(),
        is_window=is_window,
        oos_window=oos_window,
        step_size=step_size,
        expanding=expanding,
    )


# ---------------------------------------------------------------------------
# Fold generation tests
# ---------------------------------------------------------------------------


class TestFoldGeneration:
    def test_rolling_folds_correct_count(self):
        config = _make_wf_config(is_window=100, oos_window=50, step_size=50)
        analyzer = MultiStrategyWalkForwardAnalyzer()
        folds = analyzer._generate_folds(500, config)
        assert len(folds) > 0
        for _, is_e, oos_s, oos_e in folds:
            assert is_e == oos_s
            assert oos_e - oos_s == 50
            assert oos_e <= 500

    def test_expanding_folds_grow_is_window(self):
        config = _make_wf_config(
            is_window=100, oos_window=50, step_size=50, expanding=True
        )
        analyzer = MultiStrategyWalkForwardAnalyzer()
        folds = analyzer._generate_folds(500, config)
        assert len(folds) >= 2
        for i, (is_s, is_e, _, _) in enumerate(folds):
            assert is_s == 0
            assert is_e == 100 + i * 50

    def test_no_folds_if_data_too_short(self):
        config = _make_wf_config(is_window=100, oos_window=50)
        analyzer = MultiStrategyWalkForwardAnalyzer()
        folds = analyzer._generate_folds(140, config)
        assert len(folds) == 0

    def test_oos_never_exceeds_data(self):
        config = _make_wf_config(is_window=100, oos_window=50, step_size=50)
        analyzer = MultiStrategyWalkForwardAnalyzer()
        folds = analyzer._generate_folds(500, config)
        for _, _, _, oos_e in folds:
            assert oos_e <= 500


# ---------------------------------------------------------------------------
# Core analyzer tests
# ---------------------------------------------------------------------------


class TestMultiStrategyWalkForward:
    def test_run_returns_result(self):
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), _make_wf_config())
        assert isinstance(result, MultiStrategyWalkForwardResult)

    def test_result_folds_populated(self):
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), _make_wf_config())
        assert result.n_folds > 0
        assert len(result.folds) == result.n_folds

    def test_fold_results_are_multi_strategy_type(self):
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), _make_wf_config())
        for fold in result.folds:
            assert isinstance(fold, MultiStrategyFoldResult)

    def test_oos_metrics_finite(self):
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), _make_wf_config())
        assert np.isfinite(result.oos_sharpe)
        assert np.isfinite(result.oos_total_return)
        assert np.isfinite(result.oos_max_drawdown)
        assert np.isfinite(result.oos_volatility)

    def test_oos_returns_populated(self):
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), _make_wf_config())
        assert not result.oos_returns.empty
        assert not result.oos_equity_curve.empty

    def test_wfe_per_fold(self):
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), _make_wf_config())
        for fold in result.folds:
            assert -2.0 <= fold.wfe <= 2.0

    def test_mean_and_median_wfe(self):
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), _make_wf_config())
        assert np.isfinite(result.mean_wfe)
        assert np.isfinite(result.median_wfe)

    def test_oos_vs_is_sharpe_ratio(self):
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), _make_wf_config())
        assert np.isfinite(result.oos_vs_is_sharpe_ratio)

    def test_n_sleeves_tracked(self):
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), _make_wf_config())
        assert result.n_sleeves == 2

    def test_per_fold_rebalance_counts(self):
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), _make_wf_config())
        for fold in result.folds:
            assert fold.is_n_rebalances >= 0
            assert fold.oos_n_rebalances >= 0
            assert fold.n_sleeves == 2


# ---------------------------------------------------------------------------
# Window modes
# ---------------------------------------------------------------------------


class TestWindowModes:
    def test_expanding_produces_result(self):
        config = _make_wf_config(expanding=True)
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert result.n_folds > 0

    def test_expanding_vs_rolling_fold_counts(self):
        rolling = _make_wf_config(is_window=200, oos_window=63, step_size=63)
        expanding = _make_wf_config(
            is_window=200, oos_window=63, step_size=63, expanding=True
        )
        analyzer = MultiStrategyWalkForwardAnalyzer()
        r_rolling = analyzer.run(_make_returns(), rolling)
        r_expanding = analyzer.run(_make_returns(), expanding)
        # Expanding uses less data per step so may have more folds
        assert r_rolling.n_folds > 0
        assert r_expanding.n_folds > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_insufficient_data_raises(self):
        short = _make_returns(n_days=50)
        config = _make_wf_config(is_window=200, oos_window=63)
        analyzer = MultiStrategyWalkForwardAnalyzer()
        with pytest.raises(ValueError, match="Need at least"):
            analyzer.run(short, config)

    def test_empty_data_raises(self):
        empty = pd.DataFrame()
        config = _make_wf_config()
        analyzer = MultiStrategyWalkForwardAnalyzer()
        with pytest.raises(ValueError, match="empty"):
            analyzer.run(empty, config)

    def test_single_fold(self):
        # Just enough for one fold
        returns = _make_returns(n_days=300)
        config = _make_wf_config(is_window=200, oos_window=63, step_size=200)
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(returns, config)
        assert result.n_folds == 1

    def test_default_config(self):
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(n_days=400))
        assert isinstance(result, MultiStrategyWalkForwardResult)


# ---------------------------------------------------------------------------
# Multi-strategy specific diagnostics
# ---------------------------------------------------------------------------


class TestMultiStrategyDiagnostics:
    def test_fold_dates_valid(self):
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), _make_wf_config())
        for fold in result.folds:
            assert fold.is_start < fold.is_end
            assert fold.oos_start < fold.oos_end
            assert fold.is_end <= fold.oos_start

    def test_regime_changes_tracked(self):
        ms_config = _make_ms_config(regime=True)
        config = _make_wf_config(ms_config=ms_config)
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), config)
        # Regime changes should be non-negative
        assert result.total_is_regime_changes >= 0
        assert result.total_oos_regime_changes >= 0
        for fold in result.folds:
            assert fold.is_regime_changes >= 0
            assert fold.oos_regime_changes >= 0

    def test_circuit_breaker_tracked(self):
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), _make_wf_config())
        assert result.total_is_cb_trips >= 0
        assert result.total_oos_cb_trips >= 0
        for fold in result.folds:
            assert fold.is_circuit_breaker_trips >= 0
            assert fold.oos_circuit_breaker_trips >= 0

    def test_lifecycle_in_folds(self):
        ms_config = _make_ms_config(lifecycle=True)
        config = _make_wf_config(ms_config=ms_config)
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert result.n_folds > 0
        # Lifecycle is per-fold isolated; no leakage between folds
        for fold in result.folds:
            assert isinstance(fold.is_sharpe, float)

    def test_three_sleeves(self):
        ms_config = _make_ms_config(n_sleeves=3)
        config = _make_wf_config(ms_config=ms_config)
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert result.n_sleeves == 3
        for fold in result.folds:
            assert fold.n_sleeves == 3


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), _make_wf_config())
        summary = result.summary()
        assert "Multi-Strategy Walk-Forward" in summary
        assert "OOS" in summary
        assert "WFE" in summary
        assert "Assessment" in summary

    def test_summary_includes_fold_count(self):
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), _make_wf_config())
        summary = result.summary()
        assert str(result.n_folds) in summary

    def test_summary_includes_sleeve_count(self):
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), _make_wf_config())
        summary = result.summary()
        assert str(result.n_sleeves) in summary


# ---------------------------------------------------------------------------
# Integration: combined features
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_regime_and_lifecycle_combined(self):
        ms_config = _make_ms_config(lifecycle=True, regime=True)
        config = _make_wf_config(ms_config=ms_config)
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert result.n_folds > 0
        assert isinstance(result.oos_sharpe, float)

    def test_multiple_folds_oos_equity_continuous(self):
        config = _make_wf_config(is_window=200, oos_window=63, step_size=63)
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), config)
        # OOS equity should be sorted by date
        assert result.oos_equity_curve.index.is_monotonic_increasing

    def test_oos_returns_no_duplicates(self):
        config = _make_wf_config(is_window=200, oos_window=63, step_size=63)
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(_make_returns(), config)
        # No duplicate dates in concatenated OOS returns
        assert not result.oos_returns.index.has_duplicates
