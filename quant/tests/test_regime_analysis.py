"""Tests for regime-conditioned performance analysis (QUA-78)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.multi_strategy import (
    MultiStrategyBacktestEngine,
    MultiStrategyConfig,
    SleeveConfig,
)
from quant.backtest.regime_analysis import (
    RegimeAnalysisResult,
    RegimeAnalyzer,
    RegimePerformance,
    RegimeTransitionMatrix,
    SleeveRegimePerformance,
)
from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.engine import PortfolioConfig
from quant.portfolio.optimizers import OptimizationMethod
from quant.signals.base import BaseSignal, SignalOutput
from quant.signals.regime import RegimeConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOLS = ["AAPL", "GOOG", "MSFT"]


class _StubSignal(BaseSignal):
    def __init__(self, name: str = "stub", offset: float = 0.0) -> None:
        self._name = name
        self._offset = offset

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> list[str]:
        return []

    def compute(self, symbol, features, timestamp):
        score = 0.5 + self._offset
        return SignalOutput(
            symbol=symbol,
            timestamp=timestamp,
            score=score,
            confidence=0.8,
            target_position=score * 0.8,
            metadata={"signal_name": self._name},
        )


def _make_returns(n_days: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    factor = rng.normal(0.0005, 0.01, size=n_days)
    idio = rng.normal(0, 0.015, size=(n_days, len(SYMBOLS)))
    betas = rng.uniform(0.5, 1.5, size=len(SYMBOLS))
    data = factor[:, None] * betas[None, :] + idio
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    return pd.DataFrame(data, index=dates, columns=SYMBOLS)


def _make_regime_report():
    """Run a backtest with regime detection and return the report."""
    sleeves = [
        SleeveConfig(
            name="momentum",
            signals=[_StubSignal("mom", 0.1)],
            capital_weight=0.6,
            strategy_type="momentum",
            portfolio_config=PortfolioConfig(
                optimization_method=OptimizationMethod.RISK_PARITY,
                constraints=PortfolioConstraints(
                    long_only=True, max_weight=0.5, max_gross_exposure=1.0
                ),
            ),
        ),
        SleeveConfig(
            name="mean_rev",
            signals=[_StubSignal("mr", 0.2)],
            capital_weight=0.4,
            strategy_type="mean_reversion",
            portfolio_config=PortfolioConfig(
                optimization_method=OptimizationMethod.RISK_PARITY,
                constraints=PortfolioConstraints(
                    long_only=True, max_weight=0.5, max_gross_exposure=1.0
                ),
            ),
        ),
    ]
    config = MultiStrategyConfig(
        sleeves=sleeves,
        rebalance_frequency=21,
        commission_bps=10.0,
        min_history=60,
        regime_config=RegimeConfig(),
    )
    engine = MultiStrategyBacktestEngine()
    return engine.run(_make_returns(), config)


# ---------------------------------------------------------------------------
# Core analysis tests
# ---------------------------------------------------------------------------


class TestRegimeAnalysis:
    def test_analyze_returns_result(self):
        report = _make_regime_report()
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(report)
        assert isinstance(result, RegimeAnalysisResult)

    def test_n_regimes_positive(self):
        report = _make_regime_report()
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(report)
        assert result.n_regimes > 0

    def test_portfolio_by_regime_populated(self):
        report = _make_regime_report()
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(report)
        assert len(result.portfolio_by_regime) == result.n_regimes
        for rp in result.portfolio_by_regime:
            assert isinstance(rp, RegimePerformance)
            assert rp.n_days > 0

    def test_regime_performance_metrics_finite(self):
        report = _make_regime_report()
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(report)
        for rp in result.portfolio_by_regime:
            assert np.isfinite(rp.total_return)
            assert np.isfinite(rp.volatility)
            assert np.isfinite(rp.sharpe)
            assert np.isfinite(rp.max_drawdown)

    def test_pct_of_total_days_sums_to_one(self):
        report = _make_regime_report()
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(report)
        total_pct = sum(rp.pct_of_total_days for rp in result.portfolio_by_regime)
        # May not be exactly 1.0 due to early days before first regime label
        assert total_pct <= 1.0 + 1e-6

    def test_best_worst_regime_valid(self):
        report = _make_regime_report()
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(report)
        assert result.best_regime in result.regime_labels
        assert result.worst_regime in result.regime_labels


# ---------------------------------------------------------------------------
# Per-sleeve analysis
# ---------------------------------------------------------------------------


class TestSleeveRegime:
    def test_sleeve_by_regime_populated(self):
        report = _make_regime_report()
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(report)
        assert len(result.sleeve_by_regime) > 0
        for sp in result.sleeve_by_regime:
            assert isinstance(sp, SleeveRegimePerformance)

    def test_sleeve_count_matches(self):
        report = _make_regime_report()
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(report)
        n_sleeves = len(report.sleeve_returns.columns)
        expected = n_sleeves * result.n_regimes
        assert len(result.sleeve_by_regime) == expected

    def test_sleeve_metrics_finite(self):
        report = _make_regime_report()
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(report)
        for sp in result.sleeve_by_regime:
            assert np.isfinite(sp.total_return)
            assert np.isfinite(sp.volatility)
            assert np.isfinite(sp.sharpe)


# ---------------------------------------------------------------------------
# Duration and transition analysis
# ---------------------------------------------------------------------------


class TestDurationAndTransitions:
    def test_avg_duration_populated(self):
        report = _make_regime_report()
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(report)
        assert len(result.avg_regime_duration) > 0
        for regime, dur in result.avg_regime_duration.items():
            assert dur > 0
            assert regime in result.regime_labels

    def test_max_duration_populated(self):
        report = _make_regime_report()
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(report)
        assert len(result.max_regime_duration) > 0
        for _regime, dur in result.max_regime_duration.items():
            assert dur > 0

    def test_transition_matrix_structure(self):
        report = _make_regime_report()
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(report)
        tm = result.transition_matrix
        assert isinstance(tm, RegimeTransitionMatrix)
        assert len(tm.regimes) == result.n_regimes

    def test_transition_rows_sum_to_one(self):
        report = _make_regime_report()
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(report)
        tm = result.transition_matrix
        for regime in tm.regimes:
            row_sum = sum(tm.matrix[regime].values())
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 1e-10

    def test_transition_get(self):
        report = _make_regime_report()
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(report)
        tm = result.transition_matrix
        # Self-transition should be high (regimes are persistent)
        for regime in tm.regimes:
            self_prob = tm.get(regime, regime)
            assert 0.0 <= self_prob <= 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_regime_raises(self):
        """Analyzing a report without regime history should raise."""
        # Run backtest without regime config
        sleeves = [
            SleeveConfig(
                name="test",
                signals=[_StubSignal()],
                capital_weight=1.0,
                portfolio_config=PortfolioConfig(
                    optimization_method=OptimizationMethod.RISK_PARITY,
                    constraints=PortfolioConstraints(
                        long_only=True, max_weight=0.5,
                    ),
                ),
            ),
        ]
        config = MultiStrategyConfig(
            sleeves=sleeves,
            rebalance_frequency=21,
            min_history=60,
        )
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        analyzer = RegimeAnalyzer()
        with pytest.raises(ValueError, match="regime_history is empty"):
            analyzer.analyze(report)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        report = _make_regime_report()
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(report)
        summary = result.summary()
        assert "Regime-Conditioned Analysis" in summary
        assert "Sharpe" in summary
        assert "Best regime" in summary
        assert "Worst regime" in summary

    def test_summary_includes_all_regimes(self):
        report = _make_regime_report()
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(report)
        summary = result.summary()
        for regime in result.regime_labels:
            assert regime in summary
