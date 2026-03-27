"""Tests for multi-asset portfolio backtesting engine (QUA-33)."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quant.backtest.portfolio_backtest import (
    PortfolioBacktestConfig,
    PortfolioBacktestEngine,
    PortfolioBacktestReport,
)
from quant.portfolio.alpha import CombinationMethod
from quant.portfolio.engine import PortfolioConfig
from quant.portfolio.optimizers import OptimizationMethod
from quant.signals.base import BaseSignal, SignalOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(
    n: int = 300,
    n_assets: int = 5,
    seed: int = 42,
    mean: float = 0.0004,
    std: float = 0.012,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Generate synthetic daily returns for multiple assets."""
    rng = np.random.default_rng(seed)
    if symbols is None:
        symbols = [f"SYM_{i}" for i in range(n_assets)]
    dates = pd.bdate_range("2022-01-01", periods=n)
    data = rng.normal(mean, std, (n, len(symbols)))
    return pd.DataFrame(data, index=dates, columns=symbols)


class _FixedSignal(BaseSignal):
    """Signal that returns a fixed score for testing."""

    def __init__(self, score: float = 0.5, confidence: float = 0.8) -> None:
        self._score = score
        self._confidence = confidence

    @property
    def name(self) -> str:
        return "fixed"

    @property
    def required_features(self) -> list[str]:
        return ["returns"]

    def compute(
        self,
        symbol: str,
        features: dict[str, pd.Series],
        timestamp: datetime,
    ) -> SignalOutput:
        return SignalOutput(
            symbol=symbol,
            timestamp=timestamp,
            score=self._score,
            confidence=self._confidence,
            target_position=self._score * self._confidence,
        )


class _SymbolBiasSignal(BaseSignal):
    """Signal that gives different scores per symbol based on index position."""

    def __init__(self, symbols: list[str]) -> None:
        self._symbols = symbols

    @property
    def name(self) -> str:
        return "symbol_bias"

    @property
    def required_features(self) -> list[str]:
        return ["returns"]

    def compute(
        self,
        symbol: str,
        features: dict[str, pd.Series],
        timestamp: datetime,
    ) -> SignalOutput:
        idx = self._symbols.index(symbol) if symbol in self._symbols else 0
        n = len(self._symbols)
        # First symbol gets +1, last gets -1, linear interpolation
        score = 1.0 - 2.0 * idx / max(n - 1, 1)
        return SignalOutput(
            symbol=symbol,
            timestamp=timestamp,
            score=max(-1.0, min(1.0, score)),
            confidence=0.8,
            target_position=max(-1.0, min(1.0, score * 0.8)),
        )


class _FailingSignal(BaseSignal):
    """Signal that always raises."""

    @property
    def name(self) -> str:
        return "failing"

    @property
    def required_features(self) -> list[str]:
        return ["returns"]

    def compute(self, symbol, features, timestamp):
        raise RuntimeError("signal failure")


# ---------------------------------------------------------------------------
# Core engine tests
# ---------------------------------------------------------------------------


class TestPortfolioBacktestEngine:
    def test_basic_run_produces_valid_report(self):
        """Engine should produce a complete report with sensible metrics."""
        returns = _make_returns(n=200, n_assets=4)
        signal = _FixedSignal(score=0.3, confidence=0.7)
        config = PortfolioBacktestConfig(
            rebalance_frequency=21,
            commission_bps=10.0,
            initial_capital=1_000_000.0,
            min_history=60,
            portfolio_config=PortfolioConfig(
                optimization_method=OptimizationMethod.RISK_PARITY,
                rebalance_threshold=0.0,  # always rebalance
            ),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [signal], config)

        assert isinstance(report, PortfolioBacktestReport)
        assert report.n_trading_days == 200
        assert report.initial_capital == 1_000_000.0
        assert report.final_value > 0
        assert report.n_rebalances > 0
        assert -1.0 < report.total_return < 10.0
        assert 0.0 <= report.max_drawdown <= 1.0
        assert report.total_costs >= 0

    def test_no_trades_before_min_history(self):
        """No rebalances should occur before min_history days."""
        returns = _make_returns(n=200, n_assets=3)
        config = PortfolioBacktestConfig(
            min_history=100,
            rebalance_frequency=10,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [_FixedSignal()], config)

        for snap in report.rebalances:
            # Rebalance dates should be at or after day 100
            snap_idx = returns.index.get_loc(pd.Timestamp(snap.date))
            assert snap_idx >= config.min_history

    def test_portfolio_value_starts_at_initial_capital(self):
        """Equity curve should begin at the configured initial capital."""
        returns = _make_returns(n=100, n_assets=3)
        config = PortfolioBacktestConfig(
            initial_capital=500_000.0,
            min_history=60,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [_FixedSignal()], config)

        assert report.equity_curve.iloc[0] == 500_000.0

    def test_equity_curve_length_matches_data(self):
        """Equity curve should have one entry per trading day."""
        n = 150
        returns = _make_returns(n=n, n_assets=3)
        engine = PortfolioBacktestEngine()
        report = engine.run(
            returns,
            [_FixedSignal()],
            PortfolioBacktestConfig(
                min_history=60,
                portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
            ),
        )

        assert len(report.equity_curve) == n
        assert len(report.returns_series) == n
        assert len(report.weights_history) == n

    def test_weights_history_columns_match_universe(self):
        """Weights DataFrame columns should include the traded symbols."""
        symbols = ["AAPL", "GOOG", "MSFT"]
        returns = _make_returns(n=200, symbols=symbols)
        engine = PortfolioBacktestEngine()
        report = engine.run(
            returns,
            [_FixedSignal()],
            PortfolioBacktestConfig(
                min_history=60,
                portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
            ),
        )

        # After min_history, some symbols should have non-zero weights
        post_rebalance = report.weights_history.iloc[70:]
        assert post_rebalance.abs().sum().sum() > 0

    def test_default_config_works(self):
        """Engine should run with default PortfolioBacktestConfig."""
        returns = _make_returns(n=200, n_assets=3)
        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [_FixedSignal()])

        assert isinstance(report, PortfolioBacktestReport)


# ---------------------------------------------------------------------------
# Transaction cost tests
# ---------------------------------------------------------------------------


class TestTransactionCosts:
    def test_costs_are_positive(self):
        """Transaction costs should be non-negative."""
        returns = _make_returns(n=200, n_assets=3)
        config = PortfolioBacktestConfig(
            commission_bps=10.0,
            min_history=60,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [_FixedSignal()], config)

        assert report.total_costs > 0
        for snap in report.rebalances:
            assert snap.transaction_costs >= 0

    def test_zero_commission_means_zero_costs(self):
        """With commission_bps=0, total costs should be zero."""
        returns = _make_returns(n=200, n_assets=3)
        config = PortfolioBacktestConfig(
            commission_bps=0.0,
            min_history=60,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [_FixedSignal()], config)

        assert report.total_costs == 0.0

    def test_higher_commission_reduces_final_value(self):
        """Higher commissions should produce a lower final portfolio value."""
        returns = _make_returns(n=200, n_assets=3, seed=42)
        signals = [_FixedSignal()]

        low_cost = PortfolioBacktestConfig(
            commission_bps=1.0,
            min_history=60,
            rebalance_frequency=21,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )
        high_cost = PortfolioBacktestConfig(
            commission_bps=100.0,
            min_history=60,
            rebalance_frequency=21,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        low_report = engine.run(returns, signals, low_cost)
        high_report = engine.run(returns, signals, high_cost)

        assert low_report.final_value > high_report.final_value

    def test_turnover_recorded_in_snapshots(self):
        """Each rebalance snapshot should have turnover > 0."""
        returns = _make_returns(n=200, n_assets=3)
        config = PortfolioBacktestConfig(
            min_history=60,
            rebalance_frequency=21,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [_FixedSignal()], config)

        # At least the first rebalance should have turnover (from cash to invested)
        assert report.rebalances[0].turnover > 0


# ---------------------------------------------------------------------------
# Weight drift tests
# ---------------------------------------------------------------------------


class TestWeightDrift:
    def test_weights_change_between_rebalances(self):
        """Weights should drift between rebalance points due to returns."""
        returns = _make_returns(n=200, n_assets=3, std=0.03)  # high vol for drift
        config = PortfolioBacktestConfig(
            min_history=60,
            rebalance_frequency=50,  # infrequent to observe drift
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [_FixedSignal()], config)

        if len(report.rebalances) >= 2:
            # Weights at first rebalance
            first_date = pd.Timestamp(report.rebalances[0].date)
            first_idx = returns.index.get_loc(first_date)
            # Weights a few days after first rebalance should differ (drift)
            w_at_rebalance = report.weights_history.iloc[first_idx]
            w_after = report.weights_history.iloc[first_idx + 5]
            # Not exactly equal due to drift
            assert not np.allclose(
                w_at_rebalance.values, w_after.values, atol=1e-6
            )

    def test_weights_non_negative_and_bounded(self):
        """Drifted weights should stay in [0, 1] and sum to at most ~1."""
        returns = _make_returns(n=200, n_assets=4)
        config = PortfolioBacktestConfig(
            min_history=60,
            rebalance_frequency=21,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [_FixedSignal()], config)

        # After first rebalance, weights should be non-negative and sum ≤ ~1
        # (remainder is implicitly cash — optimizer may not fully invest)
        post = report.weights_history.iloc[config.min_history + 5 :]
        row_sums = post.sum(axis=1)
        invested = row_sums[row_sums > 0.01]
        if not invested.empty:
            # Weight sums should not exceed 1.0 by much (small float drift ok)
            assert all(s < 1.05 for s in invested)
            # Should be investing at least some capital
            assert invested.mean() > 0.1


# ---------------------------------------------------------------------------
# Benchmark comparison tests
# ---------------------------------------------------------------------------


class TestBenchmarkComparison:
    def test_benchmark_metrics_computed(self):
        """When benchmark is provided, report should contain active metrics."""
        returns = _make_returns(n=200, n_assets=3, symbols=["A", "B", "C"])
        # Add benchmark column
        rng = np.random.default_rng(99)
        returns["BENCH"] = rng.normal(0.0003, 0.01, len(returns))

        config = PortfolioBacktestConfig(
            benchmark="BENCH",
            min_history=60,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [_FixedSignal()], config)

        assert report.benchmark_return != 0.0 or report.tracking_error > 0
        # Benchmark column should not be in the traded universe
        assert "BENCH" not in report.weights_history.columns

    def test_no_benchmark_gives_zeros(self):
        """Without benchmark, active metrics should be zero."""
        returns = _make_returns(n=200, n_assets=3)
        config = PortfolioBacktestConfig(
            min_history=60,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [_FixedSignal()], config)

        assert report.benchmark_return == 0.0
        assert report.tracking_error == 0.0
        assert report.information_ratio == 0.0


# ---------------------------------------------------------------------------
# Signal integration tests
# ---------------------------------------------------------------------------


class TestSignalIntegration:
    def test_multiple_signals(self):
        """Engine should work with multiple signals combined."""
        returns = _make_returns(n=200, n_assets=3)
        signals = [
            _FixedSignal(score=0.5, confidence=0.9),
            _FixedSignal(score=-0.2, confidence=0.6),
        ]
        config = PortfolioBacktestConfig(
            min_history=60,
            combination_method=CombinationMethod.EQUAL_WEIGHT,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, signals, config)

        assert report.n_rebalances > 0

    def test_failing_signal_does_not_crash(self):
        """A signal that raises should be skipped, not crash the engine."""
        returns = _make_returns(n=200, n_assets=3)
        signals = [_FixedSignal(), _FailingSignal()]
        config = PortfolioBacktestConfig(
            min_history=60,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        # Should not raise
        report = engine.run(returns, signals, config)
        assert isinstance(report, PortfolioBacktestReport)

    def test_custom_feature_provider(self):
        """Engine should use the feature_provider when supplied."""
        returns = _make_returns(n=200, n_assets=3)
        calls: list[str] = []

        def provider(symbol: str, rets: pd.Series) -> dict[str, pd.Series]:
            calls.append(symbol)
            return {"returns": rets}

        config = PortfolioBacktestConfig(
            min_history=60,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        engine.run(returns, [_FixedSignal()], config, feature_provider=provider)

        assert len(calls) > 0  # provider was invoked

    def test_symbol_bias_produces_differentiated_weights(self):
        """Signals with per-symbol scores should produce non-uniform weights."""
        symbols = ["A", "B", "C", "D"]
        returns = _make_returns(n=200, n_assets=4, symbols=symbols)
        signal = _SymbolBiasSignal(symbols)
        config = PortfolioBacktestConfig(
            min_history=60,
            portfolio_config=PortfolioConfig(
                optimization_method=OptimizationMethod.MEAN_VARIANCE,
                rebalance_threshold=0.0,
            ),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [signal], config)

        # After rebalance, weights should not all be equal
        if report.rebalances:
            w = report.rebalances[0].weights
            values = [v for v in w.values() if abs(v) > 1e-9]
            if len(values) > 1:
                assert max(values) != min(values)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_insufficient_data_raises(self):
        """Too few data points should raise ValueError."""
        returns = _make_returns(n=20, n_assets=3)
        config = PortfolioBacktestConfig(min_history=60)

        engine = PortfolioBacktestEngine()
        with pytest.raises(ValueError, match="Insufficient data"):
            engine.run(returns, [_FixedSignal()], config)

    def test_empty_returns_raises(self):
        """Empty returns DataFrame should raise ValueError."""
        returns = pd.DataFrame()
        engine = PortfolioBacktestEngine()
        with pytest.raises(ValueError, match="empty"):
            engine.run(returns, [_FixedSignal()])

    def test_single_asset(self):
        """Engine should work with a single asset."""
        returns = _make_returns(n=200, n_assets=1, symbols=["ONLY"])
        config = PortfolioBacktestConfig(
            min_history=60,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [_FixedSignal()], config)

        assert report.n_rebalances > 0
        # Single asset should get weight ~1.0
        if report.rebalances:
            w = report.rebalances[0].weights
            assert abs(w.get("ONLY", 0.0)) > 0.5

    def test_nan_returns_treated_as_zero(self):
        """NaN values in returns should be handled gracefully."""
        returns = _make_returns(n=200, n_assets=3)
        # Inject some NaN values
        returns.iloc[100:110, 0] = np.nan

        config = PortfolioBacktestConfig(
            min_history=60,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [_FixedSignal()], config)

        assert np.isfinite(report.final_value)
        assert np.isfinite(report.total_return)

    def test_all_signals_fail_still_produces_report(self):
        """If all signals fail, the engine should still return a report."""
        returns = _make_returns(n=200, n_assets=3)
        config = PortfolioBacktestConfig(
            min_history=60,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [_FailingSignal()], config)

        # No rebalances should occur (no valid signals)
        assert report.n_rebalances == 0
        # Portfolio value unchanged (stayed in cash)
        assert report.final_value == report.initial_capital

    def test_high_rebalance_frequency(self):
        """Daily rebalancing should produce many snapshots."""
        returns = _make_returns(n=200, n_assets=3)
        config = PortfolioBacktestConfig(
            min_history=60,
            rebalance_frequency=1,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [_FixedSignal()], config)

        # Should have approximately (200 - 60) rebalances
        assert report.n_rebalances >= 100


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------


class TestPortfolioBacktestReport:
    def test_summary_string(self):
        """Report summary should contain key metrics."""
        returns = _make_returns(n=200, n_assets=3)
        engine = PortfolioBacktestEngine()
        report = engine.run(
            returns,
            [_FixedSignal()],
            PortfolioBacktestConfig(
                min_history=60,
                portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
            ),
        )

        summary = report.summary()
        assert "Portfolio Backtest" in summary
        assert "Sharpe" in summary
        assert "Max drawdown" in summary
        assert "CAGR" in summary
        assert "Rebalances" in summary

    def test_summary_with_benchmark(self):
        """Benchmark section should appear when benchmark is configured."""
        returns = _make_returns(n=200, n_assets=3, symbols=["A", "B", "C"])
        rng = np.random.default_rng(99)
        returns["BM"] = rng.normal(0.0003, 0.01, len(returns))

        engine = PortfolioBacktestEngine()
        report = engine.run(
            returns,
            [_FixedSignal()],
            PortfolioBacktestConfig(
                benchmark="BM",
                min_history=60,
                portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
            ),
        )

        summary = report.summary()
        assert "Benchmark return" in summary
        assert "Active return" in summary
        assert "Tracking error" in summary

    def test_report_dates(self):
        """Report start/end dates should match the data range."""
        returns = _make_returns(n=200, n_assets=3)
        engine = PortfolioBacktestEngine()
        report = engine.run(
            returns,
            [_FixedSignal()],
            PortfolioBacktestConfig(
                min_history=60,
                portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
            ),
        )

        assert report.start_date == returns.index[0].date()
        assert report.end_date == returns.index[-1].date()


# ---------------------------------------------------------------------------
# Rebalance frequency tests
# ---------------------------------------------------------------------------


class TestRebalanceFrequency:
    def test_less_frequent_means_fewer_rebalances(self):
        """Increasing rebalance_frequency should reduce the number of rebalances."""
        returns = _make_returns(n=300, n_assets=3)
        signals = [_FixedSignal()]
        pc = PortfolioConfig(rebalance_threshold=0.0)

        engine = PortfolioBacktestEngine()

        fast = engine.run(
            returns,
            signals,
            PortfolioBacktestConfig(
                rebalance_frequency=10, min_history=60, portfolio_config=pc
            ),
        )
        slow = engine.run(
            returns,
            signals,
            PortfolioBacktestConfig(
                rebalance_frequency=60, min_history=60, portfolio_config=pc
            ),
        )

        assert fast.n_rebalances > slow.n_rebalances

    def test_less_frequent_means_lower_costs(self):
        """Fewer rebalances should produce lower transaction costs."""
        returns = _make_returns(n=300, n_assets=3)
        signals = [_FixedSignal()]
        pc = PortfolioConfig(rebalance_threshold=0.0)

        engine = PortfolioBacktestEngine()

        fast = engine.run(
            returns,
            signals,
            PortfolioBacktestConfig(
                rebalance_frequency=5,
                commission_bps=10.0,
                min_history=60,
                portfolio_config=pc,
            ),
        )
        slow = engine.run(
            returns,
            signals,
            PortfolioBacktestConfig(
                rebalance_frequency=60,
                commission_bps=10.0,
                min_history=60,
                portfolio_config=pc,
            ),
        )

        assert fast.total_costs > slow.total_costs


# ---------------------------------------------------------------------------
# Factor signal integration test
# ---------------------------------------------------------------------------


class TestFactorSignalIntegration:
    def test_with_volatility_signal(self):
        """Engine should work with a real factor signal (VolatilitySignal)."""
        from quant.signals.factors import VolatilitySignal

        returns = _make_returns(n=200, n_assets=4)
        signal = VolatilitySignal(period=20)
        config = PortfolioBacktestConfig(
            min_history=60,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, [signal], config)

        assert report.n_rebalances > 0
        assert report.final_value > 0

    def test_with_multiple_factor_signals(self):
        """Engine should work with multiple factor signals combined."""
        from quant.signals.factors import (
            BreakoutSignal,
            ReturnQualitySignal,
            VolatilitySignal,
        )

        returns = _make_returns(n=250, n_assets=5)
        signals = [
            VolatilitySignal(period=20),
            ReturnQualitySignal(period=60),
            BreakoutSignal(period=20),
        ]
        config = PortfolioBacktestConfig(
            min_history=80,
            combination_method=CombinationMethod.EQUAL_WEIGHT,
            portfolio_config=PortfolioConfig(
                optimization_method=OptimizationMethod.RISK_PARITY,
                rebalance_threshold=0.0,
            ),
        )

        engine = PortfolioBacktestEngine()
        report = engine.run(returns, signals, config)

        assert report.n_rebalances > 0
        assert np.isfinite(report.sharpe_ratio)
        assert np.isfinite(report.max_drawdown)
