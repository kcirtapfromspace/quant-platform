"""Tests for walk-forward analysis engine (QUA-34)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardResult,
)
from quant.portfolio.engine import PortfolioConfig
from quant.signals.base import BaseSignal, SignalOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(
    n: int = 600,
    n_assets: int = 4,
    seed: int = 42,
    mean: float = 0.0004,
    std: float = 0.012,
) -> pd.DataFrame:
    """Generate synthetic daily returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    symbols = [f"SYM_{i}" for i in range(n_assets)]
    data = rng.normal(mean, std, (n, n_assets))
    return pd.DataFrame(data, index=dates, columns=symbols)


class _FixedSignal(BaseSignal):
    """Deterministic signal for testing."""

    def __init__(self, score: float = 0.5) -> None:
        self._score = score

    @property
    def name(self) -> str:
        return "fixed"

    @property
    def required_features(self) -> list[str]:
        return ["returns"]

    def compute(self, symbol, features, timestamp):
        return SignalOutput(
            symbol=symbol,
            timestamp=timestamp,
            score=self._score,
            confidence=0.8,
            target_position=self._score * 0.8,
        )


# ---------------------------------------------------------------------------
# Fold generation tests
# ---------------------------------------------------------------------------


class TestFoldGeneration:
    def test_rolling_folds_correct_count(self):
        """Rolling mode should produce the expected number of folds."""
        config = WalkForwardConfig(
            is_window=100,
            oos_window=50,
            step_size=50,
        )
        # With 500 days: first fold starts at 0, IS=[0:100], OOS=[100:150]
        # Next fold: IS=[50:150], OOS=[150:200], etc.
        analyzer = WalkForwardAnalyzer()
        folds = analyzer._generate_folds(500, config)
        assert len(folds) > 0
        # Each fold should have non-overlapping OOS if step == oos_window
        for _i, (_is_s, is_e, oos_s, oos_e) in enumerate(folds):
            assert is_e == oos_s  # OOS starts right after IS
            assert oos_e - oos_s == 50  # OOS window size
            assert oos_e <= 500

    def test_expanding_folds_grow_is_window(self):
        """Expanding mode should have IS windows that grow."""
        config = WalkForwardConfig(
            is_window=100,
            oos_window=50,
            step_size=50,
            expanding=True,
        )
        analyzer = WalkForwardAnalyzer()
        folds = analyzer._generate_folds(500, config)
        assert len(folds) >= 2
        # First fold IS: [0, 100], second fold IS: [0, 150], etc.
        for i, (is_s, is_e, _oos_s, _oos_e) in enumerate(folds):
            assert is_s == 0  # anchored at start
            assert is_e == 100 + i * 50  # growing

    def test_no_folds_if_data_too_short(self):
        """Should return empty list if data is shorter than one fold."""
        config = WalkForwardConfig(is_window=100, oos_window=50)
        analyzer = WalkForwardAnalyzer()
        folds = analyzer._generate_folds(140, config)
        assert len(folds) == 0

    def test_oos_never_exceeds_data(self):
        """OOS end should never exceed available data."""
        config = WalkForwardConfig(
            is_window=100, oos_window=50, step_size=50
        )
        analyzer = WalkForwardAnalyzer()
        folds = analyzer._generate_folds(500, config)
        for _, _, _, oos_e in folds:
            assert oos_e <= 500


# ---------------------------------------------------------------------------
# Core analyzer tests
# ---------------------------------------------------------------------------


class TestWalkForwardAnalyzer:
    def test_basic_run_produces_result(self):
        """Analyzer should produce a valid WalkForwardResult."""
        returns = _make_returns(n=500, n_assets=3)
        config = WalkForwardConfig(
            is_window=150,
            oos_window=50,
            step_size=50,
            min_history=40,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        analyzer = WalkForwardAnalyzer()
        result = analyzer.run(returns, [_FixedSignal()], config)

        assert isinstance(result, WalkForwardResult)
        assert result.n_folds > 0
        assert len(result.folds) == result.n_folds

    def test_folds_have_valid_dates(self):
        """Each fold should have sensible date ranges."""
        returns = _make_returns(n=500, n_assets=3)
        config = WalkForwardConfig(
            is_window=150,
            oos_window=50,
            step_size=50,
            min_history=40,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        analyzer = WalkForwardAnalyzer()
        result = analyzer.run(returns, [_FixedSignal()], config)

        for fold in result.folds:
            assert fold.is_start < fold.is_end
            assert fold.oos_start < fold.oos_end
            assert fold.is_end <= fold.oos_start or fold.is_end == fold.oos_start

    def test_oos_returns_not_empty(self):
        """Aggregated OOS returns should be non-empty."""
        returns = _make_returns(n=500, n_assets=3)
        config = WalkForwardConfig(
            is_window=150,
            oos_window=50,
            step_size=50,
            min_history=40,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        analyzer = WalkForwardAnalyzer()
        result = analyzer.run(returns, [_FixedSignal()], config)

        assert not result.oos_returns.empty
        assert not result.oos_equity_curve.empty

    def test_oos_metrics_are_finite(self):
        """All aggregate OOS metrics should be finite numbers."""
        returns = _make_returns(n=500, n_assets=3)
        config = WalkForwardConfig(
            is_window=150,
            oos_window=50,
            step_size=50,
            min_history=40,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        analyzer = WalkForwardAnalyzer()
        result = analyzer.run(returns, [_FixedSignal()], config)

        assert np.isfinite(result.oos_total_return)
        assert np.isfinite(result.oos_sharpe)
        assert np.isfinite(result.oos_max_drawdown)
        assert np.isfinite(result.oos_volatility)

    def test_wfe_per_fold(self):
        """Each fold should have a WFE value."""
        returns = _make_returns(n=500, n_assets=3)
        config = WalkForwardConfig(
            is_window=150,
            oos_window=50,
            step_size=50,
            min_history=40,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        analyzer = WalkForwardAnalyzer()
        result = analyzer.run(returns, [_FixedSignal()], config)

        for fold in result.folds:
            assert np.isfinite(fold.wfe)
            assert -2.0 <= fold.wfe <= 2.0


# ---------------------------------------------------------------------------
# Walk-forward efficiency tests
# ---------------------------------------------------------------------------


class TestWalkForwardEfficiency:
    def test_mean_and_median_wfe_computed(self):
        """Result should include mean and median WFE."""
        returns = _make_returns(n=500, n_assets=3)
        config = WalkForwardConfig(
            is_window=150,
            oos_window=50,
            step_size=50,
            min_history=40,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        analyzer = WalkForwardAnalyzer()
        result = analyzer.run(returns, [_FixedSignal()], config)

        assert np.isfinite(result.mean_wfe)
        assert np.isfinite(result.median_wfe)

    def test_oos_vs_is_sharpe_ratio(self):
        """OOS/IS Sharpe ratio should be computed."""
        returns = _make_returns(n=500, n_assets=3)
        config = WalkForwardConfig(
            is_window=150,
            oos_window=50,
            step_size=50,
            min_history=40,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        analyzer = WalkForwardAnalyzer()
        result = analyzer.run(returns, [_FixedSignal()], config)

        assert np.isfinite(result.oos_vs_is_sharpe_ratio)


# ---------------------------------------------------------------------------
# Window mode tests
# ---------------------------------------------------------------------------


class TestWindowModes:
    def test_expanding_produces_results(self):
        """Expanding window mode should work."""
        returns = _make_returns(n=500, n_assets=3)
        config = WalkForwardConfig(
            is_window=150,
            oos_window=50,
            step_size=50,
            expanding=True,
            min_history=40,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        analyzer = WalkForwardAnalyzer()
        result = analyzer.run(returns, [_FixedSignal()], config)

        assert result.n_folds > 0

    def test_expanding_is_windows_grow(self):
        """In expanding mode, IS end date should advance each fold."""
        returns = _make_returns(n=500, n_assets=3)
        config = WalkForwardConfig(
            is_window=150,
            oos_window=50,
            step_size=50,
            expanding=True,
            min_history=40,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        analyzer = WalkForwardAnalyzer()
        result = analyzer.run(returns, [_FixedSignal()], config)

        if result.n_folds >= 2:
            for i in range(1, result.n_folds):
                assert result.folds[i].is_end > result.folds[i - 1].is_end

    def test_rolling_vs_expanding_different_fold_count(self):
        """Rolling and expanding may produce different fold counts."""
        returns = _make_returns(n=500, n_assets=3)
        base = {
            "is_window": 150,
            "oos_window": 50,
            "step_size": 50,
            "min_history": 40,
            "portfolio_config": PortfolioConfig(rebalance_threshold=0.0),
        }

        analyzer = WalkForwardAnalyzer()
        rolling = analyzer.run(
            returns,
            [_FixedSignal()],
            WalkForwardConfig(**base, expanding=False),
        )
        expanding = analyzer.run(
            returns,
            [_FixedSignal()],
            WalkForwardConfig(**base, expanding=True),
        )

        # Both should produce results
        assert rolling.n_folds > 0
        assert expanding.n_folds > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestWalkForwardEdgeCases:
    def test_insufficient_data_raises(self):
        """Too few days should raise ValueError."""
        returns = _make_returns(n=50, n_assets=3)
        config = WalkForwardConfig(is_window=100, oos_window=50)

        analyzer = WalkForwardAnalyzer()
        with pytest.raises(ValueError, match="Need at least"):
            analyzer.run(returns, [_FixedSignal()], config)

    def test_empty_returns_raises(self):
        """Empty DataFrame should raise ValueError."""
        analyzer = WalkForwardAnalyzer()
        with pytest.raises(ValueError, match="empty"):
            analyzer.run(
                pd.DataFrame(), [_FixedSignal()], WalkForwardConfig()
            )

    def test_single_fold(self):
        """Should work with exactly enough data for one fold."""
        # is=100 + oos=50 = 150 needed
        returns = _make_returns(n=155, n_assets=3)
        config = WalkForwardConfig(
            is_window=100,
            oos_window=50,
            step_size=50,
            min_history=30,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        analyzer = WalkForwardAnalyzer()
        result = analyzer.run(returns, [_FixedSignal()], config)

        assert result.n_folds == 1

    def test_single_asset(self):
        """Should work with a single asset."""
        returns = _make_returns(n=500, n_assets=1)
        config = WalkForwardConfig(
            is_window=150,
            oos_window=50,
            step_size=50,
            min_history=40,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        analyzer = WalkForwardAnalyzer()
        result = analyzer.run(returns, [_FixedSignal()], config)

        assert result.n_folds > 0


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------


class TestWalkForwardReport:
    def test_summary_string(self):
        """Summary should contain key sections."""
        returns = _make_returns(n=500, n_assets=3)
        config = WalkForwardConfig(
            is_window=150,
            oos_window=50,
            step_size=50,
            min_history=40,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        analyzer = WalkForwardAnalyzer()
        result = analyzer.run(returns, [_FixedSignal()], config)
        summary = result.summary()

        assert "Walk-Forward Analysis" in summary
        assert "OOS Sharpe" in summary
        assert "IS Mean Sharpe" in summary
        assert "Mean WFE" in summary
        assert "Assessment" in summary

    def test_summary_contains_assessment(self):
        """Summary should include overfitting risk assessment."""
        returns = _make_returns(n=500, n_assets=3)
        config = WalkForwardConfig(
            is_window=150,
            oos_window=50,
            step_size=50,
            min_history=40,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        analyzer = WalkForwardAnalyzer()
        result = analyzer.run(returns, [_FixedSignal()], config)
        summary = result.summary()

        assert "overfitting risk" in summary


# ---------------------------------------------------------------------------
# Integration with real factor signals
# ---------------------------------------------------------------------------


class TestFactorSignalWalkForward:
    def test_with_volatility_signal(self):
        """Walk-forward should work with VolatilitySignal."""
        from quant.signals.factors import VolatilitySignal

        returns = _make_returns(n=500, n_assets=4)
        config = WalkForwardConfig(
            is_window=200,
            oos_window=50,
            step_size=50,
            min_history=40,
            portfolio_config=PortfolioConfig(rebalance_threshold=0.0),
        )

        analyzer = WalkForwardAnalyzer()
        result = analyzer.run(returns, [VolatilitySignal(period=20)], config)

        assert result.n_folds > 0
        assert np.isfinite(result.oos_sharpe)
        assert np.isfinite(result.mean_wfe)
