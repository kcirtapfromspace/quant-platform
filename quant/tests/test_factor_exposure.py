"""Tests for factor exposure analysis (QUA-83)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant.backtest.factor_exposure import (
    FactorBeta,
    FactorExposureAnalyzer,
    FactorExposureConfig,
    FactorExposureResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOLS = ["AAPL", "GOOG", "MSFT", "AMZN", "META"]


def _make_returns(n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0005, 0.01, size=n_days)
    idio = rng.normal(0, 0.015, size=(n_days, len(SYMBOLS)))
    betas = np.array([1.2, 0.8, 1.0, 1.4, 0.6])
    data = factor[:, None] * betas[None, :] + idio
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    return pd.DataFrame(data, index=dates, columns=SYMBOLS)


def _make_strategy_returns(
    asset_returns: pd.DataFrame,
    market_beta: float = 0.8,
    alpha: float = 0.0002,
    seed: int = 99,
) -> pd.Series:
    """Strategy returns = alpha + beta * market + noise."""
    rng = np.random.default_rng(seed)
    market = asset_returns.mean(axis=1)
    noise = rng.normal(0, 0.005, len(market))
    rets = alpha + market_beta * market.values + noise
    return pd.Series(rets, index=asset_returns.index, name="strategy")


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        assets = _make_returns()
        strategy = _make_strategy_returns(assets)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        assert isinstance(result, FactorExposureResult)

    def test_n_days_tracked(self):
        assets = _make_returns()
        strategy = _make_strategy_returns(assets)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        assert result.n_days == len(assets)

    def test_factor_betas_populated(self):
        assets = _make_returns()
        strategy = _make_strategy_returns(assets)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        assert len(result.factor_betas) > 0
        for fb in result.factor_betas:
            assert isinstance(fb, FactorBeta)

    def test_r_squared_bounded(self):
        assets = _make_returns()
        strategy = _make_strategy_returns(assets)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        assert 0.0 <= result.r_squared <= 1.0


# ---------------------------------------------------------------------------
# Factor detection
# ---------------------------------------------------------------------------


class TestFactorDetection:
    def test_market_beta_detected(self):
        """Strategy with known market beta should have detectable beta."""
        assets = _make_returns()
        strategy = _make_strategy_returns(assets, market_beta=1.0)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        market_fb = next(fb for fb in result.factor_betas if fb.factor == "market")
        # Market beta should be significantly positive
        assert market_fb.beta > 0.5
        assert abs(market_fb.t_stat) > 2.0

    def test_zero_beta_strategy(self):
        """Strategy with zero beta should have near-zero market exposure."""
        assets = _make_returns()
        strategy = _make_strategy_returns(assets, market_beta=0.0, alpha=0.0)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        market_fb = next(fb for fb in result.factor_betas if fb.factor == "market")
        assert abs(market_fb.beta) < 0.3

    def test_high_beta_strategy(self):
        """Strategy with beta > 1 should be detected."""
        assets = _make_returns()
        strategy = _make_strategy_returns(assets, market_beta=1.5)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        market_fb = next(fb for fb in result.factor_betas if fb.factor == "market")
        assert market_fb.beta > 1.0

    def test_all_factors_present(self):
        """Should construct market, momentum, volatility, size factors."""
        assets = _make_returns()
        strategy = _make_strategy_returns(assets)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        factor_names = {fb.factor for fb in result.factor_betas}
        assert "market" in factor_names
        assert "momentum" in factor_names
        assert "volatility" in factor_names
        assert "size" in factor_names


# ---------------------------------------------------------------------------
# Alpha estimation
# ---------------------------------------------------------------------------


class TestAlpha:
    def test_positive_alpha_detected(self):
        """Strategy with positive daily alpha should show positive annualized alpha."""
        assets = _make_returns()
        strategy = _make_strategy_returns(assets, market_beta=0.5, alpha=0.001)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        assert result.alpha > 0

    def test_alpha_is_annualized(self):
        """Alpha should be annualized (daily * 252)."""
        assets = _make_returns()
        strategy = _make_strategy_returns(assets, market_beta=0.5, alpha=0.001)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        # ~0.001 * 252 = ~25% annualized (will be approximate due to noise)
        assert result.alpha > 0.10  # at least 10% annualized

    def test_alpha_finite(self):
        assets = _make_returns()
        strategy = _make_strategy_returns(assets)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        assert math.isfinite(result.alpha)


# ---------------------------------------------------------------------------
# Rolling betas
# ---------------------------------------------------------------------------


class TestRollingBetas:
    def test_rolling_betas_shape(self):
        assets = _make_returns()
        strategy = _make_strategy_returns(assets)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        assert result.rolling_betas.shape[0] == len(assets)
        assert result.rolling_betas.shape[1] >= 1

    def test_rolling_betas_have_nans_early(self):
        """Before the window fills, rolling betas should be NaN."""
        assets = _make_returns()
        strategy = _make_strategy_returns(assets)
        config = FactorExposureConfig(rolling_window=63, min_periods=30)
        analyzer = FactorExposureAnalyzer(config)
        result = analyzer.analyze(strategy, assets)
        # First few rows should be NaN
        assert result.rolling_betas.iloc[0].isna().all()

    def test_rolling_betas_non_nan_later(self):
        """After the window fills, rolling betas should be finite."""
        assets = _make_returns()
        strategy = _make_strategy_returns(assets)
        config = FactorExposureConfig(rolling_window=63, min_periods=30)
        analyzer = FactorExposureAnalyzer(config)
        result = analyzer.analyze(strategy, assets)
        # Last row should have valid betas
        last = result.rolling_betas.iloc[-1]
        for val in last:
            assert math.isfinite(val)


# ---------------------------------------------------------------------------
# Residual returns
# ---------------------------------------------------------------------------


class TestResiduals:
    def test_residual_length(self):
        assets = _make_returns()
        strategy = _make_strategy_returns(assets)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        assert len(result.residual_returns) == len(assets)

    def test_residual_mean_near_zero(self):
        """OLS residuals should have near-zero mean."""
        assets = _make_returns()
        strategy = _make_strategy_returns(assets)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        assert abs(result.residual_returns.mean()) < 0.001

    def test_residual_smaller_than_original(self):
        """Residual variance should be smaller than original return variance."""
        assets = _make_returns()
        strategy = _make_strategy_returns(assets, market_beta=1.0)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        assert result.residual_returns.std() < strategy.std()


# ---------------------------------------------------------------------------
# Factor returns
# ---------------------------------------------------------------------------


class TestFactorReturns:
    def test_factor_returns_shape(self):
        assets = _make_returns()
        strategy = _make_strategy_returns(assets)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        assert result.factor_returns.shape[0] == len(assets)
        assert "market" in result.factor_returns.columns

    def test_market_factor_reasonable(self):
        """Market factor should have mean and vol in reasonable range."""
        assets = _make_returns()
        strategy = _make_strategy_returns(assets)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        mkt = result.factor_returns["market"]
        assert abs(mkt.mean()) < 0.01  # daily mean
        assert 0.001 < mkt.std() < 0.05  # daily vol


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_too_few_days_raises(self):
        assets = _make_returns(n_days=5)
        strategy = _make_strategy_returns(assets)
        analyzer = FactorExposureAnalyzer()
        with pytest.raises(ValueError, match="at least"):
            analyzer.analyze(strategy, assets)

    def test_few_assets_still_works(self):
        """Should work with as few as 3 assets."""
        rng = np.random.default_rng(42)
        n_days = 100
        dates = pd.bdate_range("2023-01-01", periods=n_days)
        assets = pd.DataFrame(
            rng.normal(0.0005, 0.01, (n_days, 3)),
            index=dates,
            columns=["A", "B", "C"],
        )
        strategy = assets.mean(axis=1) + rng.normal(0, 0.005, n_days)
        strategy = pd.Series(strategy, index=dates)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        assert isinstance(result, FactorExposureResult)

    def test_custom_config(self):
        assets = _make_returns()
        strategy = _make_strategy_returns(assets)
        config = FactorExposureConfig(
            rolling_window=42,
            min_periods=20,
            momentum_lookback=10,
            vol_lookback=10,
        )
        analyzer = FactorExposureAnalyzer(config)
        result = analyzer.analyze(strategy, assets)
        assert isinstance(result, FactorExposureResult)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        assets = _make_returns()
        strategy = _make_strategy_returns(assets)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        summary = result.summary()
        assert "Factor Exposure" in summary
        assert "R-squared" in summary
        assert "Alpha" in summary
        assert "market" in summary

    def test_summary_includes_rolling(self):
        assets = _make_returns()
        strategy = _make_strategy_returns(assets)
        analyzer = FactorExposureAnalyzer()
        result = analyzer.analyze(strategy, assets)
        summary = result.summary()
        assert "Rolling Beta" in summary
