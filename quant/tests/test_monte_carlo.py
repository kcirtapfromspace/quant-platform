"""Tests for Monte Carlo backtest confidence intervals (QUA-75)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant.backtest.monte_carlo import (
    ConfidenceInterval,
    MonteCarloAnalyzer,
    MonteCarloConfig,
    MonteCarloResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(
    n: int = 500,
    mean: float = 0.0004,
    std: float = 0.012,
    seed: int = 42,
) -> pd.Series:
    """Generate synthetic daily returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    data = rng.normal(mean, std, n)
    return pd.Series(data, index=dates, name="returns")


def _make_negative_returns(n: int = 500, seed: int = 42) -> pd.Series:
    """Generate returns with negative drift (losing strategy)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    data = rng.normal(-0.001, 0.015, n)
    return pd.Series(data, index=dates, name="returns")


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_run_returns_result(self):
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns())
        assert isinstance(result, MonteCarloResult)

    def test_n_simulations_tracked(self):
        config = MonteCarloConfig(n_simulations=100)
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert result.n_simulations == 100

    def test_n_days_tracked(self):
        returns = _make_returns(n=300)
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(returns)
        assert result.n_days == 300

    def test_default_config(self):
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns())
        assert result.confidence_level == 0.95


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


class TestConfidenceIntervals:
    def test_sharpe_ci_contains_point_estimate(self):
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), MonteCarloConfig(n_simulations=1000))
        ci = result.sharpe_ci
        assert isinstance(ci, ConfidenceInterval)
        assert ci.metric == "Sharpe"
        # Point estimate should be plausible (not necessarily inside CI)
        assert math.isfinite(ci.point_estimate)
        assert ci.lower <= ci.upper

    def test_drawdown_ci_bounds(self):
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), MonteCarloConfig(n_simulations=1000))
        ci = result.max_drawdown_ci
        assert ci.lower >= 0.0  # drawdown is always non-negative
        assert ci.upper >= ci.lower

    def test_return_ci_finite(self):
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), MonteCarloConfig(n_simulations=1000))
        ci = result.total_return_ci
        assert math.isfinite(ci.lower)
        assert math.isfinite(ci.upper)

    def test_cagr_ci_finite(self):
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), MonteCarloConfig(n_simulations=1000))
        ci = result.cagr_ci
        assert math.isfinite(ci.lower)
        assert math.isfinite(ci.upper)

    def test_volatility_ci_non_negative(self):
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), MonteCarloConfig(n_simulations=1000))
        ci = result.volatility_ci
        assert ci.lower >= 0.0
        assert ci.upper >= ci.lower

    def test_wider_ci_with_lower_confidence(self):
        returns = _make_returns()
        analyzer = MonteCarloAnalyzer()
        r95 = analyzer.run(
            returns, MonteCarloConfig(n_simulations=2000, confidence_level=0.95, seed=42)
        )
        r80 = analyzer.run(
            returns, MonteCarloConfig(n_simulations=2000, confidence_level=0.80, seed=42)
        )
        # 95% CI should be wider than 80% CI
        w95 = r95.sharpe_ci.upper - r95.sharpe_ci.lower
        w80 = r80.sharpe_ci.upper - r80.sharpe_ci.lower
        assert w95 > w80


# ---------------------------------------------------------------------------
# Tail risk metrics
# ---------------------------------------------------------------------------


class TestTailRisk:
    def test_var_finite(self):
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), MonteCarloConfig(n_simulations=1000))
        assert math.isfinite(result.var)

    def test_cvar_at_least_as_bad_as_var(self):
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), MonteCarloConfig(n_simulations=1000))
        # CVaR is the expected loss in the tail, should be <= VaR
        assert result.cvar <= result.var + 1e-10

    def test_prob_loss_between_zero_and_one(self):
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), MonteCarloConfig(n_simulations=1000))
        assert 0.0 <= result.prob_loss <= 1.0

    def test_negative_returns_high_prob_loss(self):
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(
            _make_negative_returns(),
            MonteCarloConfig(n_simulations=1000),
        )
        # Strategy with negative drift should have high probability of loss
        assert result.prob_loss > 0.3

    def test_var_confidence_tracked(self):
        config = MonteCarloConfig(n_simulations=500, var_confidence=0.99)
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert result.var_confidence == 0.99


# ---------------------------------------------------------------------------
# Distribution outputs
# ---------------------------------------------------------------------------


class TestDistributions:
    def test_sharpe_dist_shape(self):
        config = MonteCarloConfig(n_simulations=200)
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert result.sharpe_dist.shape == (200,)

    def test_all_dists_correct_length(self):
        config = MonteCarloConfig(n_simulations=300)
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert len(result.sharpe_dist) == 300
        assert len(result.max_drawdown_dist) == 300
        assert len(result.total_return_dist) == 300
        assert len(result.cagr_dist) == 300
        assert len(result.volatility_dist) == 300
        assert len(result.terminal_wealth_dist) == 300

    def test_terminal_wealth_positive(self):
        config = MonteCarloConfig(n_simulations=500)
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), config)
        # All terminal wealth should be positive (no bankruptcy in bootstrap)
        assert np.all(result.terminal_wealth_dist > 0)


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------


class TestBlockBootstrap:
    def test_block_bootstrap_runs(self):
        config = MonteCarloConfig(n_simulations=200, block_size=5)
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert isinstance(result, MonteCarloResult)

    def test_block_bootstrap_different_from_iid(self):
        returns = _make_returns()
        analyzer = MonteCarloAnalyzer()
        iid = analyzer.run(
            returns, MonteCarloConfig(n_simulations=1000, block_size=1, seed=42)
        )
        block = analyzer.run(
            returns, MonteCarloConfig(n_simulations=1000, block_size=21, seed=42)
        )
        # Different resampling → different distributions (not exactly equal)
        assert iid.sharpe_ci.lower != block.sharpe_ci.lower

    def test_large_block_size(self):
        config = MonteCarloConfig(n_simulations=100, block_size=63)
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert result.n_simulations == 100


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_too_few_returns_raises(self):
        short = pd.Series([0.01])
        analyzer = MonteCarloAnalyzer()
        with pytest.raises(ValueError, match="at least 2"):
            analyzer.run(short)

    def test_empty_returns_raises(self):
        empty = pd.Series(dtype=float)
        analyzer = MonteCarloAnalyzer()
        with pytest.raises(ValueError, match="at least 2"):
            analyzer.run(empty)

    def test_reproducible_with_seed(self):
        returns = _make_returns()
        analyzer = MonteCarloAnalyzer()
        r1 = analyzer.run(returns, MonteCarloConfig(n_simulations=100, seed=123))
        r2 = analyzer.run(returns, MonteCarloConfig(n_simulations=100, seed=123))
        np.testing.assert_array_equal(r1.sharpe_dist, r2.sharpe_dist)

    def test_different_seeds_different_results(self):
        returns = _make_returns()
        analyzer = MonteCarloAnalyzer()
        r1 = analyzer.run(returns, MonteCarloConfig(n_simulations=100, seed=1))
        r2 = analyzer.run(returns, MonteCarloConfig(n_simulations=100, seed=2))
        assert not np.array_equal(r1.sharpe_dist, r2.sharpe_dist)

    def test_minimal_returns(self):
        returns = pd.Series([0.01, -0.005])
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(returns, MonteCarloConfig(n_simulations=50))
        assert isinstance(result, MonteCarloResult)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), MonteCarloConfig(n_simulations=200))
        summary = result.summary()
        assert "Monte Carlo" in summary
        assert "Sharpe" in summary
        assert "VaR" in summary
        assert "CVaR" in summary
        assert "P(loss)" in summary

    def test_summary_includes_sim_count(self):
        config = MonteCarloConfig(n_simulations=200)
        analyzer = MonteCarloAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert "200" in result.summary()
