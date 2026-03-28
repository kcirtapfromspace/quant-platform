"""Tests for strategy performance analytics (QUA-106)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.performance_analytics import (
    AnalyticsConfig,
    PerformanceAnalyzer,
    PerformanceResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_DAYS = 500


def _returns(seed: int = 42, drift: float = 0.0003) -> pd.Series:
    """Generate daily returns with optional positive drift."""
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, 0.01, N_DAYS)
    dates = pd.bdate_range("2023-01-01", periods=N_DAYS)
    return pd.Series(r, index=dates, name="returns")


def _losing_returns(seed: int = 42) -> pd.Series:
    """Generate returns with negative drift."""
    return _returns(seed=seed, drift=-0.0005)


def _flat_returns() -> pd.Series:
    """Returns of exactly zero every day."""
    dates = pd.bdate_range("2023-01-01", periods=N_DAYS)
    return pd.Series(np.zeros(N_DAYS), index=dates, name="returns")


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert isinstance(result, PerformanceResult)

    def test_n_days(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.n_days == N_DAYS

    def test_config_accessible(self):
        cfg = AnalyticsConfig(risk_free_rate=0.02)
        analyzer = PerformanceAnalyzer(cfg)
        assert analyzer.config.risk_free_rate == 0.02

    def test_too_few_observations_raises(self):
        r = pd.Series([0.01])
        with pytest.raises(ValueError, match="at least 2"):
            PerformanceAnalyzer().analyze(r)


# ---------------------------------------------------------------------------
# Return metrics
# ---------------------------------------------------------------------------


class TestReturnMetrics:
    def test_positive_total_return(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.total_return > 0

    def test_positive_cagr(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.cagr > 0

    def test_negative_total_return_for_losers(self):
        result = PerformanceAnalyzer().analyze(_losing_returns())
        assert result.total_return < 0

    def test_best_day_positive(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.best_day > 0

    def test_worst_day_negative(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.worst_day < 0

    def test_best_above_worst(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.best_day > result.worst_day


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------


class TestRiskMetrics:
    def test_vol_positive(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.annualised_vol > 0

    def test_downside_std_positive(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.downside_std > 0

    def test_vol_annualised(self):
        """Annualised vol should be roughly daily × √252."""
        result = PerformanceAnalyzer().analyze(_returns())
        expected = result.daily_std * np.sqrt(252)
        assert result.annualised_vol == pytest.approx(expected, rel=1e-6)

    def test_var_negative(self):
        """95% VaR should be negative for typical returns."""
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.var < 0

    def test_cvar_worse_than_var(self):
        """CVaR should be more negative than VaR."""
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.cvar <= result.var


# ---------------------------------------------------------------------------
# Risk-adjusted ratios
# ---------------------------------------------------------------------------


class TestRiskAdjusted:
    def test_positive_sharpe(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.sharpe > 0

    def test_negative_sharpe_for_losers(self):
        result = PerformanceAnalyzer().analyze(_losing_returns())
        assert result.sharpe < 0

    def test_sortino_positive(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.sortino > 0

    def test_sortino_above_sharpe(self):
        """Sortino should generally be >= Sharpe for positive returns."""
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.sortino >= result.sharpe - 0.1  # Allow small tolerance

    def test_calmar_positive(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.calmar > 0

    def test_risk_free_rate_reduces_sharpe(self):
        r = _returns()
        sharpe_0 = PerformanceAnalyzer(AnalyticsConfig(risk_free_rate=0.0)).analyze(r).sharpe
        sharpe_5 = PerformanceAnalyzer(AnalyticsConfig(risk_free_rate=0.05)).analyze(r).sharpe
        assert sharpe_5 < sharpe_0


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


class TestDrawdown:
    def test_max_drawdown_negative(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.drawdown.max_drawdown < 0

    def test_avg_drawdown_negative(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.drawdown.avg_drawdown < 0

    def test_max_duration_positive(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.drawdown.max_duration_days > 0

    def test_n_drawdowns_positive(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.drawdown.n_drawdowns > 0

    def test_avg_drawdown_less_deep_than_max(self):
        result = PerformanceAnalyzer().analyze(_returns())
        # avg should be less negative than max
        assert result.drawdown.avg_drawdown >= result.drawdown.max_drawdown


# ---------------------------------------------------------------------------
# Win/loss
# ---------------------------------------------------------------------------


class TestWinLoss:
    def test_win_rate_in_range(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert 0 < result.win_rate < 1.0

    def test_profit_factor_above_one_for_winners(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.profit_factor > 1.0

    def test_profit_factor_below_one_for_losers(self):
        result = PerformanceAnalyzer().analyze(_losing_returns())
        assert result.profit_factor < 1.0

    def test_avg_win_positive(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.avg_win > 0

    def test_avg_loss_negative(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.avg_loss < 0

    def test_win_loss_ratio_positive(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert result.win_loss_ratio > 0


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------


class TestDistribution:
    def test_skewness_finite(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert np.isfinite(result.skewness)

    def test_kurtosis_finite(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert np.isfinite(result.kurtosis)

    def test_normal_returns_near_zero_skew(self):
        """Normal returns should have near-zero skewness."""
        result = PerformanceAnalyzer().analyze(_returns())
        assert abs(result.skewness) < 1.0  # Generous bound


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------


class TestRollingMetrics:
    def test_rolling_sharpe_populated(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert isinstance(result.rolling_sharpe, pd.Series)
        assert len(result.rolling_sharpe) == N_DAYS

    def test_rolling_vol_populated(self):
        result = PerformanceAnalyzer().analyze(_returns())
        assert isinstance(result.rolling_vol, pd.Series)

    def test_rolling_sharpe_has_valid_values(self):
        result = PerformanceAnalyzer().analyze(_returns())
        valid = result.rolling_sharpe.dropna()
        assert len(valid) > 0

    def test_custom_window(self):
        cfg = AnalyticsConfig(rolling_window=21)
        result = PerformanceAnalyzer(cfg).analyze(_returns())
        valid = result.rolling_sharpe.dropna()
        # With shorter window, should have more valid values
        assert len(valid) >= N_DAYS - 21


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_two_observations(self):
        r = pd.Series([0.01, -0.01])
        result = PerformanceAnalyzer().analyze(r)
        assert result.n_days == 2

    def test_all_positive_returns(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.uniform(0.001, 0.01, 100))
        result = PerformanceAnalyzer().analyze(r)
        assert result.win_rate == pytest.approx(1.0)
        assert result.avg_loss == 0.0

    def test_nan_handling(self):
        r = _returns()
        r.iloc[10] = np.nan
        r.iloc[20] = np.nan
        result = PerformanceAnalyzer().analyze(r)
        assert result.n_days == N_DAYS - 2


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        result = PerformanceAnalyzer().analyze(_returns())
        summary = result.summary()
        assert "Performance Analytics" in summary
        assert "Sharpe" in summary
        assert "Drawdown" in summary
        assert "Win rate" in summary
        assert "Calmar" in summary
