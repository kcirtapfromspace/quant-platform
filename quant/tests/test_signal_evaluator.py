"""Tests for cross-sectional signal evaluator (QUA-108)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.signal_evaluator import (
    EvaluationResult,
    EvaluatorConfig,
    QuantileStats,
    SignalEvaluator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_ASSETS = 50
N_DATES = 200


def _signal_and_returns(
    seed: int = 42, ic: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate signal and returns with a controlled IC level.

    The signal predicts a fraction ``ic`` of the return variance.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=N_DATES)
    symbols = [f"S{i}" for i in range(N_ASSETS)]

    signal_vals = rng.normal(0, 1, (N_DATES, N_ASSETS))
    noise = rng.normal(0, 1, (N_DATES, N_ASSETS))
    # Returns = ic * signal + noise (rank correlation ≈ ic for small ic)
    returns_vals = ic * signal_vals + np.sqrt(1 - ic ** 2) * noise
    returns_vals *= 0.01  # scale to realistic return magnitudes

    signal_df = pd.DataFrame(signal_vals, index=dates, columns=symbols)
    returns_df = pd.DataFrame(returns_vals, index=dates, columns=symbols)
    return signal_df, returns_df


def _perfect_signal(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Signal that perfectly predicts next-day return ranking."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=N_DATES)
    symbols = [f"S{i}" for i in range(N_ASSETS)]

    signal_vals = rng.normal(0, 1, (N_DATES, N_ASSETS))
    # ret[t+1] = signal[t] * 0.01 → signal[t] perfectly predicts fwd return
    returns_vals = np.zeros_like(signal_vals)
    returns_vals[1:] = signal_vals[:-1] * 0.01

    signal_df = pd.DataFrame(signal_vals, index=dates, columns=symbols)
    returns_df = pd.DataFrame(returns_vals, index=dates, columns=symbols)
    return signal_df, returns_df


def _zero_signal() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Signal with no predictive power."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-01", periods=N_DATES)
    symbols = [f"S{i}" for i in range(N_ASSETS)]

    signal_vals = rng.normal(0, 1, (N_DATES, N_ASSETS))
    # Returns are independent of signal
    returns_vals = rng.normal(0, 0.01, (N_DATES, N_ASSETS))

    signal_df = pd.DataFrame(signal_vals, index=dates, columns=symbols)
    returns_df = pd.DataFrame(returns_vals, index=dates, columns=symbols)
    return signal_df, returns_df


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        sig, ret = _signal_and_returns()
        result = SignalEvaluator().evaluate(sig, ret)
        assert isinstance(result, EvaluationResult)

    def test_n_dates(self):
        sig, ret = _signal_and_returns()
        result = SignalEvaluator().evaluate(sig, ret)
        # Forward period = 1, so we lose 1 date
        assert result.n_dates > 0
        assert result.n_dates <= N_DATES - 1

    def test_n_assets_avg(self):
        sig, ret = _signal_and_returns()
        result = SignalEvaluator().evaluate(sig, ret)
        assert result.n_assets_avg == pytest.approx(N_ASSETS, abs=1)

    def test_config_accessible(self):
        cfg = EvaluatorConfig(n_quantiles=10)
        evaluator = SignalEvaluator(cfg)
        assert evaluator.config.n_quantiles == 10

    def test_too_few_dates_raises(self):
        sig, ret = _signal_and_returns()
        cfg = EvaluatorConfig(min_dates=9999)
        with pytest.raises(ValueError, match="at least 9999"):
            SignalEvaluator(cfg).evaluate(sig, ret)

    def test_too_few_common_assets_raises(self):
        sig, ret = _signal_and_returns()
        # Keep only 5 assets in signal, need 20
        sig_small = sig.iloc[:, :5]
        with pytest.raises(ValueError, match="at least 20"):
            SignalEvaluator().evaluate(sig_small, ret)


# ---------------------------------------------------------------------------
# IC statistics
# ---------------------------------------------------------------------------


class TestICStatistics:
    def test_positive_ic_for_predictive_signal(self):
        sig, ret = _signal_and_returns(ic=0.10)
        result = SignalEvaluator().evaluate(sig, ret)
        assert result.mean_ic > 0

    def test_near_zero_ic_for_random_signal(self):
        sig, ret = _zero_signal()
        result = SignalEvaluator().evaluate(sig, ret)
        assert abs(result.mean_ic) < 0.05

    def test_high_ic_for_perfect_signal(self):
        sig, ret = _perfect_signal()
        result = SignalEvaluator().evaluate(sig, ret)
        assert result.mean_ic > 0.8

    def test_ic_std_positive(self):
        sig, ret = _signal_and_returns()
        result = SignalEvaluator().evaluate(sig, ret)
        assert result.ic_std > 0

    def test_ic_ir_proportional_to_signal_strength(self):
        """Stronger signal should have higher ICIR."""
        sig_weak, ret_weak = _signal_and_returns(ic=0.05, seed=42)
        sig_strong, ret_strong = _signal_and_returns(ic=0.20, seed=42)
        weak = SignalEvaluator().evaluate(sig_weak, ret_weak)
        strong = SignalEvaluator().evaluate(sig_strong, ret_strong)
        assert strong.ic_ir > weak.ic_ir

    def test_ic_series_length(self):
        sig, ret = _signal_and_returns()
        result = SignalEvaluator().evaluate(sig, ret)
        assert isinstance(result.ic_series, pd.Series)
        assert len(result.ic_series) == result.n_dates


# ---------------------------------------------------------------------------
# Quantile returns
# ---------------------------------------------------------------------------


class TestQuantileReturns:
    def test_n_quantiles(self):
        sig, ret = _signal_and_returns()
        cfg = EvaluatorConfig(n_quantiles=5)
        result = SignalEvaluator(cfg).evaluate(sig, ret)
        assert len(result.quantile_returns) == 5

    def test_quantile_stats_type(self):
        sig, ret = _signal_and_returns()
        result = SignalEvaluator().evaluate(sig, ret)
        for qs in result.quantile_returns:
            assert isinstance(qs, QuantileStats)

    def test_top_quantile_beats_bottom_for_good_signal(self):
        sig, ret = _signal_and_returns(ic=0.10)
        result = SignalEvaluator().evaluate(sig, ret)
        assert result.quantile_returns[-1].mean_return > (
            result.quantile_returns[0].mean_return
        )

    def test_long_short_positive_for_good_signal(self):
        sig, ret = _signal_and_returns(ic=0.10)
        result = SignalEvaluator().evaluate(sig, ret)
        assert result.long_short_return > 0

    def test_long_short_near_zero_for_random(self):
        sig, ret = _zero_signal()
        result = SignalEvaluator().evaluate(sig, ret)
        # Should be close to zero (within noise)
        assert abs(result.long_short_return) < 0.20  # annualised

    def test_quantile_observations_positive(self):
        sig, ret = _signal_and_returns()
        result = SignalEvaluator().evaluate(sig, ret)
        for qs in result.quantile_returns:
            assert qs.n_observations > 0

    def test_custom_n_quantiles(self):
        sig, ret = _signal_and_returns()
        cfg = EvaluatorConfig(n_quantiles=10)
        result = SignalEvaluator(cfg).evaluate(sig, ret)
        assert len(result.quantile_returns) == 10


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


class TestMonotonicity:
    def test_high_monotonicity_for_good_signal(self):
        sig, ret = _signal_and_returns(ic=0.15)
        result = SignalEvaluator().evaluate(sig, ret)
        assert result.monotonicity > 0.5

    def test_perfect_monotonicity_for_perfect_signal(self):
        sig, ret = _perfect_signal()
        result = SignalEvaluator().evaluate(sig, ret)
        assert result.monotonicity > 0.9

    def test_monotonicity_bounded(self):
        sig, ret = _signal_and_returns()
        result = SignalEvaluator().evaluate(sig, ret)
        assert -1.0 <= result.monotonicity <= 1.0


# ---------------------------------------------------------------------------
# Hit rate
# ---------------------------------------------------------------------------


class TestHitRate:
    def test_above_half_for_good_signal(self):
        sig, ret = _signal_and_returns(ic=0.10)
        result = SignalEvaluator().evaluate(sig, ret)
        assert result.hit_rate > 0.50

    def test_near_half_for_random(self):
        sig, ret = _zero_signal()
        result = SignalEvaluator().evaluate(sig, ret)
        assert 0.45 < result.hit_rate < 0.55

    def test_hit_rate_bounded(self):
        sig, ret = _signal_and_returns()
        result = SignalEvaluator().evaluate(sig, ret)
        assert 0.0 <= result.hit_rate <= 1.0


# ---------------------------------------------------------------------------
# Signal turnover
# ---------------------------------------------------------------------------


class TestTurnover:
    def test_turnover_bounded(self):
        sig, ret = _signal_and_returns()
        result = SignalEvaluator().evaluate(sig, ret)
        assert 0.0 <= result.turnover <= 2.0

    def test_low_turnover_for_stable_signal(self):
        """A signal that barely changes should have low turnover."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2023-01-01", periods=N_DATES)
        symbols = [f"S{i}" for i in range(N_ASSETS)]

        # Static signal with tiny perturbation
        base = rng.normal(0, 1, N_ASSETS)
        signal_vals = np.tile(base, (N_DATES, 1))
        signal_vals += rng.normal(0, 0.001, (N_DATES, N_ASSETS))
        returns_vals = rng.normal(0, 0.01, (N_DATES, N_ASSETS))

        sig = pd.DataFrame(signal_vals, index=dates, columns=symbols)
        ret = pd.DataFrame(returns_vals, index=dates, columns=symbols)

        result = SignalEvaluator().evaluate(sig, ret)
        assert result.turnover < 0.05

    def test_high_turnover_for_random_signal(self):
        """A signal that changes randomly should have turnover near 1.0."""
        sig, ret = _zero_signal()
        result = SignalEvaluator().evaluate(sig, ret)
        assert result.turnover > 0.5


# ---------------------------------------------------------------------------
# Forward period
# ---------------------------------------------------------------------------


class TestForwardPeriod:
    def test_custom_forward_period(self):
        sig, ret = _signal_and_returns()
        cfg = EvaluatorConfig(forward_period=5)
        result = SignalEvaluator(cfg).evaluate(sig, ret)
        assert result.n_dates > 0

    def test_longer_period_fewer_dates(self):
        sig, ret = _signal_and_returns()
        r1 = SignalEvaluator(EvaluatorConfig(forward_period=1)).evaluate(sig, ret)
        r5 = SignalEvaluator(EvaluatorConfig(forward_period=5)).evaluate(sig, ret)
        assert r5.n_dates <= r1.n_dates


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_nan_in_signal(self):
        sig, ret = _signal_and_returns()
        sig.iloc[5, 2] = np.nan
        sig.iloc[10, 3] = np.nan
        result = SignalEvaluator().evaluate(sig, ret)
        assert result.n_dates > 0

    def test_partial_column_overlap(self):
        sig, ret = _signal_and_returns()
        # Add extra columns to signal that don't exist in returns
        sig["EXTRA_1"] = 0.0
        sig["EXTRA_2"] = 0.0
        result = SignalEvaluator().evaluate(sig, ret)
        assert result.n_assets_avg == pytest.approx(N_ASSETS, abs=1)

    def test_min_observations_respected(self):
        sig, ret = _signal_and_returns()
        cfg = EvaluatorConfig(min_observations=N_ASSETS + 1)
        with pytest.raises(ValueError):
            SignalEvaluator(cfg).evaluate(sig, ret)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_contains_key_info(self):
        sig, ret = _signal_and_returns()
        result = SignalEvaluator().evaluate(sig, ret)
        summary = result.summary()
        assert "Signal Evaluation" in summary
        assert "Mean IC" in summary
        assert "ICIR" in summary
        assert "Hit rate" in summary
        assert "Turnover" in summary
        assert "Long-short" in summary
        assert "Monotonicity" in summary
