"""Tests for signal decay analysis (QUA-90)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.signal_decay import (
    DecayConfig,
    DecayResult,
    LagMetric,
    SignalDecayAnalyzer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_DATES = 200
N_SYMBOLS = 30
SYMBOLS = [f"S{i:03d}" for i in range(N_SYMBOLS)]


def _make_predictive_signal(
    decay_rate: float = 0.85,
    noise: float = 0.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a signal that predicts 1-day returns with exponential decay."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=N_DATES + 30)

    signal = pd.DataFrame(
        rng.normal(0, 1, (len(dates), N_SYMBOLS)),
        index=dates, columns=SYMBOLS,
    )

    returns = pd.DataFrame(0.0, index=dates, columns=SYMBOLS)
    for lag in range(1, 11):
        weight = decay_rate ** (lag - 1)
        returns += weight * signal.shift(lag) * 0.01

    returns += rng.normal(0, noise * 0.01, returns.shape)
    returns = returns.iloc[30:]
    signal = signal.iloc[30:]

    return signal, returns


def _make_random_signal(seed: int = 99) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a purely random signal with no predictive power."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=N_DATES)
    signal = pd.DataFrame(
        rng.normal(0, 1, (N_DATES, N_SYMBOLS)),
        index=dates, columns=SYMBOLS,
    )
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.02, (N_DATES, N_SYMBOLS)),
        index=dates, columns=SYMBOLS,
    )
    return signal, returns


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        signal, returns = _make_predictive_signal()
        result = SignalDecayAnalyzer().analyze(signal, returns)
        assert isinstance(result, DecayResult)

    def test_lag_metrics_populated(self):
        signal, returns = _make_predictive_signal()
        result = SignalDecayAnalyzer(DecayConfig(max_lag=10)).analyze(signal, returns)
        assert len(result.lag_metrics) == 10

    def test_lag_metric_types(self):
        signal, returns = _make_predictive_signal()
        result = SignalDecayAnalyzer().analyze(signal, returns)
        for lm in result.lag_metrics:
            assert isinstance(lm, LagMetric)

    def test_n_symbols_tracked(self):
        signal, returns = _make_predictive_signal()
        result = SignalDecayAnalyzer().analyze(signal, returns)
        assert result.n_symbols == N_SYMBOLS

    def test_cumulative_ic_length(self):
        signal, returns = _make_predictive_signal()
        cfg = DecayConfig(max_lag=10)
        result = SignalDecayAnalyzer(cfg).analyze(signal, returns)
        assert len(result.cumulative_ic) == 10


# ---------------------------------------------------------------------------
# Predictive signal
# ---------------------------------------------------------------------------


class TestPredictiveSignal:
    def test_peak_ic_positive(self):
        signal, returns = _make_predictive_signal()
        result = SignalDecayAnalyzer().analyze(signal, returns)
        assert result.peak_ic > 0.01

    def test_peak_lag_early(self):
        signal, returns = _make_predictive_signal(decay_rate=0.85)
        result = SignalDecayAnalyzer().analyze(signal, returns)
        assert result.peak_lag <= 5

    def test_ic_decays_from_peak(self):
        signal, returns = _make_predictive_signal()
        result = SignalDecayAnalyzer(DecayConfig(max_lag=15)).analyze(signal, returns)
        last_ic = result.lag_metrics[-1].mean_ic
        assert last_ic < result.peak_ic

    def test_half_life_exists(self):
        signal, returns = _make_predictive_signal(decay_rate=0.70)
        result = SignalDecayAnalyzer(DecayConfig(max_lag=20)).analyze(signal, returns)
        assert result.half_life is not None
        assert result.half_life > result.peak_lag

    def test_icir_positive_for_good_signal(self):
        signal, returns = _make_predictive_signal()
        result = SignalDecayAnalyzer().analyze(signal, returns)
        assert result.lag_metrics[0].icir > 0

    def test_hit_rate_above_half(self):
        signal, returns = _make_predictive_signal()
        result = SignalDecayAnalyzer().analyze(signal, returns)
        peak_lm = next(lm for lm in result.lag_metrics if lm.lag == result.peak_lag)
        assert peak_lm.hit_rate > 0.50


# ---------------------------------------------------------------------------
# Random signal
# ---------------------------------------------------------------------------


class TestRandomSignal:
    def test_random_low_ic(self):
        signal, returns = _make_random_signal()
        result = SignalDecayAnalyzer().analyze(signal, returns)
        assert abs(result.peak_ic) < 0.10

    def test_random_icir_near_zero(self):
        signal, returns = _make_random_signal()
        result = SignalDecayAnalyzer().analyze(signal, returns)
        for lm in result.lag_metrics:
            assert abs(lm.icir) < 1.0


# ---------------------------------------------------------------------------
# Cumulative IC and optimal holding
# ---------------------------------------------------------------------------


class TestCumulativeIC:
    def test_cumulative_ic_starts_positive(self):
        signal, returns = _make_predictive_signal()
        result = SignalDecayAnalyzer(DecayConfig(max_lag=5)).analyze(signal, returns)
        assert result.cumulative_ic[0] > 0

    def test_optimal_holding_positive(self):
        signal, returns = _make_predictive_signal()
        result = SignalDecayAnalyzer().analyze(signal, returns)
        assert result.optimal_holding_period >= 1

    def test_optimal_net_ic_meaningful(self):
        signal, returns = _make_predictive_signal()
        result = SignalDecayAnalyzer().analyze(signal, returns)
        assert result.optimal_net_ic > -1.0


# ---------------------------------------------------------------------------
# IC time series
# ---------------------------------------------------------------------------


class TestICSeries:
    def test_ic_series_populated(self):
        signal, returns = _make_predictive_signal()
        result = SignalDecayAnalyzer(DecayConfig(max_lag=5)).analyze(signal, returns)
        assert len(result.ic_series) > 0

    def test_ic_series_are_pandas(self):
        signal, returns = _make_predictive_signal()
        result = SignalDecayAnalyzer(DecayConfig(max_lag=5)).analyze(signal, returns)
        for _lag, series in result.ic_series.items():
            assert isinstance(series, pd.Series)
            assert len(series) > 0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfig:
    def test_max_lag_respected(self):
        signal, returns = _make_predictive_signal()
        result = SignalDecayAnalyzer(DecayConfig(max_lag=5)).analyze(signal, returns)
        assert len(result.lag_metrics) == 5

    def test_higher_cost_later_optimal(self):
        signal, returns = _make_predictive_signal()
        low_cost = SignalDecayAnalyzer(
            DecayConfig(turnover_cost_bps=5.0, max_lag=15),
        ).analyze(signal, returns)
        high_cost = SignalDecayAnalyzer(
            DecayConfig(turnover_cost_bps=100.0, max_lag=15),
        ).analyze(signal, returns)
        assert high_cost.optimal_holding_period >= low_cost.optimal_holding_period


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_too_few_dates_raises(self):
        dates = pd.bdate_range("2023-01-01", periods=5)
        signal = pd.DataFrame(np.zeros((5, 20)), index=dates)
        returns = pd.DataFrame(np.zeros((5, 20)), index=dates)
        with pytest.raises(ValueError, match="at least.*dates"):
            SignalDecayAnalyzer(DecayConfig(min_dates=20)).analyze(signal, returns)

    def test_too_few_symbols_raises(self):
        dates = pd.bdate_range("2023-01-01", periods=100)
        signal = pd.DataFrame(np.zeros((100, 3)), index=dates, columns=["A", "B", "C"])
        returns = pd.DataFrame(np.zeros((100, 3)), index=dates, columns=["A", "B", "C"])
        with pytest.raises(ValueError, match="at least.*symbols"):
            SignalDecayAnalyzer(DecayConfig(min_observations=10)).analyze(signal, returns)

    def test_partial_overlap_works(self):
        rng = np.random.default_rng(42)
        dates1 = pd.bdate_range("2023-01-01", periods=150)
        dates2 = pd.bdate_range("2023-03-01", periods=150)
        cols = [f"S{i}" for i in range(20)]
        signal = pd.DataFrame(rng.normal(0, 1, (150, 20)), index=dates1, columns=cols)
        returns = pd.DataFrame(rng.normal(0, 0.02, (150, 20)), index=dates2, columns=cols)
        result = SignalDecayAnalyzer(
            DecayConfig(min_dates=10, max_lag=5),
        ).analyze(signal, returns)
        assert isinstance(result, DecayResult)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        signal, returns = _make_predictive_signal()
        result = SignalDecayAnalyzer(DecayConfig(max_lag=5)).analyze(signal, returns)
        summary = result.summary()
        assert "Signal Decay" in summary
        assert "Peak IC" in summary
        assert "half-life" in summary.lower()
        assert "Optimal holding" in summary
        assert "ICIR" in summary
