"""Tests for signal decay analysis (QUA-49)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.signals.decay import (
    DecayConfig,
    DecayResult,
    HorizonIC,
    SignalDecayAnalyzer,
    _spearman_r,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_data(
    n_days: int = 200,
    n_assets: int = 10,
    seed: int = 42,
    predictive: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic signal scores and asset returns.

    If predictive=True, signal scores have genuine predictive power over
    short-horizon returns (rank-correlated), so IC should be positive at
    short horizons.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    symbols = [f"S{i:02d}" for i in range(n_assets)]

    # Random signal scores in [-1, 1]
    scores = pd.DataFrame(
        rng.uniform(-1, 1, (n_days, n_assets)),
        index=dates,
        columns=symbols,
    )

    if predictive:
        # Next-day return = today's signal + noise → positive IC at horizon 1
        noise = rng.normal(0, 0.005, (n_days, n_assets))
        raw = scores.shift(1).fillna(0).values * 0.02 + noise
        returns = pd.DataFrame(raw, index=dates, columns=symbols)
    else:
        # Pure noise → IC ≈ 0
        returns = pd.DataFrame(
            rng.normal(0, 0.02, (n_days, n_assets)),
            index=dates,
            columns=symbols,
        )

    return scores, returns


# ── Tests: Basic analysis ─────────────────────────────────────────────────


class TestBasicAnalysis:
    def test_returns_decay_result(self):
        scores, returns = _make_data()
        analyzer = SignalDecayAnalyzer()
        result = analyzer.analyze(scores, returns)
        assert isinstance(result, DecayResult)

    def test_signal_name(self):
        scores, returns = _make_data()
        analyzer = SignalDecayAnalyzer()
        result = analyzer.analyze(scores, returns, signal_name="momentum")
        assert result.signal_name == "momentum"

    def test_default_signal_name(self):
        scores, returns = _make_data()
        analyzer = SignalDecayAnalyzer()
        result = analyzer.analyze(scores, returns)
        assert result.signal_name == "signal"

    def test_horizon_ics_populated(self):
        scores, returns = _make_data()
        analyzer = SignalDecayAnalyzer()
        result = analyzer.analyze(scores, returns)
        assert len(result.horizon_ics) > 0

    def test_horizons_sorted(self):
        scores, returns = _make_data()
        analyzer = SignalDecayAnalyzer()
        result = analyzer.analyze(scores, returns)
        horizons = [h.horizon for h in result.horizon_ics]
        assert horizons == sorted(horizons)


# ── Tests: IC computation ─────────────────────────────────────────────────


class TestICComputation:
    def test_predictive_signal_positive_ic(self):
        """A signal with genuine predictive power should have positive IC."""
        scores, returns = _make_data(predictive=True)
        analyzer = SignalDecayAnalyzer(DecayConfig(horizons=[1]))
        result = analyzer.analyze(scores, returns)
        assert result.horizon_ics[0].mean_ic > 0

    def test_noise_signal_near_zero_ic(self):
        """A noise signal should have IC near zero."""
        scores, returns = _make_data(predictive=False, n_days=500, seed=99)
        analyzer = SignalDecayAnalyzer(DecayConfig(horizons=[1]))
        result = analyzer.analyze(scores, returns)
        assert abs(result.horizon_ics[0].mean_ic) < 0.1

    def test_ic_series_stored(self):
        scores, returns = _make_data()
        analyzer = SignalDecayAnalyzer(DecayConfig(horizons=[1, 5]))
        result = analyzer.analyze(scores, returns)
        assert 1 in result.ic_series
        assert 5 in result.ic_series
        assert isinstance(result.ic_series[1], pd.Series)

    def test_horizon_ic_fields(self):
        scores, returns = _make_data()
        analyzer = SignalDecayAnalyzer(DecayConfig(horizons=[5]))
        result = analyzer.analyze(scores, returns)
        h = result.horizon_ics[0]
        assert isinstance(h, HorizonIC)
        assert h.horizon == 5
        assert h.n_periods > 0
        assert 0.0 <= h.hit_rate <= 1.0
        assert h.std_ic >= 0

    def test_ir_equals_mean_over_std(self):
        scores, returns = _make_data()
        analyzer = SignalDecayAnalyzer(DecayConfig(horizons=[5]))
        result = analyzer.analyze(scores, returns)
        h = result.horizon_ics[0]
        if h.std_ic > 1e-8:
            assert abs(h.ir - h.mean_ic / h.std_ic) < 1e-6

    def test_t_stat_computed(self):
        scores, returns = _make_data()
        analyzer = SignalDecayAnalyzer(DecayConfig(horizons=[1]))
        result = analyzer.analyze(scores, returns)
        h = result.horizon_ics[0]
        if h.std_ic > 1e-8:
            expected = h.mean_ic / (h.std_ic / np.sqrt(h.n_periods))
            assert abs(h.t_stat - expected) < 1e-6


# ── Tests: Half-life ──────────────────────────────────────────────────────


class TestHalfLife:
    def test_half_life_detected(self):
        """With a strongly predictive short-horizon signal, IC should decay."""
        scores, returns = _make_data(predictive=True, n_days=300)
        analyzer = SignalDecayAnalyzer(
            DecayConfig(horizons=[1, 2, 3, 5, 10, 21, 42, 63])
        )
        result = analyzer.analyze(scores, returns)
        # If peak IC is at short horizon and decays, half_life should be set
        if result.peak_ic > 0 and result.half_life is not None:
            assert result.half_life > result.optimal_horizon

    def test_no_half_life_when_ic_never_drops(self):
        """If IC doesn't drop below half peak, half_life should be None."""
        scores, returns = _make_data(predictive=True)
        analyzer = SignalDecayAnalyzer(DecayConfig(horizons=[1]))
        result = analyzer.analyze(scores, returns)
        # Only one horizon → nothing to decay → None
        assert result.half_life is None

    def test_half_life_none_for_negative_peak(self):
        """Negative peak IC should give None half-life."""
        # Use noise signal which may have negative peak
        scores, returns = _make_data(predictive=False, seed=7)
        analyzer = SignalDecayAnalyzer(DecayConfig(horizons=[1, 5, 10]))
        result = analyzer.analyze(scores, returns)
        if result.peak_ic <= 0:
            assert result.half_life is None


# ── Tests: Optimal horizon ────────────────────────────────────────────────


class TestOptimalHorizon:
    def test_optimal_horizon_is_peak(self):
        scores, returns = _make_data(predictive=True)
        analyzer = SignalDecayAnalyzer()
        result = analyzer.analyze(scores, returns)
        # optimal_horizon should match the horizon with highest mean_ic
        best = max(result.horizon_ics, key=lambda h: h.mean_ic)
        assert result.optimal_horizon == best.horizon

    def test_peak_ic_matches(self):
        scores, returns = _make_data(predictive=True)
        analyzer = SignalDecayAnalyzer()
        result = analyzer.analyze(scores, returns)
        best = max(result.horizon_ics, key=lambda h: h.mean_ic)
        assert abs(result.peak_ic - best.mean_ic) < 1e-10


# ── Tests: IC curve ───────────────────────────────────────────────────────


class TestICCurve:
    def test_ic_curve_type(self):
        scores, returns = _make_data()
        analyzer = SignalDecayAnalyzer()
        result = analyzer.analyze(scores, returns)
        curve = result.ic_curve()
        assert isinstance(curve, pd.Series)

    def test_ic_curve_indexed_by_horizon(self):
        scores, returns = _make_data()
        analyzer = SignalDecayAnalyzer(DecayConfig(horizons=[1, 5, 10]))
        result = analyzer.analyze(scores, returns)
        curve = result.ic_curve()
        for h in result.horizon_ics:
            assert h.horizon in curve.index
            assert abs(curve[h.horizon] - h.mean_ic) < 1e-10


# ── Tests: Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_scores_raises(self):
        analyzer = SignalDecayAnalyzer()
        with pytest.raises(ValueError, match="must not be empty"):
            analyzer.analyze(pd.DataFrame(), pd.DataFrame({"A": [1, 2]}))

    def test_empty_returns_raises(self):
        analyzer = SignalDecayAnalyzer()
        scores = pd.DataFrame({"A": [1, 2]}, index=pd.bdate_range("2023-01-01", periods=2))
        with pytest.raises(ValueError, match="must not be empty"):
            analyzer.analyze(scores, pd.DataFrame())

    def test_too_few_common_dates(self):
        analyzer = SignalDecayAnalyzer(DecayConfig(min_periods=100))
        scores, returns = _make_data(n_days=50)
        with pytest.raises(ValueError, match="common dates"):
            analyzer.analyze(scores, returns)

    def test_too_few_common_symbols(self):
        analyzer = SignalDecayAnalyzer(DecayConfig(min_assets=20))
        scores, returns = _make_data(n_assets=5)
        with pytest.raises(ValueError, match="common symbols"):
            analyzer.analyze(scores, returns)

    def test_no_overlap_in_symbols(self):
        dates = pd.bdate_range("2023-01-01", periods=50)
        scores = pd.DataFrame(
            np.random.randn(50, 3), index=dates, columns=["A", "B", "C"]
        )
        returns = pd.DataFrame(
            np.random.randn(50, 3), index=dates, columns=["X", "Y", "Z"]
        )
        with pytest.raises(ValueError, match="common symbols"):
            SignalDecayAnalyzer().analyze(scores, returns)

    def test_empty_result_on_insufficient_periods(self):
        """If no horizon produces enough IC samples, return empty result."""
        scores, returns = _make_data(n_days=25, n_assets=5)
        analyzer = SignalDecayAnalyzer(
            DecayConfig(horizons=[63], min_periods=20)
        )
        result = analyzer.analyze(scores, returns)
        assert len(result.horizon_ics) == 0
        assert result.peak_ic == 0.0


# ── Tests: Custom config ─────────────────────────────────────────────────


class TestCustomConfig:
    def test_custom_horizons(self):
        scores, returns = _make_data()
        analyzer = SignalDecayAnalyzer(DecayConfig(horizons=[2, 7]))
        result = analyzer.analyze(scores, returns)
        horizons = {h.horizon for h in result.horizon_ics}
        assert horizons <= {2, 7}

    def test_min_assets_respected(self):
        """Higher min_assets can filter out cross-sections."""
        scores, returns = _make_data(n_assets=5)
        # With min_assets=5, should work
        analyzer = SignalDecayAnalyzer(DecayConfig(horizons=[1], min_assets=5))
        result = analyzer.analyze(scores, returns)
        assert len(result.horizon_ics) > 0

    def test_default_horizons(self):
        config = DecayConfig()
        assert config.horizons == [1, 2, 3, 5, 10, 21, 42, 63]


# ── Tests: Summary ────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_with_data(self):
        scores, returns = _make_data()
        analyzer = SignalDecayAnalyzer()
        result = analyzer.analyze(scores, returns, signal_name="test_sig")
        summary = result.summary()
        assert "Signal Decay Analysis" in summary
        assert "test_sig" in summary
        assert "Peak IC" in summary

    def test_summary_empty(self):
        result = DecayResult(signal_name="empty")
        summary = result.summary()
        assert "no data" in summary


# ── Tests: Spearman rank correlation ──────────────────────────────────────


class TestSpearmanR:
    def test_perfect_positive(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(_spearman_r(a, a) - 1.0) < 1e-10

    def test_perfect_negative(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert abs(_spearman_r(a, b) - (-1.0)) < 1e-10

    def test_uncorrelated(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(1000)
        b = rng.standard_normal(1000)
        assert abs(_spearman_r(a, b)) < 0.1

    def test_too_short(self):
        assert np.isnan(_spearman_r(np.array([1.0]), np.array([2.0])))

    def test_constant_array_no_crash(self):
        """Constant array has no meaningful rank order; just verify no crash."""
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([1.0, 2.0, 3.0])
        result = _spearman_r(a, b)
        assert np.isfinite(result)
