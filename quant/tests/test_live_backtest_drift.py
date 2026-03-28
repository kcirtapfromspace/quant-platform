"""Tests for live vs backtest drift monitor (QUA-94)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.monitoring.live_backtest_drift import (
    AlertLevel,
    DriftConfig,
    DriftMonitor,
    DriftResult,
    DriftSnapshot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_DAYS = 200
DATES = pd.bdate_range("2024-01-01", periods=N_DAYS)


def _make_aligned_returns(
    live_mean: float = 0.0003,
    bt_mean: float = 0.0003,
    live_vol: float = 0.01,
    bt_vol: float = 0.01,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """Create aligned live and backtest return series."""
    rng = np.random.default_rng(seed)
    live = pd.Series(
        rng.normal(live_mean, live_vol, N_DAYS), index=DATES, name="live",
    )
    bt = pd.Series(
        rng.normal(bt_mean, bt_vol, N_DAYS), index=DATES, name="backtest",
    )
    return live, bt


def _make_drifting_returns(seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Live significantly underperforms backtest (execution leakage)."""
    rng = np.random.default_rng(seed)
    bt = pd.Series(
        rng.normal(0.0005, 0.01, N_DAYS), index=DATES, name="backtest",
    )
    live = bt - 0.0010  # Consistent 10bps/day underperformance
    live += rng.normal(0, 0.002, N_DAYS)
    live.name = "live"
    return live, bt


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        live, bt = _make_aligned_returns()
        result = DriftMonitor().analyze(live, bt)
        assert isinstance(result, DriftResult)

    def test_n_days(self):
        live, bt = _make_aligned_returns()
        result = DriftMonitor().analyze(live, bt)
        assert result.n_days == N_DAYS

    def test_snapshots_populated(self):
        live, bt = _make_aligned_returns()
        result = DriftMonitor().analyze(live, bt)
        assert len(result.snapshots) == N_DAYS

    def test_snapshot_types(self):
        live, bt = _make_aligned_returns()
        result = DriftMonitor().analyze(live, bt)
        for s in result.snapshots:
            assert isinstance(s, DriftSnapshot)

    def test_sharpe_computed(self):
        live, bt = _make_aligned_returns(live_mean=0.001, bt_mean=0.001)
        result = DriftMonitor().analyze(live, bt)
        assert result.live_sharpe != 0.0
        assert result.backtest_sharpe != 0.0


# ---------------------------------------------------------------------------
# Aligned returns (no drift)
# ---------------------------------------------------------------------------


class TestAligned:
    def test_green_alert(self):
        live, bt = _make_aligned_returns()
        result = DriftMonitor().analyze(live, bt)
        assert result.current_alert == AlertLevel.GREEN

    def test_low_z_score(self):
        live, bt = _make_aligned_returns()
        result = DriftMonitor().analyze(live, bt)
        assert abs(result.current_z_score) < 2.0

    def test_small_return_gap(self):
        live, bt = _make_aligned_returns()
        result = DriftMonitor().analyze(live, bt)
        assert abs(result.cum_return_gap) < 0.20

    def test_few_yellow_days(self):
        live, bt = _make_aligned_returns()
        result = DriftMonitor().analyze(live, bt)
        assert result.days_in_yellow < N_DAYS * 0.3


# ---------------------------------------------------------------------------
# Drifting returns
# ---------------------------------------------------------------------------


class TestDrifting:
    def test_high_z_score(self):
        live, bt = _make_drifting_returns()
        result = DriftMonitor().analyze(live, bt)
        assert abs(result.current_z_score) > 2.0

    def test_negative_return_gap(self):
        """Live underperforms => negative gap."""
        live, bt = _make_drifting_returns()
        result = DriftMonitor().analyze(live, bt)
        assert result.cum_return_gap < -0.05

    def test_alert_triggered(self):
        live, bt = _make_drifting_returns()
        result = DriftMonitor(DriftConfig(
            yellow_threshold_z=1.5, red_threshold_z=2.5,
        )).analyze(live, bt)
        assert result.current_alert != AlertLevel.GREEN

    def test_days_in_red(self):
        live, bt = _make_drifting_returns()
        result = DriftMonitor(DriftConfig(red_threshold_z=2.0)).analyze(live, bt)
        assert result.days_in_red > 0

    def test_sharpe_gap_negative(self):
        live, bt = _make_drifting_returns()
        result = DriftMonitor().analyze(live, bt)
        assert result.live_sharpe < result.backtest_sharpe


# ---------------------------------------------------------------------------
# Alert levels
# ---------------------------------------------------------------------------


class TestAlertLevels:
    def test_green_below_yellow(self):
        """Identical returns should be GREEN."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, N_DAYS), index=DATES)
        result = DriftMonitor().analyze(returns, returns.copy())
        assert result.current_alert == AlertLevel.GREEN

    def test_configurable_thresholds(self):
        """Tighter thresholds should trigger more alerts."""
        live, bt = _make_aligned_returns(live_mean=0.0001, bt_mean=0.0003)
        tight = DriftMonitor(DriftConfig(
            yellow_threshold_z=0.5, red_threshold_z=1.0,
        )).analyze(live, bt)
        loose = DriftMonitor(DriftConfig(
            yellow_threshold_z=3.0, red_threshold_z=5.0,
        )).analyze(live, bt)
        assert tight.days_in_yellow >= loose.days_in_yellow


# ---------------------------------------------------------------------------
# Tracking error
# ---------------------------------------------------------------------------


class TestTrackingError:
    def test_tracking_error_positive(self):
        live, bt = _make_aligned_returns()
        result = DriftMonitor().analyze(live, bt)
        assert result.tracking_error > 0

    def test_identical_returns_zero_te(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, N_DAYS), index=DATES)
        result = DriftMonitor().analyze(returns, returns.copy())
        assert result.tracking_error < 0.001

    def test_larger_divergence_higher_te(self):
        small = DriftMonitor().analyze(
            *_make_aligned_returns(live_vol=0.01, bt_vol=0.01)
        )
        large = DriftMonitor().analyze(
            *_make_aligned_returns(live_vol=0.03, bt_vol=0.01)
        )
        assert large.tracking_error > small.tracking_error


# ---------------------------------------------------------------------------
# Max z-score
# ---------------------------------------------------------------------------


class TestMaxZ:
    def test_max_z_tracked(self):
        live, bt = _make_aligned_returns()
        result = DriftMonitor().analyze(live, bt)
        assert abs(result.max_z_score) >= abs(result.current_z_score) - 1e-6

    def test_max_z_date_populated(self):
        live, bt = _make_aligned_returns()
        result = DriftMonitor().analyze(live, bt)
        assert result.max_z_date is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_too_few_dates_raises(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        live = pd.Series([0.01] * 5, index=dates)
        bt = pd.Series([0.01] * 5, index=dates)
        with pytest.raises(ValueError, match="at least"):
            DriftMonitor(DriftConfig(min_observations=20)).analyze(live, bt)

    def test_partial_overlap(self):
        dates1 = pd.bdate_range("2024-01-01", periods=150)
        dates2 = pd.bdate_range("2024-03-01", periods=150)
        rng = np.random.default_rng(42)
        live = pd.Series(rng.normal(0, 0.01, 150), index=dates1)
        bt = pd.Series(rng.normal(0, 0.01, 150), index=dates2)
        result = DriftMonitor(DriftConfig(min_observations=10)).analyze(live, bt)
        assert isinstance(result, DriftResult)
        assert result.n_days > 0

    def test_constant_returns(self):
        live = pd.Series(0.001, index=DATES)
        bt = pd.Series(0.001, index=DATES)
        result = DriftMonitor().analyze(live, bt)
        assert result.current_alert == AlertLevel.GREEN


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        live, bt = _make_aligned_returns()
        result = DriftMonitor().analyze(live, bt)
        summary = result.summary()
        assert "Live vs Backtest" in summary
        assert "alert" in summary.lower()
        assert "z-score" in summary.lower()
        assert "Tracking error" in summary

    def test_summary_shows_sharpe(self):
        live, bt = _make_aligned_returns()
        result = DriftMonitor().analyze(live, bt)
        summary = result.summary()
        assert "Sharpe" in summary
