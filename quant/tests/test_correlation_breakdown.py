"""Tests for correlation breakdown detection (QUA-86)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.risk.correlation_breakdown import (
    BreakdownConfig,
    BreakdownEvent,
    BreakdownResult,
    CorrelationBreakdownMonitor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOLS = ["A", "B", "C", "D", "E"]


def _make_uncorrelated(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Independent returns — low average correlation."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0005, 0.01, (n, len(SYMBOLS)))
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame(data, index=dates, columns=SYMBOLS)


def _make_highly_correlated(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Returns driven by a single factor — high avg correlation."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0005, 0.015, n)
    data = np.column_stack([
        factor + rng.normal(0, 0.002, n) for _ in SYMBOLS
    ])
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame(data, index=dates, columns=SYMBOLS)


def _make_regime_switch(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Uncorrelated first half, highly correlated second half."""
    rng = np.random.default_rng(seed)
    half = n // 2
    # First half: independent
    data1 = rng.normal(0.0005, 0.01, (half, len(SYMBOLS)))
    # Second half: driven by single factor
    factor = rng.normal(-0.001, 0.02, n - half)
    data2 = np.column_stack([
        factor + rng.normal(0, 0.002, n - half) for _ in SYMBOLS
    ])
    data = np.vstack([data1, data2])
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame(data, index=dates, columns=SYMBOLS)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        monitor = CorrelationBreakdownMonitor()
        result = monitor.analyze(_make_uncorrelated())
        assert isinstance(result, BreakdownResult)

    def test_n_assets_tracked(self):
        monitor = CorrelationBreakdownMonitor()
        result = monitor.analyze(_make_uncorrelated())
        assert result.n_assets == len(SYMBOLS)

    def test_n_days_tracked(self):
        monitor = CorrelationBreakdownMonitor()
        result = monitor.analyze(_make_uncorrelated())
        assert result.n_days == 300

    def test_rolling_avg_corr_length(self):
        monitor = CorrelationBreakdownMonitor()
        result = monitor.analyze(_make_uncorrelated())
        assert len(result.rolling_avg_corr) == 300


# ---------------------------------------------------------------------------
# Correlation detection
# ---------------------------------------------------------------------------


class TestCorrelationDetection:
    def test_uncorrelated_low_avg(self):
        monitor = CorrelationBreakdownMonitor()
        result = monitor.analyze(_make_uncorrelated())
        assert result.current_avg_corr < 0.3

    def test_correlated_high_avg(self):
        monitor = CorrelationBreakdownMonitor()
        result = monitor.analyze(_make_highly_correlated())
        assert result.current_avg_corr > 0.5

    def test_uncorrelated_no_breakdown(self):
        monitor = CorrelationBreakdownMonitor()
        result = monitor.analyze(_make_uncorrelated())
        assert not result.is_in_breakdown

    def test_correlated_in_breakdown(self):
        monitor = CorrelationBreakdownMonitor(
            BreakdownConfig(breakdown_threshold=0.50)
        )
        result = monitor.analyze(_make_highly_correlated())
        assert result.is_in_breakdown


# ---------------------------------------------------------------------------
# Breakdown events
# ---------------------------------------------------------------------------


class TestBreakdownEvents:
    def test_regime_switch_detects_event(self):
        """Should detect breakdown in the correlated second half."""
        monitor = CorrelationBreakdownMonitor(
            BreakdownConfig(
                window=42,
                min_periods=20,
                breakdown_threshold=0.50,
            )
        )
        result = monitor.analyze(_make_regime_switch())
        assert result.n_breakdowns >= 1

    def test_events_are_breakdown_events(self):
        monitor = CorrelationBreakdownMonitor(
            BreakdownConfig(breakdown_threshold=0.50)
        )
        result = monitor.analyze(_make_highly_correlated())
        for ev in result.breakdown_events:
            assert isinstance(ev, BreakdownEvent)
            assert ev.duration_days >= 1
            assert ev.peak_correlation >= 0.50

    def test_no_events_when_uncorrelated(self):
        monitor = CorrelationBreakdownMonitor(
            BreakdownConfig(breakdown_threshold=0.70)
        )
        result = monitor.analyze(_make_uncorrelated())
        assert result.n_breakdowns == 0

    def test_pct_time_in_breakdown(self):
        monitor = CorrelationBreakdownMonitor(
            BreakdownConfig(breakdown_threshold=0.50)
        )
        result = monitor.analyze(_make_highly_correlated())
        assert 0.0 <= result.pct_time_in_breakdown <= 1.0


# ---------------------------------------------------------------------------
# Absorption ratio
# ---------------------------------------------------------------------------


class TestAbsorptionRatio:
    def test_absorption_between_zero_and_one(self):
        monitor = CorrelationBreakdownMonitor()
        result = monitor.analyze(_make_uncorrelated())
        valid = result.rolling_absorption.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1.0 + 1e-6).all()

    def test_correlated_high_absorption(self):
        """Highly correlated returns should have high absorption ratio."""
        monitor = CorrelationBreakdownMonitor()
        result = monitor.analyze(_make_highly_correlated())
        assert result.current_absorption > 0.5

    def test_uncorrelated_lower_absorption(self):
        monitor = CorrelationBreakdownMonitor()
        uncorr = monitor.analyze(_make_uncorrelated())
        corr = monitor.analyze(_make_highly_correlated())
        assert uncorr.current_absorption < corr.current_absorption


# ---------------------------------------------------------------------------
# Diversification ratio
# ---------------------------------------------------------------------------


class TestDiversificationRatio:
    def test_diversification_ratio_positive(self):
        monitor = CorrelationBreakdownMonitor()
        result = monitor.analyze(_make_uncorrelated())
        valid = result.diversification_ratio.dropna()
        assert (valid > 0).all()

    def test_uncorrelated_higher_diversification(self):
        """Uncorrelated assets should have higher diversification ratio."""
        monitor = CorrelationBreakdownMonitor()
        uncorr = monitor.analyze(_make_uncorrelated())
        corr = monitor.analyze(_make_highly_correlated())
        assert uncorr.current_diversification_ratio > corr.current_diversification_ratio

    def test_diversification_ratio_time_series(self):
        monitor = CorrelationBreakdownMonitor()
        result = monitor.analyze(_make_uncorrelated())
        assert len(result.diversification_ratio) == 300


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_too_few_assets_raises(self):
        monitor = CorrelationBreakdownMonitor()
        dates = pd.bdate_range("2023-01-01", periods=100)
        with pytest.raises(ValueError, match="at least 2"):
            monitor.analyze(pd.DataFrame({"A": np.zeros(100)}, index=dates))

    def test_too_few_days_raises(self):
        monitor = CorrelationBreakdownMonitor(BreakdownConfig(min_periods=30))
        dates = pd.bdate_range("2023-01-01", periods=10)
        data = pd.DataFrame(np.zeros((10, 3)), index=dates, columns=["A", "B", "C"])
        with pytest.raises(ValueError, match="at least 30"):
            monitor.analyze(data)

    def test_two_assets_works(self):
        rng = np.random.default_rng(42)
        n = 100
        dates = pd.bdate_range("2023-01-01", periods=n)
        data = pd.DataFrame(
            rng.normal(0, 0.01, (n, 2)),
            index=dates, columns=["A", "B"],
        )
        monitor = CorrelationBreakdownMonitor()
        result = monitor.analyze(data)
        assert isinstance(result, BreakdownResult)

    def test_custom_config(self):
        cfg = BreakdownConfig(
            window=42, min_periods=20,
            breakdown_threshold=0.80,
            absorption_n_components=2,
        )
        monitor = CorrelationBreakdownMonitor(cfg)
        result = monitor.analyze(_make_uncorrelated())
        assert isinstance(result, BreakdownResult)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        monitor = CorrelationBreakdownMonitor()
        result = monitor.analyze(_make_uncorrelated())
        summary = result.summary()
        assert "Correlation Breakdown" in summary
        assert "Current status" in summary
        assert "absorption" in summary.lower()
        assert "diversif" in summary.lower()

    def test_summary_with_events(self):
        monitor = CorrelationBreakdownMonitor(
            BreakdownConfig(breakdown_threshold=0.50)
        )
        result = monitor.analyze(_make_highly_correlated())
        summary = result.summary()
        if result.n_breakdowns > 0:
            assert "Breakdown Events" in summary
