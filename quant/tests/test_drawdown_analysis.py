"""Tests for drawdown recovery analysis (QUA-82)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.drawdown_analysis import (
    DrawdownAnalysisResult,
    DrawdownAnalyzer,
    DrawdownConfig,
    DrawdownEpisode,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_equity(n: int = 500, seed: int = 42) -> pd.Series:
    """Generate a synthetic equity curve with clear drawdowns."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.012, n)
    prices = np.cumprod(1 + rets) * 1_000_000
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(prices, index=dates, name="equity")


def _make_losing_equity(n: int = 300, seed: int = 42) -> pd.Series:
    """Equity curve with negative drift (ongoing drawdown)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(-0.001, 0.015, n)
    prices = np.cumprod(1 + rets) * 1_000_000
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(prices, index=dates, name="equity")


def _make_v_shaped_equity() -> pd.Series:
    """Equity that drops then recovers exactly: 100 → 80 → 100."""
    values = list(range(100, 80, -1)) + list(range(80, 101))
    dates = pd.bdate_range("2023-01-01", periods=len(values))
    return pd.Series(values, index=dates, dtype=float, name="equity")


def _make_flat_equity(n: int = 100) -> pd.Series:
    """Flat equity (no drawdowns)."""
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.Series([1_000_000.0] * n, index=dates, name="equity")


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        assert isinstance(result, DrawdownAnalysisResult)

    def test_n_episodes_positive(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        assert result.n_episodes > 0

    def test_episodes_are_drawdown_episodes(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        for ep in result.episodes:
            assert isinstance(ep, DrawdownEpisode)

    def test_max_drawdown_positive(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        assert result.max_drawdown > 0


# ---------------------------------------------------------------------------
# Episode detection
# ---------------------------------------------------------------------------


class TestEpisodeDetection:
    def test_v_shaped_single_episode(self):
        analyzer = DrawdownAnalyzer(DrawdownConfig(min_depth=0.0))
        result = analyzer.analyze(_make_v_shaped_equity())
        assert result.n_episodes >= 1

    def test_v_shaped_depth(self):
        analyzer = DrawdownAnalyzer(DrawdownConfig(min_depth=0.0))
        result = analyzer.analyze(_make_v_shaped_equity())
        # Should detect the 20% drawdown
        assert result.max_drawdown >= 0.19

    def test_v_shaped_recovery(self):
        analyzer = DrawdownAnalyzer(DrawdownConfig(min_depth=0.0))
        result = analyzer.analyze(_make_v_shaped_equity())
        recovered = [ep for ep in result.episodes if ep.recovery_date is not None]
        assert len(recovered) >= 1

    def test_episode_dates_ordered(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        for ep in result.episodes:
            assert ep.trough_date >= ep.peak_date
            if ep.recovery_date is not None:
                assert ep.recovery_date >= ep.trough_date

    def test_depth_within_bounds(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        for ep in result.episodes:
            assert 0 < ep.depth <= 1.0

    def test_min_depth_filter(self):
        equity = _make_equity()
        tight = DrawdownAnalyzer(DrawdownConfig(min_depth=0.001))
        loose = DrawdownAnalyzer(DrawdownConfig(min_depth=0.05))
        r_tight = tight.analyze(equity)
        r_loose = loose.analyze(equity)
        assert r_tight.n_episodes >= r_loose.n_episodes


# ---------------------------------------------------------------------------
# Depth statistics
# ---------------------------------------------------------------------------


class TestDepthStats:
    def test_avg_drawdown_bounded(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        assert 0 < result.avg_drawdown <= result.max_drawdown

    def test_median_drawdown_bounded(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        assert 0 < result.median_drawdown <= result.max_drawdown

    def test_p95_drawdown_bounded(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        assert result.p95_drawdown >= result.median_drawdown
        assert result.p95_drawdown <= result.max_drawdown


# ---------------------------------------------------------------------------
# Duration statistics
# ---------------------------------------------------------------------------


class TestDurationStats:
    def test_max_duration_positive(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        assert result.max_duration > 0

    def test_avg_duration_positive(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        assert result.avg_duration > 0

    def test_v_shaped_duration(self):
        analyzer = DrawdownAnalyzer(DrawdownConfig(min_depth=0.0))
        result = analyzer.analyze(_make_v_shaped_equity())
        # V-shape: 20 days down + 20 days up = 40 days
        deepest = max(result.episodes, key=lambda e: e.depth)
        assert deepest.drawdown_days >= 1
        if deepest.recovery_days is not None:
            assert deepest.recovery_days >= 1


# ---------------------------------------------------------------------------
# Underwater statistics
# ---------------------------------------------------------------------------


class TestUnderwaterStats:
    def test_pct_underwater_between_zero_and_one(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        assert 0.0 <= result.pct_time_underwater <= 1.0

    def test_losing_equity_high_underwater(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_losing_equity())
        # Strategy with negative drift should spend most time underwater
        assert result.pct_time_underwater > 0.5

    def test_flat_equity_no_underwater(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_flat_equity())
        assert result.total_underwater_days == 0
        assert result.pct_time_underwater == 0.0

    def test_longest_underwater_positive(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        assert result.longest_underwater >= 0


# ---------------------------------------------------------------------------
# Recovery statistics
# ---------------------------------------------------------------------------


class TestRecoveryStats:
    def test_recovery_rate_between_zero_and_one(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        assert 0.0 <= result.recovery_rate <= 1.0

    def test_n_recovered_bounded(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        assert result.n_recovered <= result.n_episodes

    def test_losing_equity_low_recovery(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_losing_equity())
        # Should have at least one unrecovered episode
        assert result.n_recovered < result.n_episodes or result.n_episodes == 0

    def test_v_shaped_full_recovery(self):
        analyzer = DrawdownAnalyzer(DrawdownConfig(min_depth=0.0))
        result = analyzer.analyze(_make_v_shaped_equity())
        assert result.recovery_rate == 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_too_short_raises(self):
        analyzer = DrawdownAnalyzer()
        with pytest.raises(ValueError, match="at least 2"):
            analyzer.analyze(pd.Series([100.0]))

    def test_empty_raises(self):
        analyzer = DrawdownAnalyzer()
        with pytest.raises(ValueError, match="at least 2"):
            analyzer.analyze(pd.Series(dtype=float))

    def test_two_points_no_drawdown(self):
        equity = pd.Series(
            [100.0, 105.0],
            index=pd.bdate_range("2023-01-01", periods=2),
        )
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(equity)
        assert result.n_episodes == 0
        assert result.max_drawdown == 0.0

    def test_monotonic_up_no_episodes(self):
        values = list(range(1, 101))
        dates = pd.bdate_range("2023-01-01", periods=100)
        equity = pd.Series(values, index=dates, dtype=float)
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(equity)
        assert result.n_episodes == 0

    def test_flat_no_episodes(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_flat_equity())
        assert result.n_episodes == 0
        assert result.recovery_rate == 0.0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        summary = result.summary()
        assert "Drawdown Analysis" in summary
        assert "Max drawdown" in summary
        assert "Recovery rate" in summary
        assert "underwater" in summary.lower()

    def test_summary_includes_top_drawdowns(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_equity())
        summary = result.summary()
        assert "Top 5" in summary

    def test_empty_summary(self):
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(_make_flat_equity())
        summary = result.summary()
        assert "0 episodes" in summary
