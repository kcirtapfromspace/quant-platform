"""Tests for execution schedule optimizer (QUA-111)."""
from __future__ import annotations

import numpy as np
import pytest

from quant.execution.schedule_optimizer import (
    CostBreakdown,
    FrontierPoint,
    ScheduleConfig,
    ScheduleOptimizer,
    ScheduleResult,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

TOTAL_SHARES = 100_000
DAILY_VOLUME = 1_000_000
VOLATILITY_BPS = 150.0
SPREAD_BPS = 5.0


def _optimize(**kwargs) -> ScheduleResult:
    """Helper: run optimiser with defaults overridden by kwargs."""
    cfg_args = {}
    opt_args = {
        "total_shares": TOTAL_SHARES,
        "daily_volume": DAILY_VOLUME,
        "volatility_bps": VOLATILITY_BPS,
        "spread_bps": SPREAD_BPS,
    }
    for k, v in kwargs.items():
        if k in ("n_periods", "risk_aversion", "temp_impact", "perm_impact"):
            cfg_args[k] = v
        else:
            opt_args[k] = v
    cfg = ScheduleConfig(**cfg_args)
    return ScheduleOptimizer(cfg).optimize(**opt_args)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_schedule_result(self):
        result = _optimize()
        assert isinstance(result, ScheduleResult)

    def test_trade_fractions_sum_to_one(self):
        result = _optimize()
        assert float(result.trade_fractions.sum()) == pytest.approx(1.0, abs=1e-10)

    def test_trade_shares_sum(self):
        result = _optimize()
        assert float(result.trade_shares.sum()) == pytest.approx(
            TOTAL_SHARES, rel=1e-6,
        )

    def test_n_periods(self):
        result = _optimize(n_periods=20)
        assert result.n_periods == 20
        assert len(result.trade_fractions) == 20

    def test_total_shares(self):
        result = _optimize()
        assert result.total_shares == TOTAL_SHARES

    def test_cost_breakdown_type(self):
        result = _optimize()
        assert isinstance(result.cost, CostBreakdown)

    def test_invalid_total_shares(self):
        with pytest.raises(ValueError, match="positive"):
            _optimize(total_shares=0)

    def test_invalid_daily_volume(self):
        with pytest.raises(ValueError, match="positive"):
            _optimize(daily_volume=-100)


# ---------------------------------------------------------------------------
# TWAP (zero risk aversion)
# ---------------------------------------------------------------------------


class TestTWAP:
    def test_uniform_fractions(self):
        """Zero risk aversion should produce equal fractions (TWAP)."""
        result = _optimize(risk_aversion=0.0, n_periods=10)
        expected = np.ones(10) / 10
        np.testing.assert_array_almost_equal(result.trade_fractions, expected)

    def test_uniform_shares(self):
        result = _optimize(risk_aversion=0.0, n_periods=5)
        expected = np.ones(5) * TOTAL_SHARES / 5
        np.testing.assert_array_almost_equal(result.trade_shares, expected)

    def test_remaining_linear(self):
        """TWAP remaining inventory should decrease linearly."""
        result = _optimize(risk_aversion=0.0, n_periods=5)
        expected = np.array([80000, 60000, 40000, 20000, 0])
        np.testing.assert_array_almost_equal(result.remaining, expected, decimal=0)


# ---------------------------------------------------------------------------
# Front-loaded (high risk aversion)
# ---------------------------------------------------------------------------


class TestFrontLoaded:
    def test_first_period_largest(self):
        """High risk aversion should front-load: first period trades most."""
        result = _optimize(risk_aversion=1.0, n_periods=10)
        assert result.trade_fractions[0] > result.trade_fractions[-1]

    def test_monotonically_decreasing(self):
        """Trade fractions should decrease over time."""
        result = _optimize(risk_aversion=1.0, n_periods=10)
        for i in range(len(result.trade_fractions) - 1):
            assert result.trade_fractions[i] >= result.trade_fractions[i + 1] - 1e-10

    def test_more_aggressive_than_twap(self):
        """First period should trade more than TWAP rate."""
        twap = _optimize(risk_aversion=0.0, n_periods=10)
        aggressive = _optimize(risk_aversion=1.0, n_periods=10)
        assert aggressive.trade_fractions[0] > twap.trade_fractions[0]


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


class TestCost:
    def test_costs_positive(self):
        result = _optimize()
        assert result.cost.total_bps > 0
        assert result.cost.temporary_bps >= 0
        assert result.cost.permanent_bps >= 0

    def test_total_is_sum(self):
        result = _optimize()
        expected = (
            result.cost.temporary_bps
            + result.cost.permanent_bps
            + result.cost.risk_bps
        )
        assert result.cost.total_bps == pytest.approx(expected, rel=1e-6)

    def test_larger_order_higher_cost(self):
        small = _optimize(total_shares=10_000)
        large = _optimize(total_shares=500_000)
        assert large.cost.total_bps > small.cost.total_bps

    def test_more_periods_lower_temp_impact(self):
        """Spreading over more periods should reduce temporary impact."""
        short = _optimize(n_periods=5, risk_aversion=0.0)
        long = _optimize(n_periods=20, risk_aversion=0.0)
        assert long.cost.temporary_bps < short.cost.temporary_bps


# ---------------------------------------------------------------------------
# Participation rate
# ---------------------------------------------------------------------------


class TestParticipation:
    def test_participation_reasonable(self):
        result = _optimize()
        assert 0 < result.participation < 1

    def test_participation_proportional(self):
        """Larger order relative to volume → higher participation."""
        small = _optimize(total_shares=10_000)
        large = _optimize(total_shares=500_000)
        assert large.participation > small.participation


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------


class TestFrontier:
    def test_returns_list(self):
        opt = ScheduleOptimizer()
        frontier = opt.efficient_frontier(
            TOTAL_SHARES, DAILY_VOLUME, VOLATILITY_BPS, n_points=10,
        )
        assert isinstance(frontier, list)
        assert len(frontier) == 10

    def test_frontier_point_type(self):
        opt = ScheduleOptimizer()
        frontier = opt.efficient_frontier(
            TOTAL_SHARES, DAILY_VOLUME, VOLATILITY_BPS, n_points=5,
        )
        for fp in frontier:
            assert isinstance(fp, FrontierPoint)

    def test_cost_increases_with_urgency(self):
        """Higher risk aversion → higher expected cost."""
        opt = ScheduleOptimizer()
        frontier = opt.efficient_frontier(
            TOTAL_SHARES, DAILY_VOLUME, VOLATILITY_BPS, n_points=10,
        )
        # First point has lowest lambda, last has highest
        # Expected cost should generally increase with urgency
        assert frontier[-1].expected_cost >= frontier[0].expected_cost

    def test_risk_decreases_with_urgency(self):
        """Higher risk aversion → lower execution risk."""
        opt = ScheduleOptimizer()
        frontier = opt.efficient_frontier(
            TOTAL_SHARES, DAILY_VOLUME, VOLATILITY_BPS, n_points=10,
        )
        assert frontier[-1].risk <= frontier[0].risk


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_period(self):
        result = _optimize(n_periods=1)
        assert len(result.trade_fractions) == 1
        assert result.trade_fractions[0] == pytest.approx(1.0)

    def test_very_small_order(self):
        result = _optimize(total_shares=1)
        assert result.cost.total_bps > 0

    def test_zero_volatility(self):
        result = _optimize(volatility_bps=0.0)
        # Should produce TWAP (no risk to penalise)
        expected = np.ones(10) / 10
        np.testing.assert_array_almost_equal(
            result.trade_fractions, expected, decimal=5,
        )

    def test_config_accessible(self):
        cfg = ScheduleConfig(n_periods=5)
        opt = ScheduleOptimizer(cfg)
        assert opt.config.n_periods == 5


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_contains_key_info(self):
        result = _optimize()
        summary = result.summary()
        assert "Optimal Execution Schedule" in summary
        assert "Temporary" in summary
        assert "Permanent" in summary
        assert "Total" in summary
        assert "Schedule" in summary
