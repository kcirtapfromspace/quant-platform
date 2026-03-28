"""Tests for turnover-optimized rebalancing (QUA-87)."""
from __future__ import annotations

import pytest

from quant.portfolio.turnover_optimizer import (
    DriftReport,
    OptimizedTrade,
    PriorityMethod,
    TurnoverConfig,
    TurnoverOptimizationResult,
    TurnoverOptimizer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CURRENT = {"AAPL": 0.30, "GOOG": 0.25, "MSFT": 0.25, "AMZN": 0.20}
TARGET = {"AAPL": 0.25, "GOOG": 0.35, "MSFT": 0.20, "AMZN": 0.20}


def _default_optimizer(**overrides) -> TurnoverOptimizer:
    cfg = TurnoverConfig(**overrides)
    return TurnoverOptimizer(cfg)


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------


class TestDrift:
    def test_drift_report_type(self):
        opt = _default_optimizer()
        report = opt.compute_drift(CURRENT, TARGET)
        assert isinstance(report, DriftReport)

    def test_drift_total(self):
        opt = _default_optimizer(no_trade_band=0.0)
        report = opt.compute_drift(CURRENT, TARGET)
        # |0.05| + |0.10| + |0.05| + |0.00| = 0.20
        assert abs(report.total_drift - 0.20) < 1e-8

    def test_max_drift_symbol(self):
        opt = _default_optimizer()
        report = opt.compute_drift(CURRENT, TARGET)
        assert report.max_drift_symbol == "GOOG"
        assert abs(report.max_drift - 0.10) < 1e-8

    def test_needs_rebalance_when_large_drift(self):
        opt = _default_optimizer(no_trade_band=0.02)
        report = opt.compute_drift(CURRENT, TARGET)
        assert report.needs_rebalance

    def test_no_rebalance_when_within_band(self):
        opt = _default_optimizer(no_trade_band=0.50)
        report = opt.compute_drift(CURRENT, TARGET)
        assert not report.needs_rebalance

    def test_n_drifted(self):
        opt = _default_optimizer(no_trade_band=0.06)
        report = opt.compute_drift(CURRENT, TARGET)
        # Only GOOG drifts 0.10 >= 0.06
        assert report.n_drifted == 1

    def test_drift_with_new_position(self):
        """Target adds a new symbol not in current."""
        opt = _default_optimizer(no_trade_band=0.0)
        report = opt.compute_drift(
            {"A": 1.0}, {"A": 0.6, "B": 0.4}
        )
        assert report.total_drift == pytest.approx(0.80)  # |0.4|+|0.4|

    def test_drift_with_exited_position(self):
        """Current has a position not in target (should sell entirely)."""
        opt = _default_optimizer(no_trade_band=0.0)
        report = opt.compute_drift(
            {"A": 0.5, "B": 0.5}, {"A": 1.0}
        )
        # drift_A = +0.5, drift_B = -0.5 => total = 1.0
        assert report.total_drift == pytest.approx(1.0)

    def test_zero_drift_when_aligned(self):
        opt = _default_optimizer()
        report = opt.compute_drift(CURRENT, CURRENT)
        assert report.total_drift == pytest.approx(0.0)
        assert not report.needs_rebalance


# ---------------------------------------------------------------------------
# Basic optimization
# ---------------------------------------------------------------------------


class TestBasicOptimization:
    def test_returns_result_type(self):
        opt = _default_optimizer()
        result = opt.optimize(CURRENT, TARGET)
        assert isinstance(result, TurnoverOptimizationResult)

    def test_trades_are_optimized_trades(self):
        opt = _default_optimizer()
        result = opt.optimize(CURRENT, TARGET)
        for t in result.trades:
            assert isinstance(t, OptimizedTrade)

    def test_naive_turnover_computed(self):
        opt = _default_optimizer()
        result = opt.optimize(CURRENT, TARGET)
        assert result.naive_turnover == pytest.approx(0.20)

    def test_optimized_weights_populated(self):
        opt = _default_optimizer()
        result = opt.optimize(CURRENT, TARGET)
        assert len(result.optimized_weights) > 0

    def test_turnover_saved_non_negative(self):
        opt = _default_optimizer()
        result = opt.optimize(CURRENT, TARGET)
        assert result.turnover_saved >= -1e-10

    def test_total_turnover_le_naive(self):
        opt = _default_optimizer()
        result = opt.optimize(CURRENT, TARGET)
        assert result.total_turnover <= result.naive_turnover + 1e-10


# ---------------------------------------------------------------------------
# No-trade band filtering
# ---------------------------------------------------------------------------


class TestNoTradeBand:
    def test_band_filters_small_drifts(self):
        """With band=0.06, only GOOG (drift=0.10) should trade."""
        opt = _default_optimizer(no_trade_band=0.06, max_turnover=None)
        result = opt.optimize(CURRENT, TARGET)
        executed = [t for t in result.trades if t.included]
        assert len(executed) == 1
        assert executed[0].symbol == "GOOG"

    def test_zero_band_trades_all(self):
        """Band of 0 should include all non-zero drifts."""
        opt = _default_optimizer(no_trade_band=0.0, max_turnover=None)
        result = opt.optimize(CURRENT, TARGET)
        # AAPL (-0.05), GOOG (+0.10), MSFT (-0.05), AMZN (0) => 3 trades
        executed = [t for t in result.trades if t.included]
        assert len(executed) == 3

    def test_large_band_skips_all(self):
        opt = _default_optimizer(no_trade_band=1.0)
        result = opt.optimize(CURRENT, TARGET)
        executed = [t for t in result.trades if t.included]
        assert len(executed) == 0
        assert result.total_turnover == pytest.approx(0.0)

    def test_turnover_saved_with_band(self):
        """Should save turnover when band filters positions."""
        opt = _default_optimizer(no_trade_band=0.06, max_turnover=None)
        result = opt.optimize(CURRENT, TARGET)
        assert result.turnover_saved > 0.05  # Saved at least AAPL + MSFT trades


# ---------------------------------------------------------------------------
# Turnover budget
# ---------------------------------------------------------------------------


class TestTurnoverBudget:
    def test_respects_max_turnover(self):
        opt = _default_optimizer(no_trade_band=0.0, max_turnover=0.08)
        result = opt.optimize(CURRENT, TARGET)
        assert result.total_turnover <= 0.08 + 1e-8

    def test_unlimited_turnover(self):
        opt = _default_optimizer(no_trade_band=0.0, max_turnover=None)
        result = opt.optimize(CURRENT, TARGET)
        assert result.total_turnover == pytest.approx(0.20)

    def test_budget_skips_low_priority_trades(self):
        """With tight budget, high-priority trades should be included first."""
        opt = _default_optimizer(
            no_trade_band=0.0,
            max_turnover=0.10,
            priority_method=PriorityMethod.LARGEST_DRIFT,
        )
        result = opt.optimize(CURRENT, TARGET)
        executed = [t for t in result.trades if t.included]
        # GOOG has largest drift (0.10), should be first
        assert any(t.symbol == "GOOG" for t in executed)

    def test_partial_trade_at_budget_boundary(self):
        """When budget runs out mid-trade, should scale the last trade."""
        opt = _default_optimizer(
            no_trade_band=0.0,
            max_turnover=0.12,
            priority_method=PriorityMethod.LARGEST_DRIFT,
        )
        result = opt.optimize(CURRENT, TARGET)
        # GOOG=0.10 fully included, 0.02 budget left for partial
        assert result.total_turnover == pytest.approx(0.12, abs=1e-6)


# ---------------------------------------------------------------------------
# Priority methods
# ---------------------------------------------------------------------------


class TestPriorityMethods:
    def test_largest_drift_prioritizes_goog(self):
        opt = _default_optimizer(
            no_trade_band=0.0,
            max_turnover=0.10,
            priority_method=PriorityMethod.LARGEST_DRIFT,
        )
        result = opt.optimize(CURRENT, TARGET)
        executed = [t for t in result.trades if t.included]
        # First executed should be GOOG (drift=0.10)
        top = max(executed, key=lambda t: t.priority_score)
        assert top.symbol == "GOOG"

    def test_alpha_cost_ratio_prioritizes_best_value(self):
        # AAPL: alpha=5, cost=10 => ratio=0.5
        # GOOG: alpha=30, cost=3 => ratio=10
        # MSFT: alpha=1, cost=8 => ratio=0.125
        alpha = {"AAPL": 5, "GOOG": 30, "MSFT": 1}
        cost = {"AAPL": 10, "GOOG": 3, "MSFT": 8}
        opt = _default_optimizer(
            no_trade_band=0.0,
            max_turnover=0.10,
            priority_method=PriorityMethod.ALPHA_COST_RATIO,
        )
        result = opt.optimize(CURRENT, TARGET, expected_alpha=alpha, cost_estimates=cost)
        executed = [t for t in result.trades if t.included]
        top = max(executed, key=lambda t: t.priority_score)
        assert top.symbol == "GOOG"

    def test_equal_priority_gives_same_score(self):
        opt = _default_optimizer(
            no_trade_band=0.0,
            max_turnover=None,
            priority_method=PriorityMethod.EQUAL,
        )
        result = opt.optimize(CURRENT, TARGET)
        scores = {t.priority_score for t in result.trades if t.included}
        assert len(scores) == 1
        assert 1.0 in scores

    def test_alpha_cost_ratio_without_data_falls_back(self):
        """When alpha/cost not provided, should still work (default scores)."""
        opt = _default_optimizer(
            no_trade_band=0.0,
            max_turnover=0.10,
            priority_method=PriorityMethod.ALPHA_COST_RATIO,
        )
        result = opt.optimize(CURRENT, TARGET)
        assert result.n_trades_executed >= 1


# ---------------------------------------------------------------------------
# Partial rebalancing
# ---------------------------------------------------------------------------


class TestPartialRebalance:
    def test_full_rebalance_closes_drift(self):
        opt = _default_optimizer(
            no_trade_band=0.0,
            max_turnover=None,
            partial_rebalance_frac=1.0,
        )
        result = opt.optimize(CURRENT, TARGET)
        for t in result.trades:
            if t.included:
                assert t.optimized_weight == pytest.approx(t.target_weight, abs=1e-8)

    def test_half_rebalance_moves_halfway(self):
        opt = _default_optimizer(
            no_trade_band=0.0,
            max_turnover=None,
            partial_rebalance_frac=0.5,
        )
        result = opt.optimize(CURRENT, TARGET)
        for t in result.trades:
            if t.included:
                expected = t.current_weight + (t.target_weight - t.current_weight) * 0.5
                assert t.optimized_weight == pytest.approx(expected, abs=1e-8)

    def test_partial_reduces_turnover(self):
        full = _default_optimizer(
            no_trade_band=0.0, max_turnover=None, partial_rebalance_frac=1.0,
        ).optimize(CURRENT, TARGET)
        half = _default_optimizer(
            no_trade_band=0.0, max_turnover=None, partial_rebalance_frac=0.5,
        ).optimize(CURRENT, TARGET)
        assert half.total_turnover < full.total_turnover

    def test_zero_partial_no_trades(self):
        opt = _default_optimizer(
            no_trade_band=0.0,
            max_turnover=None,
            partial_rebalance_frac=0.0,
        )
        result = opt.optimize(CURRENT, TARGET)
        # partial_drift = 0 for all => min_trade_weight filters them
        assert result.n_trades_executed == 0


# ---------------------------------------------------------------------------
# Rebalance efficiency
# ---------------------------------------------------------------------------


class TestEfficiency:
    def test_efficiency_positive_when_trading(self):
        opt = _default_optimizer(no_trade_band=0.0, max_turnover=None)
        result = opt.optimize(CURRENT, TARGET)
        assert result.rebalance_efficiency > 0

    def test_efficiency_higher_for_full_rebalance(self):
        """Full rebalance (no constraint) should have efficiency ~1.0."""
        opt = _default_optimizer(no_trade_band=0.0, max_turnover=None)
        result = opt.optimize(CURRENT, TARGET)
        assert result.rebalance_efficiency == pytest.approx(1.0, abs=0.01)

    def test_efficiency_zero_when_no_trades(self):
        opt = _default_optimizer(no_trade_band=1.0)
        result = opt.optimize(CURRENT, TARGET)
        assert result.rebalance_efficiency == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_portfolios(self):
        opt = _default_optimizer()
        result = opt.optimize({}, {})
        assert result.total_turnover == pytest.approx(0.0)
        assert result.n_trades_executed == 0

    def test_current_empty_target_populated(self):
        opt = _default_optimizer(no_trade_band=0.0, max_turnover=None)
        result = opt.optimize({}, {"A": 0.5, "B": 0.5})
        assert result.total_turnover == pytest.approx(1.0)
        assert result.n_trades_executed == 2

    def test_target_empty_sells_all(self):
        opt = _default_optimizer(no_trade_band=0.0, max_turnover=None)
        result = opt.optimize({"A": 0.5, "B": 0.5}, {})
        assert result.total_turnover == pytest.approx(1.0)

    def test_single_position(self):
        opt = _default_optimizer(no_trade_band=0.0, max_turnover=None)
        result = opt.optimize({"A": 1.0}, {"A": 0.8, "B": 0.2})
        assert result.n_trades_executed == 2

    def test_very_tight_budget(self):
        opt = _default_optimizer(no_trade_band=0.0, max_turnover=0.001)
        result = opt.optimize(CURRENT, TARGET)
        assert result.total_turnover <= 0.001 + 1e-8

    def test_identical_portfolios(self):
        opt = _default_optimizer()
        result = opt.optimize(CURRENT, CURRENT)
        assert result.n_trades_executed == 0
        assert result.total_turnover == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        opt = _default_optimizer(no_trade_band=0.0, max_turnover=None)
        result = opt.optimize(CURRENT, TARGET)
        summary = result.summary()
        assert "Turnover-Optimized" in summary
        assert "Naive turnover" in summary
        assert "Optimized turnover" in summary
        assert "Trade Schedule" in summary

    def test_summary_with_no_trades(self):
        opt = _default_optimizer(no_trade_band=1.0)
        result = opt.optimize(CURRENT, TARGET)
        summary = result.summary()
        assert "Trades executed" in summary
        assert "0" in summary
