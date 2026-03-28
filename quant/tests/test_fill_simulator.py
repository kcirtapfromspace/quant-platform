"""Tests for execution fill simulator (QUA-102)."""
from __future__ import annotations

import pandas as pd
import pytest

from quant.execution.fill_simulator import (
    CapacityEstimate,
    ExecutionSimulator,
    ExecutionSummary,
    FillModel,
    OrderFill,
    SimulatorConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOLS = ["AAPL", "GOOG", "MSFT", "AMZN", "META"]


def _prices() -> pd.Series:
    return pd.Series(
        [150.0, 140.0, 380.0, 180.0, 500.0], index=SYMBOLS,
    )


def _volumes() -> pd.Series:
    """ADV in USD."""
    return pd.Series(
        [5e9, 3e9, 2e9, 4e9, 1e9], index=SYMBOLS,
    )


def _volatilities() -> pd.Series:
    return pd.Series(
        [0.25, 0.28, 0.22, 0.30, 0.35], index=SYMBOLS,
    )


def _target_weights() -> dict[str, float]:
    return {"AAPL": 0.25, "GOOG": 0.20, "MSFT": 0.30, "AMZN": 0.15, "META": 0.10}


def _current_weights() -> dict[str, float]:
    return {"AAPL": 0.20, "GOOG": 0.25, "MSFT": 0.25, "AMZN": 0.20, "META": 0.10}


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_summary(self):
        sim = ExecutionSimulator()
        result = sim.simulate_rebalance(_target_weights())
        assert isinstance(result, ExecutionSummary)

    def test_n_orders(self):
        sim = ExecutionSimulator()
        result = sim.simulate_rebalance(_target_weights())
        assert result.n_orders == 5

    def test_fills_populated(self):
        sim = ExecutionSimulator()
        result = sim.simulate_rebalance(_target_weights())
        assert len(result.fills) == result.n_orders

    def test_fill_types(self):
        sim = ExecutionSimulator()
        result = sim.simulate_rebalance(_target_weights())
        for f in result.fills:
            assert isinstance(f, OrderFill)

    def test_all_buys_from_zero(self):
        sim = ExecutionSimulator()
        result = sim.simulate_rebalance(_target_weights())
        for f in result.fills:
            assert f.side == "buy"

    def test_no_orders_when_on_target(self):
        sim = ExecutionSimulator()
        w = _target_weights()
        result = sim.simulate_rebalance(w, current_weights=w)
        assert result.n_orders == 0

    def test_config_accessible(self):
        cfg = SimulatorConfig(spread_bps=10.0)
        sim = ExecutionSimulator(cfg)
        assert sim.config.spread_bps == 10.0


# ---------------------------------------------------------------------------
# Instant fill model
# ---------------------------------------------------------------------------


class TestInstantFill:
    def test_full_fill(self):
        cfg = SimulatorConfig(fill_model=FillModel.INSTANT)
        result = ExecutionSimulator(cfg).simulate_rebalance(_target_weights())
        for f in result.fills:
            assert f.fill_fraction == pytest.approx(1.0)

    def test_no_impact_cost(self):
        cfg = SimulatorConfig(fill_model=FillModel.INSTANT)
        result = ExecutionSimulator(cfg).simulate_rebalance(_target_weights())
        for f in result.fills:
            assert f.impact_cost_bps == 0.0

    def test_spread_cost_applied(self):
        cfg = SimulatorConfig(fill_model=FillModel.INSTANT, spread_bps=10.0)
        result = ExecutionSimulator(cfg).simulate_rebalance(_target_weights())
        for f in result.fills:
            assert f.spread_cost_bps == 10.0

    def test_buy_price_above_arrival(self):
        cfg = SimulatorConfig(fill_model=FillModel.INSTANT, spread_bps=10.0)
        result = ExecutionSimulator(cfg).simulate_rebalance(
            _target_weights(), prices=_prices(),
        )
        for f in result.fills:
            assert f.fill_price > f.arrival_price

    def test_zero_participation(self):
        cfg = SimulatorConfig(fill_model=FillModel.INSTANT)
        result = ExecutionSimulator(cfg).simulate_rebalance(_target_weights())
        for f in result.fills:
            assert f.participation_rate == 0.0


# ---------------------------------------------------------------------------
# Participation fill model
# ---------------------------------------------------------------------------


class TestParticipationFill:
    def test_full_fill_liquid(self):
        """Liquid stocks should fill completely with reasonable AUM."""
        cfg = SimulatorConfig(
            fill_model=FillModel.PARTICIPATION,
            max_participation=0.10,
            aum=100_000_000,
        )
        result = ExecutionSimulator(cfg).simulate_rebalance(
            _target_weights(), volumes=_volumes(),
        )
        for f in result.fills:
            assert f.fill_fraction == pytest.approx(1.0)

    def test_partial_fill_illiquid(self):
        """Low-volume stock should have partial fill."""
        cfg = SimulatorConfig(
            fill_model=FillModel.PARTICIPATION,
            max_participation=0.01,
            aum=1_000_000_000,
        )
        volumes = pd.Series([1e6], index=["TINY"])
        result = ExecutionSimulator(cfg).simulate_rebalance(
            {"TINY": 0.50}, volumes=volumes,
        )
        assert result.fills[0].fill_fraction < 1.0

    def test_participation_rate_bounded(self):
        cfg = SimulatorConfig(
            fill_model=FillModel.PARTICIPATION,
            max_participation=0.05,
        )
        result = ExecutionSimulator(cfg).simulate_rebalance(
            _target_weights(), volumes=_volumes(),
        )
        for f in result.fills:
            assert f.participation_rate <= 0.05 + 1e-9

    def test_no_impact_modelling(self):
        """Participation model does not compute market impact."""
        cfg = SimulatorConfig(fill_model=FillModel.PARTICIPATION)
        result = ExecutionSimulator(cfg).simulate_rebalance(
            _target_weights(), volumes=_volumes(),
        )
        for f in result.fills:
            assert f.impact_cost_bps == 0.0


# ---------------------------------------------------------------------------
# Market impact fill model
# ---------------------------------------------------------------------------


class TestMarketImpactFill:
    def test_impact_positive(self):
        cfg = SimulatorConfig(fill_model=FillModel.MARKET_IMPACT)
        result = ExecutionSimulator(cfg).simulate_rebalance(
            _target_weights(), volumes=_volumes(), volatilities=_volatilities(),
        )
        # At least some orders should have positive impact
        impact_fills = [f for f in result.fills if f.impact_cost_bps > 0]
        assert len(impact_fills) > 0

    def test_higher_vol_more_impact(self):
        """Higher volatility should cause more market impact."""
        cfg = SimulatorConfig(fill_model=FillModel.MARKET_IMPACT, aum=100_000_000)

        low_vol = pd.Series([0.10], index=["X"])
        high_vol = pd.Series([0.50], index=["X"])
        volumes = pd.Series([1e9], index=["X"])

        r_low = ExecutionSimulator(cfg).simulate_rebalance(
            {"X": 0.50}, volumes=volumes, volatilities=low_vol,
        )
        r_high = ExecutionSimulator(cfg).simulate_rebalance(
            {"X": 0.50}, volumes=volumes, volatilities=high_vol,
        )
        assert r_high.fills[0].impact_cost_bps > r_low.fills[0].impact_cost_bps

    def test_higher_participation_more_impact(self):
        """Trading more of ADV should cause more impact."""
        cfg = SimulatorConfig(
            fill_model=FillModel.MARKET_IMPACT, max_participation=1.0,
        )

        vols = pd.Series([0.25], index=["X"])
        big_adv = pd.Series([1e10], index=["X"])
        small_adv = pd.Series([1e7], index=["X"])

        r_small_part = ExecutionSimulator(cfg).simulate_rebalance(
            {"X": 0.10}, volumes=big_adv, volatilities=vols,
        )
        r_large_part = ExecutionSimulator(cfg).simulate_rebalance(
            {"X": 0.10}, volumes=small_adv, volatilities=vols,
        )
        assert r_large_part.fills[0].impact_cost_bps > r_small_part.fills[0].impact_cost_bps

    def test_impact_plus_spread(self):
        """Total cost should include both impact and spread."""
        cfg = SimulatorConfig(
            fill_model=FillModel.MARKET_IMPACT, spread_bps=5.0,
        )
        result = ExecutionSimulator(cfg).simulate_rebalance(
            _target_weights(), volumes=_volumes(), volatilities=_volatilities(),
        )
        for f in result.fills:
            assert f.total_cost_bps >= f.spread_cost_bps + f.impact_cost_bps - 1e-6

    def test_volume_limited(self):
        """Large orders in illiquid names should be partially filled."""
        cfg = SimulatorConfig(
            fill_model=FillModel.MARKET_IMPACT,
            max_participation=0.01,
            aum=10_000_000_000,
        )
        volumes = pd.Series([1e7], index=["ILLIQ"])
        result = ExecutionSimulator(cfg).simulate_rebalance(
            {"ILLIQ": 0.50}, volumes=volumes, volatilities=pd.Series([0.30], index=["ILLIQ"]),
        )
        assert result.fills[0].fill_fraction < 1.0
        assert result.n_partially_filled == 1


# ---------------------------------------------------------------------------
# Cost decomposition
# ---------------------------------------------------------------------------


class TestCostDecomposition:
    def test_fill_total_equals_components(self):
        result = ExecutionSimulator().simulate_rebalance(
            _target_weights(), volumes=_volumes(), volatilities=_volatilities(),
        )
        for f in result.fills:
            expected = f.impact_cost_bps + f.spread_cost_bps + f.commission_cost_bps
            assert f.total_cost_bps == pytest.approx(expected, abs=1e-6)

    def test_summary_cost_decomposition(self):
        result = ExecutionSimulator().simulate_rebalance(
            _target_weights(), volumes=_volumes(), volatilities=_volatilities(),
        )
        expected = result.impact_cost_bps + result.spread_cost_bps + result.commission_cost_bps
        assert result.total_cost_bps == pytest.approx(expected, abs=1e-6)

    def test_zero_spread_no_spread_cost(self):
        cfg = SimulatorConfig(spread_bps=0.0, commission_bps=0.0)
        result = ExecutionSimulator(cfg).simulate_rebalance(
            _target_weights(), volumes=_volumes(), volatilities=_volatilities(),
        )
        assert result.spread_cost_bps == pytest.approx(0.0)
        assert result.commission_cost_bps == pytest.approx(0.0)

    def test_positive_total_cost(self):
        result = ExecutionSimulator().simulate_rebalance(
            _target_weights(), volumes=_volumes(), volatilities=_volatilities(),
        )
        assert result.total_cost_bps > 0


# ---------------------------------------------------------------------------
# Current weights / rebalance
# ---------------------------------------------------------------------------


class TestCurrentWeights:
    def test_smaller_trade_less_notional(self):
        """Trading from existing position should have less notional."""
        sim = ExecutionSimulator(SimulatorConfig(fill_model=FillModel.MARKET_IMPACT))
        r_zero = sim.simulate_rebalance(
            _target_weights(), volumes=_volumes(), volatilities=_volatilities(),
        )
        r_existing = sim.simulate_rebalance(
            _target_weights(), current_weights=_current_weights(),
            volumes=_volumes(), volatilities=_volatilities(),
        )
        assert r_existing.total_notional < r_zero.total_notional

    def test_buy_and_sell_sides(self):
        """Rebalance from existing should have both buys and sells."""
        sim = ExecutionSimulator()
        result = sim.simulate_rebalance(
            _target_weights(), current_weights=_current_weights(),
        )
        sides = {f.side for f in result.fills}
        assert "buy" in sides
        assert "sell" in sides

    def test_sell_price_below_arrival(self):
        """Selling should fill below arrival price (adverse slippage)."""
        cfg = SimulatorConfig(fill_model=FillModel.INSTANT, spread_bps=10.0)
        result = ExecutionSimulator(cfg).simulate_rebalance(
            _target_weights(), current_weights=_current_weights(),
            prices=_prices(),
        )
        sells = [f for f in result.fills if f.side == "sell"]
        for f in sells:
            assert f.fill_price < f.arrival_price

    def test_exit_position(self):
        """Exiting a position should generate a sell order."""
        sim = ExecutionSimulator()
        result = sim.simulate_rebalance(
            {"AAPL": 1.0}, current_weights={"AAPL": 0.5, "GOOG": 0.5},
        )
        sells = [f for f in result.fills if f.side == "sell"]
        assert len(sells) == 1
        assert sells[0].symbol == "GOOG"


# ---------------------------------------------------------------------------
# Implementation shortfall
# ---------------------------------------------------------------------------


class TestImplementationShortfall:
    def test_is_positive(self):
        result = ExecutionSimulator().simulate_rebalance(
            _target_weights(), volumes=_volumes(), volatilities=_volatilities(),
        )
        assert result.implementation_shortfall_bps > 0

    def test_is_at_least_total_cost(self):
        """IS should be >= total cost (it includes opportunity cost)."""
        result = ExecutionSimulator().simulate_rebalance(
            _target_weights(), volumes=_volumes(), volatilities=_volatilities(),
        )
        assert result.implementation_shortfall_bps >= result.total_cost_bps - 1e-9

    def test_is_equals_cost_when_fully_filled(self):
        """When all orders fill, IS = total cost (no opportunity cost)."""
        cfg = SimulatorConfig(fill_model=FillModel.INSTANT)
        result = ExecutionSimulator(cfg).simulate_rebalance(_target_weights())
        assert result.implementation_shortfall_bps == pytest.approx(
            result.total_cost_bps, abs=1e-9,
        )


# ---------------------------------------------------------------------------
# Capacity estimation
# ---------------------------------------------------------------------------


class TestCapacityEstimation:
    def test_returns_estimate(self):
        sim = ExecutionSimulator()
        estimate = sim.estimate_capacity(
            _target_weights(), _volumes(), _volatilities(),
        )
        assert isinstance(estimate, CapacityEstimate)

    def test_cost_curve_increasing(self):
        """Execution cost should generally increase with AUM."""
        sim = ExecutionSimulator()
        estimate = sim.estimate_capacity(
            _target_weights(), _volumes(), _volatilities(), n_points=8,
        )
        # Cost should be non-decreasing (modulo small float issues)
        for i in range(1, len(estimate.cost_curve_bps)):
            assert estimate.cost_curve_bps[i] >= estimate.cost_curve_bps[i - 1] - 1e-6

    def test_max_aum_positive(self):
        sim = ExecutionSimulator()
        estimate = sim.estimate_capacity(
            _target_weights(), _volumes(), _volatilities(),
        )
        assert estimate.max_aum > 0

    def test_breakeven_above_max(self):
        """Breakeven AUM (cost = alpha) should be >= max AUM (cost = 25% alpha)."""
        sim = ExecutionSimulator()
        estimate = sim.estimate_capacity(
            _target_weights(), _volumes(), _volatilities(),
        )
        assert estimate.breakeven_aum >= estimate.max_aum - 1e-6

    def test_fill_curve_populated(self):
        sim = ExecutionSimulator()
        estimate = sim.estimate_capacity(
            _target_weights(), _volumes(), _volatilities(), n_points=5,
        )
        assert len(estimate.fill_curve) == 5
        for fill in estimate.fill_curve:
            assert 0 <= fill <= 1.0 + 1e-9

    def test_low_alpha_lower_capacity(self):
        """Lower expected alpha → lower capacity threshold."""
        sim = ExecutionSimulator()
        cap_high = sim.estimate_capacity(
            _target_weights(), _volumes(), _volatilities(),
            expected_alpha_bps=100.0,
        )
        cap_low = sim.estimate_capacity(
            _target_weights(), _volumes(), _volatilities(),
            expected_alpha_bps=10.0,
        )
        assert cap_low.max_aum <= cap_high.max_aum


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_target(self):
        sim = ExecutionSimulator()
        result = sim.simulate_rebalance({})
        assert result.n_orders == 0
        assert result.total_cost_bps == 0.0

    def test_single_asset(self):
        sim = ExecutionSimulator()
        result = sim.simulate_rebalance({"X": 1.0})
        assert result.n_orders == 1
        assert result.fills[0].symbol == "X"

    def test_zero_weight_delta_no_trade(self):
        sim = ExecutionSimulator()
        result = sim.simulate_rebalance(
            {"X": 0.50}, current_weights={"X": 0.50},
        )
        assert result.n_orders == 0

    def test_very_small_trade(self):
        """Tiny weight changes should still generate a fill."""
        sim = ExecutionSimulator()
        result = sim.simulate_rebalance({"X": 1e-6})
        assert result.n_orders == 1
        assert result.fills[0].target_notional > 0

    def test_default_market_data(self):
        """Should work without any market data (uses defaults)."""
        sim = ExecutionSimulator()
        result = sim.simulate_rebalance(_target_weights())
        assert result.n_orders == 5
        assert result.total_cost_bps > 0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_execution_summary_readable(self):
        result = ExecutionSimulator().simulate_rebalance(
            _target_weights(), volumes=_volumes(), volatilities=_volatilities(),
        )
        summary = result.summary()
        assert "Execution Summary" in summary
        assert "orders" in summary.lower()
        assert "cost" in summary.lower()
        assert "impact" in summary.lower() or "Impact" in summary

    def test_capacity_summary_readable(self):
        sim = ExecutionSimulator()
        estimate = sim.estimate_capacity(
            _target_weights(), _volumes(), _volatilities(),
        )
        summary = estimate.summary()
        assert "Capacity Estimate" in summary
        assert "alpha" in summary.lower()
        assert "AUM" in summary

    def test_empty_summary_no_crash(self):
        result = ExecutionSimulator().simulate_rebalance({})
        summary = result.summary()
        assert "0 orders" in summary
