"""Tests for strategy capacity estimation (QUA-85)."""
from __future__ import annotations

from quant.risk.capacity import (
    CapacityConfig,
    CapacityEstimator,
    CapacityPoint,
    CapacityResult,
)

# ---------------------------------------------------------------------------
# Default test inputs
# ---------------------------------------------------------------------------

DEFAULT_ALPHA = 50.0  # 50 bps annualised gross alpha
DEFAULT_TURNOVER = 0.30  # 30% one-way turnover per rebalance
DEFAULT_N_ASSETS = 20
DEFAULT_ADV = 50_000_000.0  # $50M avg daily volume per asset
DEFAULT_AUM = 100_000_000.0  # $100M current AUM


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        est = CapacityEstimator()
        result = est.estimate(
            DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS,
            DEFAULT_ADV, DEFAULT_AUM,
        )
        assert isinstance(result, CapacityResult)

    def test_capacity_aum_positive(self):
        est = CapacityEstimator()
        result = est.estimate(
            DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS,
            DEFAULT_ADV, DEFAULT_AUM,
        )
        assert result.capacity_aum > 0

    def test_current_net_alpha_positive(self):
        """At small AUM relative to liquidity, net alpha should be positive."""
        est = CapacityEstimator()
        # Use small AUM ($10M) to ensure we're below capacity
        result = est.estimate(
            DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS,
            DEFAULT_ADV, 10_000_000,
        )
        assert result.current_net_alpha_bps > 0

    def test_capacity_curve_populated(self):
        est = CapacityEstimator()
        result = est.estimate(
            DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS,
            DEFAULT_ADV, DEFAULT_AUM,
        )
        assert len(result.capacity_curve) > 0
        for pt in result.capacity_curve:
            assert isinstance(pt, CapacityPoint)


# ---------------------------------------------------------------------------
# Capacity logic
# ---------------------------------------------------------------------------


class TestCapacityLogic:
    def test_higher_alpha_more_capacity(self):
        est = CapacityEstimator()
        low = est.estimate(30.0, DEFAULT_TURNOVER, DEFAULT_N_ASSETS, DEFAULT_ADV, DEFAULT_AUM)
        high = est.estimate(100.0, DEFAULT_TURNOVER, DEFAULT_N_ASSETS, DEFAULT_ADV, DEFAULT_AUM)
        assert high.capacity_aum > low.capacity_aum

    def test_higher_turnover_less_capacity(self):
        est = CapacityEstimator()
        low_turn = est.estimate(DEFAULT_ALPHA, 0.10, DEFAULT_N_ASSETS, DEFAULT_ADV, DEFAULT_AUM)
        high_turn = est.estimate(DEFAULT_ALPHA, 0.50, DEFAULT_N_ASSETS, DEFAULT_ADV, DEFAULT_AUM)
        assert low_turn.capacity_aum > high_turn.capacity_aum

    def test_more_liquid_assets_more_capacity(self):
        est = CapacityEstimator()
        illiquid = est.estimate(DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS, 10_000_000, DEFAULT_AUM)
        liquid = est.estimate(DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS, 200_000_000, DEFAULT_AUM)
        assert liquid.capacity_aum > illiquid.capacity_aum

    def test_more_assets_more_capacity(self):
        est = CapacityEstimator()
        few = est.estimate(DEFAULT_ALPHA, DEFAULT_TURNOVER, 5, DEFAULT_ADV, DEFAULT_AUM)
        many = est.estimate(DEFAULT_ALPHA, DEFAULT_TURNOVER, 50, DEFAULT_ADV, DEFAULT_AUM)
        assert many.capacity_aum > few.capacity_aum

    def test_net_alpha_decreases_with_aum(self):
        """Net alpha should monotonically decrease as AUM increases."""
        est = CapacityEstimator()
        result = est.estimate(
            DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS,
            DEFAULT_ADV, DEFAULT_AUM,
        )
        prev_alpha = float("inf")
        for pt in result.capacity_curve:
            assert pt.net_alpha_bps <= prev_alpha + 1e-6
            prev_alpha = pt.net_alpha_bps


# ---------------------------------------------------------------------------
# Cost decomposition
# ---------------------------------------------------------------------------


class TestCostDecomposition:
    def test_costs_sum_to_total(self):
        est = CapacityEstimator()
        result = est.estimate(
            DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS,
            DEFAULT_ADV, DEFAULT_AUM,
        )
        for pt in result.capacity_curve:
            expected = pt.impact_cost_bps + pt.spread_cost_bps + pt.commission_cost_bps
            assert abs(pt.total_cost_bps - expected) < 1e-6

    def test_impact_increases_with_aum(self):
        est = CapacityEstimator()
        result = est.estimate(
            DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS,
            DEFAULT_ADV, DEFAULT_AUM,
        )
        prev_impact = 0.0
        for pt in result.capacity_curve:
            assert pt.impact_cost_bps >= prev_impact - 1e-6
            prev_impact = pt.impact_cost_bps

    def test_spread_constant_across_aum(self):
        """Spread cost (in bps) should be constant regardless of AUM."""
        est = CapacityEstimator()
        result = est.estimate(
            DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS,
            DEFAULT_ADV, DEFAULT_AUM,
        )
        spreads = [pt.spread_cost_bps for pt in result.capacity_curve]
        assert max(spreads) - min(spreads) < 1e-6


# ---------------------------------------------------------------------------
# Utilisation
# ---------------------------------------------------------------------------


class TestUtilisation:
    def test_utilisation_positive(self):
        est = CapacityEstimator()
        # Use small AUM to be under capacity
        result = est.estimate(
            DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS,
            DEFAULT_ADV, 10_000_000,
        )
        assert 0.0 < result.capacity_utilisation < 1.0

    def test_high_aum_high_utilisation(self):
        est = CapacityEstimator()
        # Use very high AUM relative to capacity
        result = est.estimate(
            20.0, 0.50, 5, 10_000_000, 500_000_000,
        )
        assert result.capacity_utilisation > 0.5


# ---------------------------------------------------------------------------
# Participation rate
# ---------------------------------------------------------------------------


class TestParticipationRate:
    def test_participation_positive(self):
        est = CapacityEstimator()
        result = est.estimate(
            DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS,
            DEFAULT_ADV, DEFAULT_AUM,
        )
        for pt in result.capacity_curve:
            assert pt.participation_rate >= 0

    def test_participation_increases_with_aum(self):
        est = CapacityEstimator()
        result = est.estimate(
            DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS,
            DEFAULT_ADV, DEFAULT_AUM,
        )
        prev = 0.0
        for pt in result.capacity_curve:
            assert pt.participation_rate >= prev - 1e-12
            prev = pt.participation_rate


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfig:
    def test_custom_impact_coefficient(self):
        """Higher impact coefficient should reduce capacity."""
        low_eta = CapacityEstimator(CapacityConfig(impact_coefficient=0.05))
        high_eta = CapacityEstimator(CapacityConfig(impact_coefficient=0.20))
        r_low = low_eta.estimate(DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS, DEFAULT_ADV, DEFAULT_AUM)
        r_high = high_eta.estimate(DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS, DEFAULT_ADV, DEFAULT_AUM)
        assert r_low.capacity_aum > r_high.capacity_aum

    def test_custom_vol(self):
        """Higher volatility should reduce capacity (more market impact)."""
        low_vol = CapacityEstimator(CapacityConfig(annualised_vol=0.10))
        high_vol = CapacityEstimator(CapacityConfig(annualised_vol=0.30))
        r_low = low_vol.estimate(DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS, DEFAULT_ADV, DEFAULT_AUM)
        r_high = high_vol.estimate(DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS, DEFAULT_ADV, DEFAULT_AUM)
        assert r_low.capacity_aum > r_high.capacity_aum


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_assets(self):
        est = CapacityEstimator()
        result = est.estimate(DEFAULT_ALPHA, DEFAULT_TURNOVER, 0, DEFAULT_ADV, DEFAULT_AUM)
        assert result.capacity_aum == 0.0

    def test_zero_volume(self):
        est = CapacityEstimator()
        result = est.estimate(DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS, 0, DEFAULT_AUM)
        assert result.capacity_aum == 0.0

    def test_zero_turnover(self):
        est = CapacityEstimator()
        result = est.estimate(DEFAULT_ALPHA, 0.0, DEFAULT_N_ASSETS, DEFAULT_ADV, DEFAULT_AUM)
        assert result.capacity_aum == 0.0

    def test_very_high_alpha(self):
        """Extremely high alpha should produce very high capacity."""
        est = CapacityEstimator()
        result = est.estimate(10_000, DEFAULT_TURNOVER, DEFAULT_N_ASSETS, DEFAULT_ADV, DEFAULT_AUM)
        assert result.capacity_aum > DEFAULT_AUM * 10


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        est = CapacityEstimator()
        result = est.estimate(
            DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS,
            DEFAULT_ADV, DEFAULT_AUM,
        )
        summary = result.summary()
        assert "Capacity" in summary
        assert "Gross alpha" in summary
        assert "Utilisation" in summary
        assert "net alpha" in summary.lower()

    def test_summary_includes_curve(self):
        est = CapacityEstimator()
        result = est.estimate(
            DEFAULT_ALPHA, DEFAULT_TURNOVER, DEFAULT_N_ASSETS,
            DEFAULT_ADV, DEFAULT_AUM,
        )
        summary = result.summary()
        assert "Capacity Curve" in summary
