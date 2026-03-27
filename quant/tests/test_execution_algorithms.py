"""Tests for execution algorithms — TWAP, VWAP, market impact (QUA-35)."""
from __future__ import annotations

import math

import pytest

from quant.execution.algorithms import (
    ExecutionSchedule,
    TWAPAlgorithm,
    VWAPAlgorithm,
    estimate_market_impact,
)

# ---------------------------------------------------------------------------
# TWAP tests
# ---------------------------------------------------------------------------


class TestTWAPAlgorithm:
    def test_basic_schedule(self):
        """TWAP should produce a valid schedule with expected slice count."""
        algo = TWAPAlgorithm(n_slices=5)
        schedule = algo.schedule("AAPL", "buy", 1000, 3600)

        assert isinstance(schedule, ExecutionSchedule)
        assert schedule.symbol == "AAPL"
        assert schedule.side == "buy"
        assert schedule.total_quantity == 1000
        assert schedule.algorithm == "twap"
        assert schedule.n_slices == 5
        assert len(schedule.slices) == 5

    def test_quantities_sum_to_total(self):
        """All slice quantities should sum to the parent order quantity."""
        algo = TWAPAlgorithm(n_slices=7)
        schedule = algo.schedule("GOOG", "sell", 5000, 7200)

        total = sum(s.quantity for s in schedule.slices)
        assert abs(total - 5000) < 0.01

    def test_equal_size_slices(self):
        """TWAP slices should be approximately equal in size."""
        algo = TWAPAlgorithm(n_slices=10)
        schedule = algo.schedule("MSFT", "buy", 10000, 3600)

        quantities = [s.quantity for s in schedule.slices]
        expected = 10000 / 10
        for q in quantities:
            assert abs(q - expected) < 0.01

    def test_pct_of_parent_sums_to_one(self):
        """Percentage of parent should sum to 1.0."""
        algo = TWAPAlgorithm(n_slices=5)
        schedule = algo.schedule("AAPL", "buy", 1000, 3600)

        pct_total = sum(s.pct_of_parent for s in schedule.slices)
        assert abs(pct_total - 1.0) < 0.001

    def test_scheduled_times_are_ordered(self):
        """Slices should be scheduled in ascending time order."""
        algo = TWAPAlgorithm(n_slices=10)
        schedule = algo.schedule("AAPL", "buy", 1000, 3600)

        times = [s.scheduled_seconds for s in schedule.slices]
        assert times == sorted(times)

    def test_first_slice_at_time_zero(self):
        """First slice should be at time 0."""
        algo = TWAPAlgorithm(n_slices=5)
        schedule = algo.schedule("AAPL", "buy", 1000, 3600)

        assert schedule.slices[0].scheduled_seconds == 0.0

    def test_even_time_intervals(self):
        """TWAP should have equal time intervals between slices."""
        algo = TWAPAlgorithm(n_slices=4)
        schedule = algo.schedule("AAPL", "buy", 1000, 3600)

        interval = 3600 / 4
        for i, s in enumerate(schedule.slices):
            assert abs(s.scheduled_seconds - i * interval) < 0.01

    def test_single_slice(self):
        """n_slices=1 should produce a single slice with full quantity."""
        algo = TWAPAlgorithm(n_slices=1)
        schedule = algo.schedule("AAPL", "buy", 500, 3600)

        assert schedule.n_slices == 1
        assert schedule.slices[0].quantity == 500

    def test_more_slices_than_shares(self):
        """If n_slices > quantity, cap at quantity (one share per slice)."""
        algo = TWAPAlgorithm(n_slices=100)
        schedule = algo.schedule("AAPL", "buy", 5, 3600)

        assert schedule.n_slices == 5

    def test_limit_offset_propagated(self):
        """Limit offset should appear on each slice."""
        algo = TWAPAlgorithm(n_slices=3, limit_offset_bps=5.0)
        schedule = algo.schedule("AAPL", "buy", 1000, 3600)

        for s in schedule.slices:
            assert s.limit_offset_bps == 5.0

    def test_name(self):
        assert TWAPAlgorithm().name == "twap"

    def test_invalid_n_slices_raises(self):
        with pytest.raises(ValueError, match="n_slices"):
            TWAPAlgorithm(n_slices=0)

    def test_zero_quantity_raises(self):
        algo = TWAPAlgorithm()
        with pytest.raises(ValueError, match="quantity"):
            algo.schedule("AAPL", "buy", 0, 3600)

    def test_zero_duration_raises(self):
        algo = TWAPAlgorithm()
        with pytest.raises(ValueError, match="duration"):
            algo.schedule("AAPL", "buy", 1000, 0)

    def test_schedule_properties(self):
        """avg_slice_quantity and slice_interval_seconds should be correct."""
        algo = TWAPAlgorithm(n_slices=5)
        schedule = algo.schedule("AAPL", "buy", 1000, 3600)

        assert abs(schedule.avg_slice_quantity - 200.0) < 0.01
        assert abs(schedule.slice_interval_seconds - 720.0) < 0.01


# ---------------------------------------------------------------------------
# VWAP tests
# ---------------------------------------------------------------------------


class TestVWAPAlgorithm:
    def test_basic_schedule(self):
        """VWAP should produce a valid schedule."""
        algo = VWAPAlgorithm(n_slices=5)
        schedule = algo.schedule("AAPL", "buy", 1000, 3600)

        assert isinstance(schedule, ExecutionSchedule)
        assert schedule.algorithm == "vwap"
        assert schedule.n_slices == 5

    def test_quantities_sum_to_total(self):
        """VWAP slice quantities should sum to parent order."""
        algo = VWAPAlgorithm(n_slices=10)
        schedule = algo.schedule("AAPL", "sell", 5000, 7200)

        total = sum(s.quantity for s in schedule.slices)
        assert abs(total - 5000) < 0.01

    def test_pct_of_parent_sums_to_one(self):
        """Percentage of parent should sum to 1.0."""
        algo = VWAPAlgorithm(n_slices=5)
        schedule = algo.schedule("AAPL", "buy", 1000, 3600)

        pct_total = sum(s.pct_of_parent for s in schedule.slices)
        assert abs(pct_total - 1.0) < 0.01

    def test_custom_volume_profile(self):
        """Custom volume profile should shape slice sizes."""
        # Heavily front-loaded profile
        profile = [0.5, 0.3, 0.2]
        algo = VWAPAlgorithm(n_slices=3, volume_profile=profile)
        schedule = algo.schedule("AAPL", "buy", 1000, 3600)

        assert schedule.n_slices == 3
        # First slice should be largest
        assert schedule.slices[0].quantity > schedule.slices[2].quantity

    def test_default_profile_u_shape(self):
        """Default profile should have higher volume at open and close."""
        algo = VWAPAlgorithm(n_slices=13)  # matches default profile length
        schedule = algo.schedule("AAPL", "buy", 13000, 23400)  # 6.5 hours

        # First and last slices should be larger than middle
        first = schedule.slices[0].quantity
        middle = schedule.slices[6].quantity
        last = schedule.slices[-1].quantity
        assert first > middle
        assert last > middle

    def test_n_slices_capped_at_profile_length(self):
        """If n_slices > len(profile), cap at profile length."""
        profile = [0.5, 0.5]
        algo = VWAPAlgorithm(n_slices=10, volume_profile=profile)
        schedule = algo.schedule("AAPL", "buy", 1000, 3600)

        assert schedule.n_slices == 2

    def test_uniform_profile_equals_twap(self):
        """With a uniform volume profile, VWAP should behave like TWAP."""
        profile = [1.0, 1.0, 1.0, 1.0, 1.0]
        algo = VWAPAlgorithm(n_slices=5, volume_profile=profile)
        schedule = algo.schedule("AAPL", "buy", 5000, 3600)

        expected = 5000 / 5
        for s in schedule.slices:
            assert abs(s.quantity - expected) < 0.01

    def test_name(self):
        assert VWAPAlgorithm().name == "vwap"

    def test_zero_profile_degrades_to_uniform(self):
        """All-zero profile should degrade to equal-weight (TWAP)."""
        profile = [0.0, 0.0, 0.0]
        algo = VWAPAlgorithm(n_slices=3, volume_profile=profile)
        schedule = algo.schedule("AAPL", "buy", 3000, 3600)

        quantities = [s.quantity for s in schedule.slices]
        assert abs(quantities[0] - quantities[1]) < 0.01


# ---------------------------------------------------------------------------
# Market impact estimation tests
# ---------------------------------------------------------------------------


class TestMarketImpactEstimation:
    def test_basic_impact(self):
        """Impact should be positive for valid inputs."""
        impact = estimate_market_impact(
            quantity=10_000,
            daily_volume=1_000_000,
            volatility=0.25,
        )
        assert impact > 0

    def test_larger_order_more_impact(self):
        """Larger orders should have higher impact."""
        small = estimate_market_impact(1_000, 1_000_000, 0.25)
        large = estimate_market_impact(100_000, 1_000_000, 0.25)
        assert large > small

    def test_higher_volatility_more_impact(self):
        """Higher volatility should increase impact."""
        low_vol = estimate_market_impact(10_000, 1_000_000, 0.10)
        high_vol = estimate_market_impact(10_000, 1_000_000, 0.50)
        assert high_vol > low_vol

    def test_higher_volume_less_impact(self):
        """Higher daily volume should reduce impact."""
        illiquid = estimate_market_impact(10_000, 100_000, 0.25)
        liquid = estimate_market_impact(10_000, 10_000_000, 0.25)
        assert illiquid > liquid

    def test_missing_volume_returns_zero(self):
        """Impact should be 0 when volume is missing."""
        assert estimate_market_impact(10_000, None, 0.25) == 0.0

    def test_missing_volatility_returns_zero(self):
        """Impact should be 0 when volatility is missing."""
        assert estimate_market_impact(10_000, 1_000_000, None) == 0.0

    def test_zero_quantity_returns_zero(self):
        assert estimate_market_impact(0, 1_000_000, 0.25) == 0.0

    def test_zero_volume_returns_zero(self):
        assert estimate_market_impact(10_000, 0, 0.25) == 0.0

    def test_custom_eta(self):
        """Higher eta should increase impact proportionally."""
        low_eta = estimate_market_impact(10_000, 1_000_000, 0.25, eta=0.25)
        high_eta = estimate_market_impact(10_000, 1_000_000, 0.25, eta=1.0)
        assert abs(high_eta / low_eta - 4.0) < 0.01

    def test_impact_in_schedule(self):
        """Schedule should include impact estimate when vol/volume provided."""
        algo = TWAPAlgorithm(n_slices=5)
        schedule = algo.schedule(
            "AAPL", "buy", 50_000, 3600,
            daily_volume=5_000_000,
            volatility=0.30,
        )
        assert schedule.estimated_impact_bps > 0

    def test_no_impact_without_data(self):
        """Schedule should have zero impact without vol/volume data."""
        algo = TWAPAlgorithm(n_slices=5)
        schedule = algo.schedule("AAPL", "buy", 50_000, 3600)
        assert schedule.estimated_impact_bps == 0.0

    def test_square_root_scaling(self):
        """Impact should scale as sqrt(participation rate)."""
        # Double the order size: impact should increase by sqrt(2)
        base = estimate_market_impact(10_000, 1_000_000, 0.25)
        doubled = estimate_market_impact(20_000, 1_000_000, 0.25)
        ratio = doubled / base
        assert abs(ratio - math.sqrt(2)) < 0.01


# ---------------------------------------------------------------------------
# Cross-algorithm tests
# ---------------------------------------------------------------------------


class TestCrossAlgorithm:
    def test_both_algos_produce_same_total(self):
        """TWAP and VWAP should both produce the requested total quantity."""
        for algo in [TWAPAlgorithm(n_slices=5), VWAPAlgorithm(n_slices=5)]:
            schedule = algo.schedule("AAPL", "buy", 2000, 3600)
            total = sum(s.quantity for s in schedule.slices)
            assert abs(total - 2000) < 0.01, f"{algo.name}: total={total}"

    def test_slice_sequences_are_monotonic(self):
        """Slice sequences should go 0, 1, 2, ..."""
        for algo in [TWAPAlgorithm(n_slices=5), VWAPAlgorithm(n_slices=5)]:
            schedule = algo.schedule("AAPL", "buy", 2000, 3600)
            for i, s in enumerate(schedule.slices):
                assert s.sequence == i

    def test_all_quantities_positive(self):
        """Every slice should have positive quantity."""
        for algo in [TWAPAlgorithm(n_slices=5), VWAPAlgorithm(n_slices=5)]:
            schedule = algo.schedule("AAPL", "buy", 2000, 3600)
            for s in schedule.slices:
                assert s.quantity > 0, f"{algo.name}: slice {s.sequence} has qty={s.quantity}"
