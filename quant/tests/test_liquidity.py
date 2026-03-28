"""Tests for liquidity risk monitoring (QUA-88)."""
from __future__ import annotations

import pytest

from quant.risk.liquidity import (
    LiquidityConfig,
    LiquidityMonitor,
    LiquidityResult,
    PositionLiquidity,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

POSITIONS = {"AAPL": 5_000_000, "GOOG": 3_000_000, "ILLIQ": 2_000_000}
ADV = {"AAPL": 100_000_000, "GOOG": 80_000_000, "ILLIQ": 1_000_000}
VOL = {"AAPL": 0.25, "GOOG": 0.30, "ILLIQ": 0.45}


def _monitor(**overrides) -> LiquidityMonitor:
    return LiquidityMonitor(LiquidityConfig(**overrides))


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        assert isinstance(result, LiquidityResult)

    def test_positions_populated(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        assert len(result.positions) == 3

    def test_position_types(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        for p in result.positions:
            assert isinstance(p, PositionLiquidity)

    def test_portfolio_value_correct(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        assert result.portfolio_value == pytest.approx(10_000_000)


# ---------------------------------------------------------------------------
# Days to liquidate
# ---------------------------------------------------------------------------


class TestDaysToLiquidate:
    def test_dtl_positive(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        for p in result.positions:
            assert p.days_to_liquidate > 0

    def test_liquid_position_low_dtl(self):
        """AAPL: $5M / ($100M * 0.10) = 0.5 days."""
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        aapl = next(p for p in result.positions if p.symbol == "AAPL")
        assert aapl.days_to_liquidate == pytest.approx(0.5)

    def test_illiquid_position_high_dtl(self):
        """ILLIQ: $2M / ($1M * 0.10) = 20 days."""
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        illiq = next(p for p in result.positions if p.symbol == "ILLIQ")
        assert illiq.days_to_liquidate == pytest.approx(20.0)

    def test_stressed_dtl_higher(self):
        result = _monitor(stress_volume_haircut=0.50).analyze(POSITIONS, ADV, VOL)
        for p in result.positions:
            assert p.days_to_liquidate_stressed >= p.days_to_liquidate - 1e-10

    def test_stressed_dtl_doubles_with_50pct_haircut(self):
        result = _monitor(stress_volume_haircut=0.50).analyze(POSITIONS, ADV, VOL)
        for p in result.positions:
            assert p.days_to_liquidate_stressed == pytest.approx(
                p.days_to_liquidate * 2.0
            )

    def test_portfolio_dtl_is_weighted_average(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        expected = sum(
            p.days_to_liquidate * p.position_value / result.portfolio_value
            for p in result.positions
        )
        assert result.portfolio_dtl == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Market impact / LaR
# ---------------------------------------------------------------------------


class TestLiquidityAtRisk:
    def test_lar_positive(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        assert result.portfolio_lar_bps > 0
        assert result.portfolio_lar_dollars > 0

    def test_stressed_lar_higher(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        assert result.portfolio_lar_stressed_bps >= result.portfolio_lar_bps - 1e-10

    def test_impact_increases_with_position_size(self):
        small = _monitor().analyze({"A": 1_000_000}, {"A": 100_000_000}, {"A": 0.20})
        large = _monitor().analyze({"A": 50_000_000}, {"A": 100_000_000}, {"A": 0.20})
        a_small = small.positions[0].impact_cost_bps
        a_large = large.positions[0].impact_cost_bps
        assert a_large > a_small

    def test_impact_increases_with_volatility(self):
        low_vol = _monitor().analyze({"A": 5_000_000}, {"A": 50_000_000}, {"A": 0.10})
        high_vol = _monitor().analyze({"A": 5_000_000}, {"A": 50_000_000}, {"A": 0.40})
        assert high_vol.positions[0].impact_cost_bps > low_vol.positions[0].impact_cost_bps

    def test_lar_dollars_consistent_with_bps(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        expected_dollars = result.portfolio_lar_bps / 10_000 * result.portfolio_value
        assert result.portfolio_lar_dollars == pytest.approx(expected_dollars, rel=1e-6)


# ---------------------------------------------------------------------------
# Liquidity score
# ---------------------------------------------------------------------------


class TestLiquidityScore:
    def test_score_between_0_and_100(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        for p in result.positions:
            assert 0 <= p.liquidity_score <= 100

    def test_liquid_position_high_score(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        aapl = next(p for p in result.positions if p.symbol == "AAPL")
        assert aapl.liquidity_score > 90  # DTL=0.5 days

    def test_illiquid_position_low_score(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        illiq = next(p for p in result.positions if p.symbol == "ILLIQ")
        assert illiq.liquidity_score == 0  # DTL=20, cap=20 => score=0

    def test_portfolio_score_weighted(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        expected = sum(
            p.liquidity_score * p.position_value / result.portfolio_value
            for p in result.positions
        )
        assert result.portfolio_liquidity_score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Warnings and concentration
# ---------------------------------------------------------------------------


class TestWarnings:
    def test_illiquid_flagged(self):
        result = _monitor(dtl_warning_days=5.0).analyze(POSITIONS, ADV, VOL)
        illiq = next(p for p in result.positions if p.symbol == "ILLIQ")
        assert illiq.is_warning

    def test_liquid_not_flagged(self):
        result = _monitor(dtl_warning_days=5.0).analyze(POSITIONS, ADV, VOL)
        aapl = next(p for p in result.positions if p.symbol == "AAPL")
        assert not aapl.is_warning

    def test_n_warnings_counted(self):
        result = _monitor(dtl_warning_days=5.0).analyze(POSITIONS, ADV, VOL)
        assert result.n_warnings == 1  # Only ILLIQ

    def test_worst_position_identified(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        assert result.worst_position == "ILLIQ"

    def test_concentration_between_0_and_1(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        assert 0 <= result.concentration <= 1.0


# ---------------------------------------------------------------------------
# Participation rate
# ---------------------------------------------------------------------------


class TestParticipation:
    def test_participation_max_capped(self):
        """Max participation rate config is respected in DTL calculation."""
        low_part = _monitor(max_participation_rate=0.05).analyze(POSITIONS, ADV, VOL)
        high_part = _monitor(max_participation_rate=0.20).analyze(POSITIONS, ADV, VOL)
        # Lower participation => higher DTL
        assert low_part.portfolio_dtl > high_part.portfolio_dtl

    def test_pct_of_adv_correct(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        aapl = next(p for p in result.positions if p.symbol == "AAPL")
        assert aapl.pct_of_adv == pytest.approx(5_000_000 / 100_000_000)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_portfolio(self):
        result = _monitor().analyze({}, {})
        assert len(result.positions) == 0

    def test_no_common_symbols_raises(self):
        with pytest.raises(ValueError, match="No common symbols"):
            _monitor().analyze({"A": 1_000_000}, {"B": 50_000_000})

    def test_single_position(self):
        result = _monitor().analyze(
            {"A": 10_000_000}, {"A": 50_000_000}, {"A": 0.20}
        )
        assert len(result.positions) == 1
        assert result.portfolio_value == pytest.approx(10_000_000)

    def test_default_volatility_used(self):
        """When volatility not provided, default 0.20 is assumed."""
        result = _monitor().analyze({"A": 1_000_000}, {"A": 50_000_000})
        assert result.positions[0].volatility == 0.20

    def test_very_large_position(self):
        """Position larger than ADV should still work."""
        result = _monitor().analyze(
            {"A": 500_000_000}, {"A": 10_000_000}, {"A": 0.30}
        )
        assert result.positions[0].days_to_liquidate > 100
        assert result.positions[0].is_warning

    def test_zero_adv_guarded(self):
        """ADV of 0 should not cause division by zero; DTL should be very high."""
        result = _monitor().analyze(
            {"A": 1_000_000}, {"A": 0.0}, {"A": 0.20}
        )
        assert result.positions[0].days_to_liquidate > 1_000_000
        assert result.positions[0].liquidity_score == 0.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfig:
    def test_higher_impact_coefficient(self):
        low = _monitor(impact_coefficient=0.05).analyze(POSITIONS, ADV, VOL)
        high = _monitor(impact_coefficient=0.20).analyze(POSITIONS, ADV, VOL)
        assert high.portfolio_lar_bps > low.portfolio_lar_bps

    def test_custom_score_cap(self):
        """Wider cap makes scores more lenient."""
        tight = _monitor(liquidity_score_cap=10.0).analyze(POSITIONS, ADV, VOL)
        wide = _monitor(liquidity_score_cap=100.0).analyze(POSITIONS, ADV, VOL)
        # ILLIQ DTL=20: tight=>score=0, wide=>score=80
        illiq_tight = next(p for p in tight.positions if p.symbol == "ILLIQ")
        illiq_wide = next(p for p in wide.positions if p.symbol == "ILLIQ")
        assert illiq_wide.liquidity_score > illiq_tight.liquidity_score


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        summary = result.summary()
        assert "Liquidity Risk Monitor" in summary
        assert "Portfolio DTL" in summary
        assert "Liquidity-at-Risk" in summary
        assert "liquidity score" in summary.lower()

    def test_summary_shows_positions(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        summary = result.summary()
        assert "AAPL" in summary
        assert "ILLIQ" in summary

    def test_summary_shows_warnings(self):
        result = _monitor().analyze(POSITIONS, ADV, VOL)
        summary = result.summary()
        assert "!!" in summary  # ILLIQ should be flagged
