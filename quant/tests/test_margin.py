"""Tests for leverage and margin management (QUA-114)."""
from __future__ import annotations

import pytest

from quant.risk.margin import (
    MarginConfig,
    MarginManager,
    MarginStatus,
    PositionMargin,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _long_only() -> dict[str, float]:
    """Simple long-only portfolio, total $1M in positions."""
    return {"AAPL": 400_000, "MSFT": 300_000, "GOOG": 300_000}


def _long_short() -> dict[str, float]:
    """130/30 portfolio: $1.3M long, $300K short."""
    return {
        "AAPL": 400_000,
        "MSFT": 500_000,
        "GOOG": 400_000,
        "TSLA": -150_000,
        "META": -150_000,
    }


def _leveraged() -> dict[str, float]:
    """Highly leveraged portfolio: $3M gross on $1M NAV."""
    return {
        "AAPL": 1_000_000,
        "MSFT": 500_000,
        "GOOG": 500_000,
        "TSLA": -500_000,
        "META": -500_000,
    }


NAV = 1_000_000.0


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_margin_status(self):
        result = MarginManager().compute(_long_only(), NAV)
        assert isinstance(result, MarginStatus)

    def test_default_config(self):
        mgr = MarginManager()
        assert mgr.config.initial_margin_rate == 0.50
        assert mgr.config.maintenance_margin_rate == 0.25

    def test_custom_config(self):
        cfg = MarginConfig(borrow_rate_annual=0.08)
        mgr = MarginManager(cfg)
        assert mgr.config.borrow_rate_annual == 0.08

    def test_zero_nav_raises(self):
        with pytest.raises(ValueError, match="positive"):
            MarginManager().compute(_long_only(), 0.0)

    def test_negative_nav_raises(self):
        with pytest.raises(ValueError, match="positive"):
            MarginManager().compute(_long_only(), -100)

    def test_empty_positions(self):
        result = MarginManager().compute({}, NAV)
        assert result.gross_leverage == pytest.approx(0.0)
        assert not result.margin_call


# ---------------------------------------------------------------------------
# Leverage
# ---------------------------------------------------------------------------


class TestLeverage:
    def test_long_only_leverage(self):
        result = MarginManager().compute(_long_only(), NAV)
        assert result.gross_leverage == pytest.approx(1.0)
        assert result.net_leverage == pytest.approx(1.0)

    def test_long_short_gross_leverage(self):
        """130/30 portfolio → gross = 1.6x."""
        result = MarginManager().compute(_long_short(), NAV)
        assert result.gross_leverage == pytest.approx(1.6)

    def test_long_short_net_leverage(self):
        """130/30 portfolio → net = 1.0x (long - short)."""
        result = MarginManager().compute(_long_short(), NAV)
        # long=1.3M, short=0.3M → net = 1.0M / 1.0M = 1.0x
        assert result.net_leverage == pytest.approx(1.0)

    def test_exposures(self):
        result = MarginManager().compute(_long_short(), NAV)
        assert result.long_exposure == pytest.approx(1_300_000)
        assert result.short_exposure == pytest.approx(300_000)
        assert result.gross_exposure == pytest.approx(1_600_000)

    def test_leverage_headroom(self):
        """With 4x max and 1.6x gross, headroom = 2.4M."""
        result = MarginManager().compute(_long_short(), NAV)
        expected = 4.0 * NAV - 1_600_000
        assert result.leverage_headroom == pytest.approx(expected)

    def test_position_leverage_contrib(self):
        result = MarginManager().compute({"AAPL": 500_000}, NAV)
        pm = result.position_margins[0]
        assert pm.leverage_contrib == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Margin requirements
# ---------------------------------------------------------------------------


class TestMarginReq:
    def test_initial_margin_long_only(self):
        """$1M gross × 50% = $500K initial margin."""
        result = MarginManager().compute(_long_only(), NAV)
        assert result.initial_margin_req == pytest.approx(500_000)

    def test_maintenance_margin_long_only(self):
        """$1M gross × 25% = $250K maintenance."""
        result = MarginManager().compute(_long_only(), NAV)
        assert result.maintenance_margin_req == pytest.approx(250_000)

    def test_excess_margin_long_only(self):
        """$1M NAV - $250K maint = $750K excess."""
        result = MarginManager().compute(_long_only(), NAV)
        assert result.excess_margin == pytest.approx(750_000)

    def test_margin_utilisation(self):
        result = MarginManager().compute(_long_only(), NAV)
        assert result.margin_utilisation == pytest.approx(0.25)

    def test_no_margin_call_long_only(self):
        result = MarginManager().compute(_long_only(), NAV)
        assert result.margin_call is False

    def test_margin_call_when_overleveraged(self):
        """Extreme leverage: $5M gross on $1M NAV → maint=$1.25M > $1M NAV."""
        positions = {"AAPL": 2_500_000, "MSFT": -2_500_000}
        result = MarginManager().compute(positions, NAV)
        assert result.margin_call is True
        assert result.excess_margin < 0

    def test_distance_to_call(self):
        result = MarginManager().compute(_long_only(), NAV)
        # excess/nav = 750K/1M = 0.75
        assert result.distance_to_call_pct == pytest.approx(0.75)

    def test_margin_override(self):
        """Per-symbol margin rate override."""
        cfg = MarginConfig(margin_overrides={"AAPL": 0.40})
        result = MarginManager(cfg).compute({"AAPL": 1_000_000}, NAV)
        pm = result.position_margins[0]
        assert pm.margin_rate == pytest.approx(0.40)
        assert pm.maintenance_required == pytest.approx(400_000)


# ---------------------------------------------------------------------------
# Financing costs
# ---------------------------------------------------------------------------


class TestFinancing:
    def test_no_borrowing_no_cost(self):
        """Long-only with gross=NAV → no borrowing, zero cost."""
        result = MarginManager().compute(_long_only(), NAV)
        assert result.daily_financing_cost == pytest.approx(0.0)
        assert result.annual_financing_cost == pytest.approx(0.0)

    def test_borrowing_cost(self):
        """130/30: gross=1.6M, NAV=1M → borrowed=600K."""
        cfg = MarginConfig(borrow_rate_annual=0.05)
        result = MarginManager(cfg).compute(_long_short(), NAV)
        borrowed = 1_600_000 - NAV  # 600K
        expected_annual = borrowed * 0.05  # 30K
        assert result.annual_financing_cost == pytest.approx(expected_annual)
        assert result.daily_financing_cost == pytest.approx(expected_annual / 252)

    def test_higher_leverage_higher_cost(self):
        low = MarginManager().compute(_long_short(), NAV)
        high = MarginManager().compute(_leveraged(), NAV)
        assert high.annual_financing_cost > low.annual_financing_cost

    def test_financing_impact_series(self):
        mgr = MarginManager()
        series = mgr.financing_impact(_long_short(), NAV, holding_days=10)
        assert len(series) == 10
        # Cumulative → monotonically increasing
        assert all(series.iloc[i] <= series.iloc[i + 1] for i in range(9))

    def test_financing_impact_linear(self):
        """Cumulative cost should grow linearly (constant daily cost)."""
        mgr = MarginManager()
        series = mgr.financing_impact(_long_short(), NAV, holding_days=5)
        diffs = series.diff().dropna()
        # All daily increments should be equal
        assert diffs.std() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Liquidation value
# ---------------------------------------------------------------------------


class TestLiquidation:
    def test_no_haircut(self):
        liq = MarginManager().liquidation_value(_long_only(), NAV)
        assert liq == pytest.approx(NAV)

    def test_with_haircuts(self):
        haircuts = {"AAPL": 0.02, "MSFT": 0.01, "GOOG": 0.03}
        liq = MarginManager().liquidation_value(_long_only(), NAV, haircuts)
        expected_cost = 400_000 * 0.02 + 300_000 * 0.01 + 300_000 * 0.03
        assert liq == pytest.approx(NAV - expected_cost)

    def test_partial_haircuts(self):
        """Only some symbols have haircuts."""
        haircuts = {"AAPL": 0.05}
        liq = MarginManager().liquidation_value(_long_only(), NAV, haircuts)
        expected_cost = 400_000 * 0.05
        assert liq == pytest.approx(NAV - expected_cost)

    def test_short_positions_haircut(self):
        """Haircut applies to absolute value."""
        positions = {"AAPL": -500_000}
        haircuts = {"AAPL": 0.02}
        liq = MarginManager().liquidation_value(positions, NAV, haircuts)
        assert liq == pytest.approx(NAV - 500_000 * 0.02)


# ---------------------------------------------------------------------------
# Per-position analytics
# ---------------------------------------------------------------------------


class TestPositionMargins:
    def test_position_margin_types(self):
        result = MarginManager().compute(_long_only(), NAV)
        for pm in result.position_margins:
            assert isinstance(pm, PositionMargin)

    def test_position_count(self):
        result = MarginManager().compute(_long_only(), NAV)
        assert len(result.position_margins) == 3

    def test_abs_value(self):
        result = MarginManager().compute({"AAPL": -500_000}, NAV)
        pm = result.position_margins[0]
        assert pm.abs_value == pytest.approx(500_000)
        assert pm.market_value == pytest.approx(-500_000)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_position(self):
        result = MarginManager().compute({"AAPL": 500_000}, NAV)
        assert result.gross_leverage == pytest.approx(0.5)

    def test_all_short(self):
        positions = {"AAPL": -500_000, "MSFT": -500_000}
        result = MarginManager().compute(positions, NAV)
        assert result.long_exposure == pytest.approx(0.0)
        assert result.short_exposure == pytest.approx(1_000_000)
        assert result.net_leverage == pytest.approx(-1.0)

    def test_very_small_nav(self):
        result = MarginManager().compute({"AAPL": 1_000_000}, 1.0)
        assert result.gross_leverage == pytest.approx(1_000_000.0)
        assert result.margin_call is True


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_content(self):
        result = MarginManager().compute(_long_short(), NAV)
        summary = result.summary()
        assert "Margin & Leverage" in summary
        assert "Gross leverage" in summary
        assert "MARGIN CALL" in summary

    def test_summary_no_call(self):
        result = MarginManager().compute(_long_only(), NAV)
        summary = result.summary()
        assert "MARGIN CALL        : No" in summary
