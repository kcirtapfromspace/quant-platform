"""Tests for portfolio scenario analyzer (QUA-109)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.portfolio.scenario_analyzer import (
    MarginalRisk,
    ScenarioAnalyzer,
    ScenarioConfig,
    ScenarioResult,
    ShockResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
N = len(SYMBOLS)


def _covariance(seed: int = 42) -> pd.DataFrame:
    """Generate a realistic daily covariance matrix."""
    rng = np.random.default_rng(seed)
    # Factor structure
    loadings = rng.normal(0, 0.01, (N, 2))
    factor_cov = np.eye(2) * 0.0001
    specific = np.diag(rng.uniform(0.00005, 0.0002, N))
    cov = loadings @ factor_cov @ loadings.T + specific
    # Ensure symmetric PD
    cov = (cov + cov.T) / 2.0
    return pd.DataFrame(cov, index=SYMBOLS, columns=SYMBOLS)


def _equal_weights() -> dict[str, float]:
    return dict.fromkeys(SYMBOLS, 1.0 / N)


def _concentrated_weights() -> dict[str, float]:
    """60% AAPL, 10% each for the rest."""
    w = dict.fromkeys(SYMBOLS, 0.1)
    w["AAPL"] = 0.60
    return w


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_weight_change_returns_result(self):
        cov = _covariance()
        analyzer = ScenarioAnalyzer(cov)
        result = analyzer.weight_change(
            _equal_weights(), {"AAPL": 0.05, "MSFT": -0.05},
        )
        assert isinstance(result, ScenarioResult)

    def test_config_accessible(self):
        cfg = ScenarioConfig(annualise=False)
        analyzer = ScenarioAnalyzer(_covariance(), cfg)
        assert analyzer.config.annualise is False

    def test_symbols_from_covariance(self):
        analyzer = ScenarioAnalyzer(_covariance())
        assert analyzer.symbols == SYMBOLS

    def test_no_change_zero_delta(self):
        """Zero changes should produce zero risk delta."""
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.weight_change(_equal_weights(), {})
        assert result.risk_delta.vol_change == pytest.approx(0.0, abs=1e-10)

    def test_empty_portfolio(self):
        """Empty portfolio should have zero vol."""
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.weight_change({}, {"AAPL": 0.50})
        assert result.risk_delta.vol_before == pytest.approx(0.0)
        assert result.risk_delta.vol_after > 0


# ---------------------------------------------------------------------------
# Weight change scenarios
# ---------------------------------------------------------------------------


class TestWeightChange:
    def test_vol_changes(self):
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.weight_change(
            _equal_weights(), {"AAPL": 0.20, "MSFT": -0.20},
        )
        assert result.risk_delta.vol_change != 0.0

    def test_tracking_error_positive(self):
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.weight_change(
            _equal_weights(), {"AAPL": 0.10, "GOOG": -0.10},
        )
        assert result.risk_delta.tracking_error > 0

    def test_weights_before_after(self):
        w = _equal_weights()
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.weight_change(w, {"AAPL": 0.10})
        assert result.weights_before == w
        assert result.weights_after["AAPL"] == pytest.approx(0.20 + 0.10)

    def test_marginal_risks_populated(self):
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.weight_change(
            _equal_weights(), {"AAPL": 0.10, "MSFT": -0.10},
        )
        assert len(result.marginal_risks) == 2
        syms = {mr.symbol for mr in result.marginal_risks}
        assert syms == {"AAPL", "MSFT"}

    def test_marginal_risk_type(self):
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.weight_change(
            _equal_weights(), {"AAPL": 0.10},
        )
        for mr in result.marginal_risks:
            assert isinstance(mr, MarginalRisk)

    def test_vol_change_pct(self):
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.weight_change(
            _equal_weights(), {"AAPL": 0.10},
        )
        rd = result.risk_delta
        expected_pct = rd.vol_change / rd.vol_before
        assert rd.vol_change_pct == pytest.approx(expected_pct, rel=1e-6)


# ---------------------------------------------------------------------------
# Add position
# ---------------------------------------------------------------------------


class TestAddPosition:
    def test_adds_new_position(self):
        w = {"AAPL": 0.50, "MSFT": 0.50}
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.add_position(w, "GOOG", 0.10)
        assert "GOOG" in result.weights_after
        assert result.weights_after["GOOG"] == pytest.approx(0.10)

    def test_fund_from_reduces_source(self):
        w = {"AAPL": 0.50, "MSFT": 0.50}
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.add_position(w, "GOOG", 0.10, fund_from="AAPL")
        assert result.weights_after["AAPL"] == pytest.approx(0.40)
        assert result.weights_after["GOOG"] == pytest.approx(0.10)

    def test_name_includes_symbol(self):
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.add_position(_equal_weights(), "GOOG", 0.05)
        assert "GOOG" in result.name


# ---------------------------------------------------------------------------
# Remove position
# ---------------------------------------------------------------------------


class TestRemovePosition:
    def test_removes_position(self):
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.remove_position(_equal_weights(), "AAPL")
        assert result.weights_after.get("AAPL", 0.0) == pytest.approx(0.0, abs=1e-15)

    def test_redistribute(self):
        w = dict.fromkeys(["AAPL", "MSFT", "GOOG", "AMZN"], 0.25)
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.remove_position(w, "AAPL", redistribute=True)
        # Remaining should sum to original total
        remaining = {s: v for s, v in result.weights_after.items() if abs(v) > 1e-15}
        assert "AAPL" not in remaining
        total = sum(remaining.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_name_includes_symbol(self):
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.remove_position(_equal_weights(), "META")
        assert "META" in result.name


# ---------------------------------------------------------------------------
# Market shock
# ---------------------------------------------------------------------------


class TestMarketShock:
    def test_returns_shock_result(self):
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.market_shock(
            _equal_weights(), {"AAPL": -0.05, "MSFT": -0.03},
        )
        assert isinstance(result, ShockResult)

    def test_pnl_calculation(self):
        w = {"AAPL": 0.50, "MSFT": 0.50}
        shocks = {"AAPL": -0.10, "MSFT": 0.05}
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.market_shock(w, shocks)
        expected = 0.50 * (-0.10) + 0.50 * 0.05
        assert result.portfolio_pnl == pytest.approx(expected)

    def test_position_pnls(self):
        w = {"AAPL": 0.60, "MSFT": 0.40}
        shocks = {"AAPL": -0.05}
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.market_shock(w, shocks)
        assert "AAPL" in result.position_pnls
        assert result.position_pnls["AAPL"] == pytest.approx(0.60 * -0.05)

    def test_zero_shock_zero_pnl(self):
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.market_shock(_equal_weights(), {})
        assert result.portfolio_pnl == pytest.approx(0.0)

    def test_vol_before_populated(self):
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.market_shock(_equal_weights(), {"AAPL": -0.10})
        assert result.vol_before > 0


# ---------------------------------------------------------------------------
# Vol shock
# ---------------------------------------------------------------------------


class TestVolShock:
    def test_vol_scales_linearly(self):
        """Vol should scale by the multiplier."""
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.vol_shock(_equal_weights(), 2.0)
        assert result.risk_delta.vol_after == pytest.approx(
            result.risk_delta.vol_before * 2.0, rel=1e-6,
        )

    def test_half_vol(self):
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.vol_shock(_equal_weights(), 0.5)
        assert result.risk_delta.vol_after < result.risk_delta.vol_before

    def test_no_tracking_error(self):
        """Vol shock doesn't change weights, so TE should be zero."""
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.vol_shock(_equal_weights(), 1.5)
        assert result.risk_delta.tracking_error == 0.0

    def test_custom_name(self):
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.vol_shock(_equal_weights(), 2.0, name="Crisis")
        assert result.name == "Crisis"


# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------


class TestAnnualisation:
    def test_annualised_vol_larger(self):
        cov = _covariance()
        ann = ScenarioAnalyzer(cov, ScenarioConfig(annualise=True))
        raw = ScenarioAnalyzer(cov, ScenarioConfig(annualise=False))
        ann_result = ann.weight_change(_equal_weights(), {"AAPL": 0.10})
        raw_result = raw.weight_change(_equal_weights(), {"AAPL": 0.10})
        assert ann_result.risk_delta.vol_after > raw_result.risk_delta.vol_after


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unknown_symbol_in_weights(self):
        """Symbols not in covariance should be silently ignored."""
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.weight_change(
            {"AAPL": 0.50, "UNKNOWN": 0.50},
            {"AAPL": 0.10},
        )
        assert result.risk_delta.vol_before > 0

    def test_single_position(self):
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.weight_change({"AAPL": 1.0}, {})
        assert result.risk_delta.vol_before > 0

    def test_long_short(self):
        w = {"AAPL": 0.50, "MSFT": -0.50}
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.weight_change(w, {"GOOG": 0.20})
        assert result.risk_delta.vol_before > 0
        assert result.risk_delta.vol_after > 0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_scenario_summary(self):
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.weight_change(
            _equal_weights(), {"AAPL": 0.10, "MSFT": -0.10},
        )
        summary = result.summary()
        assert "Scenario" in summary
        assert "Vol before" in summary
        assert "Vol after" in summary
        assert "Tracking error" in summary

    def test_shock_summary(self):
        analyzer = ScenarioAnalyzer(_covariance())
        result = analyzer.market_shock(
            _equal_weights(), {"AAPL": -0.10},
        )
        summary = result.summary()
        assert "Shock Scenario" in summary
        assert "Portfolio P&L" in summary
