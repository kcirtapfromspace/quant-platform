"""Tests for risk reporting — VaR, stress tests, concentration (QUA-30)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.risk.reporting import (
    RiskReporter,
    StressScenario,
    VaRMethod,
    _inv_norm,
)


def _make_returns(
    n: int = 252,
    mean: float = 0.0004,
    std: float = 0.012,
    seed: int = 42,
) -> pd.Series:
    """Generate synthetic daily returns."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


# ── VaR tests ────────────────────────────────────────────────────────────────


class TestHistoricalVaR:
    def test_var_positive_for_normal_returns(self):
        returns = _make_returns()
        reporter = RiskReporter(confidence_levels=(0.95,))
        report = reporter.generate_report(
            returns, positions={"AAPL": 50_000}, portfolio_value=100_000
        )
        hist_var = [v for v in report.var_results if v.method == VaRMethod.HISTORICAL]
        assert len(hist_var) == 1
        assert hist_var[0].var > 0

    def test_cvar_ge_var(self):
        """CVaR (expected shortfall) should be >= VaR."""
        returns = _make_returns()
        reporter = RiskReporter(confidence_levels=(0.95,))
        report = reporter.generate_report(
            returns, positions={"AAPL": 50_000}, portfolio_value=100_000
        )
        for v in report.var_results:
            assert v.cvar >= v.var or v.cvar == pytest.approx(v.var, abs=1e-8)

    def test_higher_confidence_higher_var(self):
        returns = _make_returns()
        reporter = RiskReporter(confidence_levels=(0.90, 0.99))
        report = reporter.generate_report(
            returns, positions={"AAPL": 50_000}, portfolio_value=100_000
        )
        hist = [v for v in report.var_results if v.method == VaRMethod.HISTORICAL]
        var_90 = next(v for v in hist if v.confidence == 0.90)
        var_99 = next(v for v in hist if v.confidence == 0.99)
        assert var_99.var > var_90.var

    def test_empty_returns_no_var(self):
        returns = pd.Series([], dtype=float)
        reporter = RiskReporter(confidence_levels=(0.95,))
        report = reporter.generate_report(returns, {}, 100_000)
        assert len(report.var_results) == 0


class TestParametricVaR:
    def test_parametric_var_positive(self):
        returns = _make_returns()
        reporter = RiskReporter(confidence_levels=(0.95,))
        report = reporter.generate_report(
            returns, positions={"AAPL": 50_000}, portfolio_value=100_000
        )
        param_var = [v for v in report.var_results if v.method == VaRMethod.PARAMETRIC]
        assert len(param_var) == 1
        assert param_var[0].var > 0

    def test_zero_volatility_returns_zero(self):
        returns = pd.Series([0.001] * 100)  # constant returns, zero vol
        reporter = RiskReporter(confidence_levels=(0.95,))
        report = reporter.generate_report(returns, {}, 100_000)
        param_var = [v for v in report.var_results if v.method == VaRMethod.PARAMETRIC]
        assert param_var[0].var == 0.0


class TestInvNorm:
    def test_inv_norm_0_5_is_zero(self):
        assert _inv_norm(0.5) == pytest.approx(0.0, abs=0.001)

    def test_inv_norm_0_05_is_negative(self):
        z = _inv_norm(0.05)
        assert z < -1.5  # should be ~-1.645

    def test_inv_norm_0_95_is_positive(self):
        z = _inv_norm(0.95)
        assert z > 1.5  # should be ~1.645

    def test_inv_norm_symmetry(self):
        assert _inv_norm(0.05) == pytest.approx(-_inv_norm(0.95), abs=0.001)

    def test_inv_norm_boundary_low(self):
        assert _inv_norm(0.0) == -10.0

    def test_inv_norm_boundary_high(self):
        assert _inv_norm(1.0) == 10.0


# ── Stress tests ─────────────────────────────────────────────────────────────


class TestStressTests:
    def test_portfolio_level_shock(self):
        scenario = StressScenario(name="crash", shocks={"__portfolio__": -0.20})
        reporter = RiskReporter(scenarios=[scenario])
        positions = {"AAPL": 60_000, "MSFT": 40_000}
        report = reporter.generate_report(_make_returns(), positions, 100_000)
        assert len(report.stress_results) == 1
        s = report.stress_results[0]
        assert s.portfolio_pnl == pytest.approx(-20_000)
        assert s.portfolio_return == pytest.approx(-0.20)

    def test_per_asset_shock(self):
        scenario = StressScenario(
            name="tech crash",
            shocks={"AAPL": -0.30, "MSFT": -0.25},
        )
        reporter = RiskReporter(scenarios=[scenario])
        positions = {"AAPL": 50_000, "MSFT": 30_000}
        report = reporter.generate_report(_make_returns(), positions, 100_000)
        s = report.stress_results[0]
        expected_pnl = 50_000 * (-0.30) + 30_000 * (-0.25)
        assert s.portfolio_pnl == pytest.approx(expected_pnl)
        assert s.per_asset["AAPL"] == pytest.approx(-15_000)
        assert s.per_asset["MSFT"] == pytest.approx(-7_500)

    def test_mixed_shock_per_asset_and_portfolio(self):
        """Per-asset shocks override portfolio-level for specified symbols."""
        scenario = StressScenario(
            name="mixed",
            shocks={"__portfolio__": -0.10, "AAPL": -0.30},
        )
        reporter = RiskReporter(scenarios=[scenario])
        positions = {"AAPL": 50_000, "MSFT": 50_000}
        report = reporter.generate_report(_make_returns(), positions, 100_000)
        s = report.stress_results[0]
        # AAPL uses -30%, MSFT falls back to __portfolio__ -10%
        assert s.per_asset["AAPL"] == pytest.approx(-15_000)
        assert s.per_asset["MSFT"] == pytest.approx(-5_000)

    def test_default_scenarios_present(self):
        reporter = RiskReporter()
        report = reporter.generate_report(
            _make_returns(), {"AAPL": 50_000}, 100_000
        )
        names = {s.scenario_name for s in report.stress_results}
        assert "2008 GFC" in names
        assert "2020 COVID crash" in names

    def test_empty_positions_portfolio_shock(self):
        scenario = StressScenario(name="crash", shocks={"__portfolio__": -0.15})
        reporter = RiskReporter(scenarios=[scenario])
        report = reporter.generate_report(_make_returns(), {}, 100_000)
        s = report.stress_results[0]
        assert s.portfolio_pnl == pytest.approx(-15_000)


# ── Concentration tests ──────────────────────────────────────────────────────


class TestConcentration:
    def test_single_position_hhi_one(self):
        reporter = RiskReporter(scenarios=[])
        report = reporter.generate_report(
            _make_returns(), {"AAPL": 100_000}, 100_000
        )
        c = report.concentration
        assert c is not None
        assert c.hhi == pytest.approx(1.0)
        assert c.effective_n == pytest.approx(1.0)
        assert c.top1_weight == pytest.approx(1.0)

    def test_equal_weight_positions(self):
        positions = {f"SYM{i}": 10_000 for i in range(10)}
        reporter = RiskReporter(scenarios=[])
        report = reporter.generate_report(_make_returns(), positions, 100_000)
        c = report.concentration
        assert c is not None
        assert c.hhi == pytest.approx(0.10, abs=0.001)
        assert c.effective_n == pytest.approx(10.0, abs=0.1)
        assert c.top1_weight == pytest.approx(0.10, abs=0.001)
        assert c.n_positions == 10

    def test_concentrated_portfolio(self):
        positions = {"AAPL": 80_000, "MSFT": 10_000, "GOOG": 10_000}
        reporter = RiskReporter(scenarios=[])
        report = reporter.generate_report(_make_returns(), positions, 100_000)
        c = report.concentration
        assert c is not None
        assert c.hhi > 0.5  # concentrated
        assert c.top1_weight == pytest.approx(0.80)

    def test_empty_positions_no_concentration(self):
        reporter = RiskReporter(scenarios=[])
        report = reporter.generate_report(_make_returns(), {}, 100_000)
        assert report.concentration is None


# ── Volatility and drawdown ──────────────────────────────────────────────────


class TestVolatilityAndDrawdown:
    def test_annualised_volatility_reasonable(self):
        returns = _make_returns(std=0.012)
        reporter = RiskReporter(scenarios=[])
        report = reporter.generate_report(returns, {}, 100_000)
        # 1.2% daily × √252 ≈ 19%
        assert 0.10 < report.annualised_volatility < 0.30

    def test_max_drawdown_positive(self):
        returns = _make_returns()
        reporter = RiskReporter(scenarios=[])
        report = reporter.generate_report(returns, {}, 100_000)
        assert report.max_drawdown > 0

    def test_max_drawdown_all_positive_returns(self):
        returns = pd.Series([0.01] * 100)  # always going up
        reporter = RiskReporter(scenarios=[])
        report = reporter.generate_report(returns, {}, 100_000)
        assert report.max_drawdown == pytest.approx(0.0)


# ── Report summary ───────────────────────────────────────────────────────────


class TestRiskReportSummary:
    def test_summary_contains_key_sections(self):
        returns = _make_returns()
        positions = {"AAPL": 50_000, "MSFT": 30_000, "GOOG": 20_000}
        reporter = RiskReporter()
        report = reporter.generate_report(returns, positions, 100_000)
        text = report.summary()
        assert "Risk Report" in text
        assert "VaR" in text
        assert "Stress" in text
        assert "Concentration" in text
        assert "volatility" in text.lower()
