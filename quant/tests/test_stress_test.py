"""Tests for portfolio stress testing framework (QUA-84)."""
from __future__ import annotations

from quant.risk.stress_test import (
    HistoricalScenario,
    ReverseStressResult,
    ScenarioResult,
    StressTestEngine,
    StressTestResult,
    SyntheticScenario,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WEIGHTS = {"AAPL": 0.3, "GOOG": 0.4, "MSFT": 0.3}
PORTFOLIO_VALUE = 10_000_000.0


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        engine = StressTestEngine()
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, [HistoricalScenario.GFC_2008])
        assert isinstance(result, StressTestResult)

    def test_n_scenarios_tracked(self):
        engine = StressTestEngine()
        scenarios = [HistoricalScenario.GFC_2008, HistoricalScenario.COVID_2020]
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, scenarios)
        assert result.n_scenarios == 2

    def test_portfolio_value_preserved(self):
        engine = StressTestEngine()
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, [HistoricalScenario.GFC_2008])
        assert result.portfolio_value == PORTFOLIO_VALUE

    def test_empty_scenarios(self):
        engine = StressTestEngine()
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, [])
        assert result.n_scenarios == 0
        assert result.worst_scenario == "N/A"


# ---------------------------------------------------------------------------
# Historical scenarios
# ---------------------------------------------------------------------------


class TestHistoricalScenarios:
    def test_gfc_negative_return(self):
        engine = StressTestEngine()
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, [HistoricalScenario.GFC_2008])
        sr = result.scenario_results[0]
        assert sr.portfolio_return < 0

    def test_gfc_pnl_matches_return(self):
        engine = StressTestEngine()
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, [HistoricalScenario.GFC_2008])
        sr = result.scenario_results[0]
        expected_pnl = sr.portfolio_return * PORTFOLIO_VALUE
        assert abs(sr.portfolio_pnl - expected_pnl) < 1.0

    def test_all_historical_scenarios(self):
        engine = StressTestEngine()
        scenarios = list(HistoricalScenario)
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, scenarios)
        assert result.n_scenarios == len(HistoricalScenario)
        for sr in result.scenario_results:
            assert isinstance(sr, ScenarioResult)
            assert sr.portfolio_return <= 0  # All crises are negative

    def test_gfc_worst_among_common(self):
        engine = StressTestEngine()
        scenarios = [
            HistoricalScenario.GFC_2008,
            HistoricalScenario.TAPER_TANTRUM_2013,
            HistoricalScenario.FLASH_CRASH_2010,
        ]
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, scenarios)
        assert result.worst_scenario == "GFC 2008"

    def test_scenario_names_populated(self):
        engine = StressTestEngine()
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, [HistoricalScenario.COVID_2020])
        sr = result.scenario_results[0]
        assert sr.scenario_name == "COVID 2020"
        assert "COVID" in sr.description


# ---------------------------------------------------------------------------
# Synthetic scenarios
# ---------------------------------------------------------------------------


class TestSyntheticScenarios:
    def test_per_asset_shock(self):
        engine = StressTestEngine()
        scenario = SyntheticScenario(
            name="Tech crash",
            shocks={"AAPL": -0.30, "GOOG": -0.25, "MSFT": -0.20},
        )
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, [scenario])
        sr = result.scenario_results[0]
        # AAPL gets worst shock
        assert sr.worst_asset == "AAPL"
        assert sr.worst_asset_return == -0.30
        # MSFT gets best (least bad) shock
        assert sr.best_asset == "MSFT"
        assert sr.best_asset_return == -0.20

    def test_uniform_shock(self):
        engine = StressTestEngine()
        scenario = SyntheticScenario(
            name="Uniform -15%",
            uniform_shock=-0.15,
        )
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, [scenario])
        sr = result.scenario_results[0]
        assert abs(sr.portfolio_return - (-0.15)) < 1e-10

    def test_partial_shock(self):
        """Assets not in shocks dict and without uniform_shock get zero."""
        engine = StressTestEngine()
        scenario = SyntheticScenario(
            name="AAPL only",
            shocks={"AAPL": -0.50},
        )
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, [scenario])
        sr = result.scenario_results[0]
        # Only AAPL is shocked: 0.3 * -0.50 = -0.15 portfolio return
        assert abs(sr.portfolio_return - (-0.15)) < 1e-10
        assert sr.asset_returns["GOOG"] == 0.0
        assert sr.asset_returns["MSFT"] == 0.0

    def test_mixed_shocks(self):
        """Some assets explicitly shocked, others get uniform_shock."""
        engine = StressTestEngine()
        scenario = SyntheticScenario(
            name="Mixed",
            shocks={"AAPL": -0.40},
            uniform_shock=-0.10,
        )
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, [scenario])
        sr = result.scenario_results[0]
        assert sr.asset_returns["AAPL"] == -0.40
        assert sr.asset_returns["GOOG"] == -0.10
        assert sr.asset_returns["MSFT"] == -0.10

    def test_positive_shock(self):
        """Positive shocks should produce positive returns."""
        engine = StressTestEngine()
        scenario = SyntheticScenario(
            name="Rally",
            uniform_shock=0.10,
        )
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, [scenario])
        sr = result.scenario_results[0]
        assert sr.portfolio_return > 0

    def test_zero_shock(self):
        engine = StressTestEngine()
        scenario = SyntheticScenario(name="No shock")
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, [scenario])
        sr = result.scenario_results[0]
        assert abs(sr.portfolio_return) < 1e-12


# ---------------------------------------------------------------------------
# Asset-level P&L
# ---------------------------------------------------------------------------


class TestAssetPnL:
    def test_asset_pnls_sum_to_total(self):
        engine = StressTestEngine()
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, [HistoricalScenario.GFC_2008])
        sr = result.scenario_results[0]
        assert abs(sum(sr.asset_pnls.values()) - sr.portfolio_pnl) < 1.0

    def test_asset_pnl_proportional(self):
        engine = StressTestEngine()
        scenario = SyntheticScenario(
            name="Uniform",
            uniform_shock=-0.10,
        )
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, [scenario])
        sr = result.scenario_results[0]
        # AAPL: 0.3 * 10M * -0.10 = -300K
        assert abs(sr.asset_pnls["AAPL"] - (-300_000)) < 1.0
        # GOOG: 0.4 * 10M * -0.10 = -400K
        assert abs(sr.asset_pnls["GOOG"] - (-400_000)) < 1.0


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


class TestAggregates:
    def test_worst_scenario_is_minimum(self):
        engine = StressTestEngine()
        scenarios = list(HistoricalScenario)
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, scenarios)
        worst_ret = min(sr.portfolio_return for sr in result.scenario_results)
        assert result.worst_scenario_return == worst_ret

    def test_avg_scenario_return(self):
        engine = StressTestEngine()
        scenarios = [HistoricalScenario.GFC_2008, HistoricalScenario.TAPER_TANTRUM_2013]
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, scenarios)
        expected_avg = sum(sr.portfolio_return for sr in result.scenario_results) / 2
        assert abs(result.avg_scenario_return - expected_avg) < 1e-10

    def test_var_95_bounded(self):
        engine = StressTestEngine()
        scenarios = list(HistoricalScenario)
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, scenarios)
        assert result.total_var_95 <= result.avg_scenario_return
        assert result.total_var_95 >= result.worst_scenario_return


# ---------------------------------------------------------------------------
# Reverse stress test
# ---------------------------------------------------------------------------


class TestReverseStress:
    def test_returns_result(self):
        engine = StressTestEngine()
        result = engine.reverse_stress(
            WEIGHTS, PORTFOLIO_VALUE, threshold_drawdown=0.20
        )
        assert isinstance(result, ReverseStressResult)

    def test_finds_minimum_shock(self):
        engine = StressTestEngine()
        result = engine.reverse_stress(
            WEIGHTS, PORTFOLIO_VALUE, threshold_drawdown=0.20
        )
        # Minimum shock should be close to 0.20 for a uniform shock
        assert abs(result.min_uniform_shock - 0.20) < 0.02

    def test_higher_threshold_needs_bigger_shock(self):
        engine = StressTestEngine()
        r_small = engine.reverse_stress(
            WEIGHTS, PORTFOLIO_VALUE, threshold_drawdown=0.10
        )
        r_large = engine.reverse_stress(
            WEIGHTS, PORTFOLIO_VALUE, threshold_drawdown=0.30
        )
        assert r_large.min_uniform_shock >= r_small.min_uniform_shock

    def test_named_scenarios_breach(self):
        engine = StressTestEngine()
        result = engine.reverse_stress(
            WEIGHTS,
            PORTFOLIO_VALUE,
            threshold_drawdown=0.10,
            scenarios=[HistoricalScenario.GFC_2008, HistoricalScenario.FLASH_CRASH_2010],
        )
        # GFC (-40%) should breach 10% threshold
        assert "GFC 2008" in result.scenarios_that_breach
        # Flash Crash (-7%) should not breach 10% threshold
        assert "Flash Crash 2010" not in result.scenarios_that_breach


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_asset_portfolio(self):
        engine = StressTestEngine()
        result = engine.run(
            {"AAPL": 1.0},
            PORTFOLIO_VALUE,
            [SyntheticScenario(name="Crash", shocks={"AAPL": -0.50})],
        )
        sr = result.scenario_results[0]
        assert abs(sr.portfolio_return - (-0.50)) < 1e-10

    def test_zero_weight_asset_no_impact(self):
        engine = StressTestEngine()
        weights = {"AAPL": 0.5, "GOOG": 0.5, "MSFT": 0.0}
        result = engine.run(
            weights,
            PORTFOLIO_VALUE,
            [SyntheticScenario(name="Crash", shocks={"MSFT": -1.0})],
        )
        sr = result.scenario_results[0]
        assert abs(sr.portfolio_pnl) < 1.0

    def test_mixed_historical_and_synthetic(self):
        engine = StressTestEngine()
        scenarios = [
            HistoricalScenario.COVID_2020,
            SyntheticScenario(name="Custom", uniform_shock=-0.15),
        ]
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, scenarios)
        assert result.n_scenarios == 2


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        engine = StressTestEngine()
        scenarios = [
            HistoricalScenario.GFC_2008,
            HistoricalScenario.COVID_2020,
            SyntheticScenario(name="Custom -20%", uniform_shock=-0.20),
        ]
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, scenarios)
        summary = result.summary()
        assert "Stress Test" in summary
        assert "GFC 2008" in summary
        assert "COVID 2020" in summary
        assert "Custom -20%" in summary
        assert "Worst scenario" in summary

    def test_summary_sorted_by_return(self):
        engine = StressTestEngine()
        scenarios = [
            HistoricalScenario.TAPER_TANTRUM_2013,  # -6%
            HistoricalScenario.GFC_2008,  # -40%
        ]
        result = engine.run(WEIGHTS, PORTFOLIO_VALUE, scenarios)
        summary = result.summary()
        # GFC should appear before Taper Tantrum (sorted worst first)
        gfc_pos = summary.index("GFC 2008")
        taper_pos = summary.index("Taper Tantrum")
        assert gfc_pos < taper_pos
