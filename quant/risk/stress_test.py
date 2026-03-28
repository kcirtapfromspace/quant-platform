"""Portfolio stress testing framework.

Applies historical crisis replays or synthetic shocks to a portfolio and
measures the impact on P&L, drawdown, and risk metrics.  Helps the CIO
understand tail-risk exposure and set appropriate risk limits.

Scenario types:

  * **Historical**: Replay a named crisis period (GFC, COVID, Taper Tantrum,
    etc.) by applying the factor shocks observed during that period.
  * **Synthetic**: Apply user-defined percentage shocks to individual assets
    or the whole portfolio (e.g. "equities -20%, rates +200bp").
  * **Reverse stress**: Find the minimum shock that breaches a given
    drawdown threshold.

Usage::

    from quant.risk.stress_test import (
        StressTestEngine,
        HistoricalScenario,
        SyntheticScenario,
    )

    engine = StressTestEngine()
    results = engine.run(
        portfolio_weights={"AAPL": 0.3, "GOOG": 0.4, "MSFT": 0.3},
        portfolio_value=10_000_000,
        scenarios=[
            HistoricalScenario.GFC_2008,
            SyntheticScenario(name="Equity crash", shocks={"AAPL": -0.25, "GOOG": -0.30, "MSFT": -0.20}),
        ],
    )
    print(results.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


class HistoricalScenario(Enum):
    """Pre-defined historical crisis scenarios.

    Each value is a tuple of (name, description, default_shock_pct) where
    default_shock_pct is the broad market shock applied uniformly when
    per-asset shocks are not available.
    """

    GFC_2008 = ("GFC 2008", "Global Financial Crisis Sep-Nov 2008", -0.40)
    COVID_2020 = ("COVID 2020", "COVID-19 selloff Feb-Mar 2020", -0.34)
    DOT_COM_2000 = ("Dot-Com 2000", "Tech bubble burst 2000-2002", -0.45)
    TAPER_TANTRUM_2013 = ("Taper Tantrum 2013", "Fed taper announcement May-Jun 2013", -0.06)
    VOLMAGEDDON_2018 = ("Volmageddon 2018", "VIX spike Feb 2018", -0.10)
    RATE_SHOCK_2022 = ("Rate Shock 2022", "Fed hiking cycle 2022", -0.25)
    BLACK_MONDAY_1987 = ("Black Monday 1987", "Single-day crash Oct 19 1987", -0.22)
    EURO_CRISIS_2011 = ("Euro Crisis 2011", "European sovereign debt crisis", -0.19)
    FLASH_CRASH_2010 = ("Flash Crash 2010", "Intraday crash May 6 2010", -0.07)
    CHINA_DEVAL_2015 = ("China Deval 2015", "CNY devaluation Aug 2015", -0.12)


@dataclass(frozen=True, slots=True)
class SyntheticScenario:
    """User-defined shock scenario.

    Attributes:
        name:    Descriptive label.
        shocks:  Dict mapping asset symbols to return shocks (e.g. -0.20 for -20%).
                 Assets not in the dict receive zero shock.
        uniform_shock:  If set, apply this shock to all assets not explicitly
                        listed in ``shocks``.
    """

    name: str
    shocks: dict[str, float] = field(default_factory=dict)
    uniform_shock: float | None = None


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Impact of a single stress scenario on the portfolio."""

    scenario_name: str
    description: str
    portfolio_pnl: float  # Dollar P&L
    portfolio_return: float  # Return as fraction
    worst_asset: str
    worst_asset_return: float
    best_asset: str
    best_asset_return: float
    asset_pnls: dict[str, float]  # Per-asset dollar P&L
    asset_returns: dict[str, float]  # Per-asset return applied


@dataclass
class StressTestResult:
    """Aggregate stress test results across all scenarios."""

    n_scenarios: int
    portfolio_value: float
    portfolio_weights: dict[str, float]

    scenario_results: list[ScenarioResult]

    # Aggregates
    worst_scenario: str
    worst_scenario_return: float
    avg_scenario_return: float
    total_var_95: float  # 95th percentile loss across scenarios

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Stress Test Results ({self.n_scenarios} scenarios)",
            "=" * 65,
            "",
            f"Portfolio value         : ${self.portfolio_value:,.0f}",
            f"Positions               : {len(self.portfolio_weights)}",
            "",
            f"Worst scenario          : {self.worst_scenario}  ({self.worst_scenario_return:+.2%})",
            f"Avg scenario return     : {self.avg_scenario_return:+.2%}",
            f"VaR (95%, across scen.) : {self.total_var_95:+.2%}",
            "",
            f"{'Scenario':<30s} {'Return':>9s} {'P&L ($)':>14s}",
            "-" * 55,
        ]
        for sr in sorted(self.scenario_results, key=lambda x: x.portfolio_return):
            lines.append(
                f"{sr.scenario_name:<30s} {sr.portfolio_return:>+8.2%} "
                f"{sr.portfolio_pnl:>+13,.0f}"
            )

        return "\n".join(lines)


@dataclass
class ReverseStressResult:
    """Result of a reverse stress test."""

    threshold_drawdown: float
    min_uniform_shock: float  # Minimum uniform shock to breach threshold
    scenarios_that_breach: list[str]  # Named scenarios that breach


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class StressTestEngine:
    """Portfolio stress testing engine."""

    def run(
        self,
        portfolio_weights: dict[str, float],
        portfolio_value: float,
        scenarios: list[HistoricalScenario | SyntheticScenario],
    ) -> StressTestResult:
        """Run stress scenarios against a portfolio.

        Args:
            portfolio_weights: {symbol: weight} for current positions.
            portfolio_value:   Current portfolio AUM.
            scenarios:         List of scenarios to evaluate.

        Returns:
            :class:`StressTestResult` with per-scenario impacts.
        """
        if not scenarios:
            return StressTestResult(
                n_scenarios=0,
                portfolio_value=portfolio_value,
                portfolio_weights=dict(portfolio_weights),
                scenario_results=[],
                worst_scenario="N/A",
                worst_scenario_return=0.0,
                avg_scenario_return=0.0,
                total_var_95=0.0,
            )

        results: list[ScenarioResult] = []

        for scenario in scenarios:
            sr = self._evaluate_scenario(
                portfolio_weights, portfolio_value, scenario
            )
            results.append(sr)

        returns = [r.portfolio_return for r in results]
        worst = min(results, key=lambda r: r.portfolio_return)

        # VaR across scenarios (5th percentile = worst 5%)
        var_95 = float(np.percentile(returns, 5)) if len(returns) >= 2 else min(returns)

        return StressTestResult(
            n_scenarios=len(results),
            portfolio_value=portfolio_value,
            portfolio_weights=dict(portfolio_weights),
            scenario_results=results,
            worst_scenario=worst.scenario_name,
            worst_scenario_return=worst.portfolio_return,
            avg_scenario_return=float(np.mean(returns)),
            total_var_95=var_95,
        )

    def reverse_stress(
        self,
        portfolio_weights: dict[str, float],
        portfolio_value: float,
        threshold_drawdown: float,
        scenarios: list[HistoricalScenario | SyntheticScenario] | None = None,
        search_step: float = 0.01,
    ) -> ReverseStressResult:
        """Find the minimum uniform shock that breaches a drawdown threshold.

        Also reports which named scenarios (if provided) breach the threshold.

        Args:
            portfolio_weights: Current positions.
            portfolio_value:   Current AUM.
            threshold_drawdown: Drawdown threshold as positive fraction (e.g. 0.20 = 20%).
            scenarios:          Optional named scenarios to check.
            search_step:        Step size for binary-like search (default 1%).

        Returns:
            :class:`ReverseStressResult`.
        """
        # Binary search for minimum uniform shock
        lo, hi = 0.0, 1.0
        while hi - lo > search_step / 2:
            mid = (lo + hi) / 2
            test_scenario = SyntheticScenario(
                name="_reverse_test", uniform_shock=-mid
            )
            sr = self._evaluate_scenario(
                portfolio_weights, portfolio_value, test_scenario
            )
            if abs(sr.portfolio_return) >= threshold_drawdown:
                hi = mid
            else:
                lo = mid

        min_shock = (lo + hi) / 2

        # Check named scenarios
        breaching: list[str] = []
        if scenarios:
            for scenario in scenarios:
                sr = self._evaluate_scenario(
                    portfolio_weights, portfolio_value, scenario
                )
                if abs(sr.portfolio_return) >= threshold_drawdown:
                    breaching.append(sr.scenario_name)

        return ReverseStressResult(
            threshold_drawdown=threshold_drawdown,
            min_uniform_shock=min_shock,
            scenarios_that_breach=breaching,
        )

    def _evaluate_scenario(
        self,
        portfolio_weights: dict[str, float],
        portfolio_value: float,
        scenario: HistoricalScenario | SyntheticScenario,
    ) -> ScenarioResult:
        """Evaluate one scenario's impact on the portfolio."""
        if isinstance(scenario, HistoricalScenario):
            name, description, default_shock = scenario.value
            asset_shocks = dict.fromkeys(portfolio_weights, default_shock)
        else:
            name = scenario.name
            description = f"Synthetic: {name}"
            asset_shocks: dict[str, float] = {}
            for sym in portfolio_weights:
                if sym in scenario.shocks:
                    asset_shocks[sym] = scenario.shocks[sym]
                elif scenario.uniform_shock is not None:
                    asset_shocks[sym] = scenario.uniform_shock
                else:
                    asset_shocks[sym] = 0.0

        # Compute P&L
        asset_pnls: dict[str, float] = {}
        total_pnl = 0.0
        for sym, weight in portfolio_weights.items():
            shock = asset_shocks.get(sym, 0.0)
            position_value = weight * portfolio_value
            pnl = position_value * shock
            asset_pnls[sym] = pnl
            total_pnl += pnl

        portfolio_return = total_pnl / portfolio_value if portfolio_value > 0 else 0.0

        # Best/worst asset
        if asset_shocks:
            worst_sym = min(asset_shocks, key=lambda s: asset_shocks[s])
            best_sym = max(asset_shocks, key=lambda s: asset_shocks[s])
        else:
            worst_sym = best_sym = ""

        return ScenarioResult(
            scenario_name=name,
            description=description,
            portfolio_pnl=total_pnl,
            portfolio_return=portfolio_return,
            worst_asset=worst_sym,
            worst_asset_return=asset_shocks.get(worst_sym, 0.0),
            best_asset=best_sym,
            best_asset_return=asset_shocks.get(best_sym, 0.0),
            asset_pnls=asset_pnls,
            asset_returns=asset_shocks,
        )
