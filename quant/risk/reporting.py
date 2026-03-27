"""Risk reporting — VaR, CVaR, stress tests, concentration metrics.

Produces structured risk reports from portfolio returns and position data.
All calculations use numpy only (no scipy dependency).

Key outputs:
  * **Value at Risk (VaR)**: Historical and parametric, at configurable
    confidence levels.
  * **Conditional VaR (CVaR)**: Expected shortfall — the average loss in
    the tail beyond the VaR threshold.
  * **Stress scenarios**: Apply historical or custom shocks to the portfolio
    and estimate P&L impact.
  * **Concentration metrics**: Herfindahl-Hirschman Index (HHI), top-N
    exposure, effective number of bets.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class VaRMethod(str, Enum):
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"


@dataclass
class VaRResult:
    """Value-at-Risk and Conditional VaR for a single confidence level."""

    confidence: float  # e.g. 0.95
    method: VaRMethod
    var: float  # positive = loss (as a fraction of portfolio)
    cvar: float  # expected shortfall (conditional on exceeding VaR)
    n_observations: int


@dataclass
class StressScenario:
    """A named stress scenario with per-asset or portfolio-level shocks.

    Attributes:
        name: Human-readable label (e.g. "2008 GFC", "Rates +200bp").
        shocks: Mapping of symbol → return shock (e.g. {"AAPL": -0.30}).
            A special key ``"__portfolio__"`` applies a uniform shock to the
            entire portfolio.
    """

    name: str
    shocks: dict[str, float]


@dataclass
class StressResult:
    """Outcome of applying a stress scenario to the current portfolio."""

    scenario_name: str
    portfolio_pnl: float  # dollar P&L (negative = loss)
    portfolio_return: float  # as fraction of portfolio value
    per_asset: dict[str, float]  # symbol → dollar P&L


@dataclass
class ConcentrationMetrics:
    """Portfolio concentration statistics."""

    hhi: float  # Herfindahl-Hirschman Index (0 = perfectly diversified, 1 = single stock)
    effective_n: float  # 1/HHI — effective number of independent bets
    top1_weight: float  # weight of the largest position
    top5_weight: float  # combined weight of the 5 largest positions
    n_positions: int


@dataclass
class RiskReport:
    """Aggregated risk report."""

    var_results: list[VaRResult]
    stress_results: list[StressResult]
    concentration: ConcentrationMetrics | None
    annualised_volatility: float
    max_drawdown: float
    portfolio_value: float

    def summary(self) -> str:
        """Human-readable risk summary."""
        lines = [
            f"Risk Report — Portfolio ${self.portfolio_value:,.0f}",
            "═" * 55,
            f"  Ann. volatility     : {self.annualised_volatility:.2%}",
            f"  Max drawdown        : {self.max_drawdown:.2%}",
        ]

        if self.var_results:
            lines.append("")
            lines.append("  Value at Risk:")
            for v in self.var_results:
                lines.append(
                    f"    {v.method.value:12s} {v.confidence:.0%} VaR = {v.var:.4%}  "
                    f"CVaR = {v.cvar:.4%}  (n={v.n_observations})"
                )

        if self.stress_results:
            lines.append("")
            lines.append("  Stress Tests:")
            for s in self.stress_results:
                lines.append(
                    f"    {s.scenario_name:25s}  P&L = ${s.portfolio_pnl:>+12,.0f}"
                    f"  ({s.portfolio_return:+.2%})"
                )

        if self.concentration is not None:
            c = self.concentration
            lines.append("")
            lines.append("  Concentration:")
            lines.append(f"    HHI             : {c.hhi:.4f}")
            lines.append(f"    Effective N     : {c.effective_n:.1f}")
            lines.append(f"    Top-1 weight    : {c.top1_weight:.2%}")
            lines.append(f"    Top-5 weight    : {c.top5_weight:.2%}")
            lines.append(f"    # positions     : {c.n_positions}")

        return "\n".join(lines)


# ── Predefined stress scenarios ──────────────────────────────────────────────

SCENARIOS_2008_GFC = StressScenario(
    name="2008 GFC",
    shocks={"__portfolio__": -0.38},  # S&P500 peak-to-trough ~-57%, single-month ~-17%
)

SCENARIOS_COVID_CRASH = StressScenario(
    name="2020 COVID crash",
    shocks={"__portfolio__": -0.34},  # S&P500 Feb-Mar 2020
)

SCENARIOS_RATES_SHOCK = StressScenario(
    name="Rates +200bp",
    shocks={"__portfolio__": -0.12},  # Approximate equity impact of sharp rate rise
)

SCENARIOS_FLASH_CRASH = StressScenario(
    name="Flash crash (5-min)",
    shocks={"__portfolio__": -0.08},  # Sudden liquidity vacuum
)

DEFAULT_SCENARIOS = [
    SCENARIOS_2008_GFC,
    SCENARIOS_COVID_CRASH,
    SCENARIOS_RATES_SHOCK,
    SCENARIOS_FLASH_CRASH,
]


# ── Risk calculator ──────────────────────────────────────────────────────────


class RiskReporter:
    """Computes risk metrics from portfolio returns and positions.

    Args:
        confidence_levels: VaR confidence levels (default: 95% and 99%).
        scenarios: Stress scenarios to apply (default: built-in set).
        trading_days_per_year: For annualising volatility (default 252).
    """

    def __init__(
        self,
        confidence_levels: tuple[float, ...] = (0.95, 0.99),
        scenarios: list[StressScenario] | None = None,
        trading_days_per_year: int = 252,
    ) -> None:
        self._confidence_levels = confidence_levels
        self._scenarios = scenarios if scenarios is not None else list(DEFAULT_SCENARIOS)
        self._trading_days = trading_days_per_year

    def generate_report(
        self,
        returns: pd.Series,
        positions: dict[str, float],
        portfolio_value: float,
    ) -> RiskReport:
        """Generate a full risk report.

        Args:
            returns: Daily portfolio returns as a Series (e.g. from
                ``equity_curve.pct_change()``).
            positions: Current positions as ``{symbol: dollar_value}``.
            portfolio_value: Current total portfolio value.
        """
        returns_clean = returns.dropna()

        var_results = self._compute_var(returns_clean)
        stress_results = self._run_stress_tests(positions, portfolio_value)
        concentration = self._compute_concentration(positions, portfolio_value)
        ann_vol = self._annualised_volatility(returns_clean)
        max_dd = self._max_drawdown(returns_clean)

        return RiskReport(
            var_results=var_results,
            stress_results=stress_results,
            concentration=concentration,
            annualised_volatility=ann_vol,
            max_drawdown=max_dd,
            portfolio_value=portfolio_value,
        )

    # ── VaR / CVaR ─────────────────────────────────────────────────────

    def _compute_var(self, returns: pd.Series) -> list[VaRResult]:
        results: list[VaRResult] = []
        if len(returns) < 2:
            return results

        arr = returns.to_numpy()
        n = len(arr)

        for conf in self._confidence_levels:
            # Historical VaR
            h_var, h_cvar = self._historical_var(arr, conf)
            results.append(
                VaRResult(
                    confidence=conf,
                    method=VaRMethod.HISTORICAL,
                    var=h_var,
                    cvar=h_cvar,
                    n_observations=n,
                )
            )

            # Parametric VaR (Gaussian)
            p_var, p_cvar = self._parametric_var(arr, conf)
            results.append(
                VaRResult(
                    confidence=conf,
                    method=VaRMethod.PARAMETRIC,
                    var=p_var,
                    cvar=p_cvar,
                    n_observations=n,
                )
            )

        return results

    @staticmethod
    def _historical_var(returns: np.ndarray, confidence: float) -> tuple[float, float]:
        """Historical VaR and CVaR."""
        alpha = 1.0 - confidence
        cutoff = np.percentile(returns, alpha * 100)
        var = -cutoff  # positive = loss
        tail = returns[returns <= cutoff]
        cvar = -float(np.mean(tail)) if len(tail) > 0 else var
        return float(var), float(cvar)

    @staticmethod
    def _parametric_var(returns: np.ndarray, confidence: float) -> tuple[float, float]:
        """Parametric (Gaussian) VaR and CVaR."""
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        if sigma == 0:
            return 0.0, 0.0

        alpha = 1.0 - confidence
        # Approximate inverse normal CDF using rational approximation
        z = _inv_norm(alpha)
        var = -(mu + z * sigma)  # positive = loss

        # CVaR for Gaussian: mu - sigma * phi(z) / alpha
        # where phi is the standard normal PDF
        phi_z = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        cvar = -(mu - sigma * phi_z / alpha)

        return float(max(var, 0.0)), float(max(cvar, 0.0))

    # ── Stress tests ───────────────────────────────────────────────────

    def _run_stress_tests(
        self,
        positions: dict[str, float],
        portfolio_value: float,
    ) -> list[StressResult]:
        results: list[StressResult] = []
        for scenario in self._scenarios:
            result = self._apply_scenario(scenario, positions, portfolio_value)
            results.append(result)
        return results

    @staticmethod
    def _apply_scenario(
        scenario: StressScenario,
        positions: dict[str, float],
        portfolio_value: float,
    ) -> StressResult:
        per_asset: dict[str, float] = {}
        total_pnl = 0.0

        portfolio_shock = scenario.shocks.get("__portfolio__")

        for symbol, dollar_value in positions.items():
            shock = scenario.shocks.get(symbol, portfolio_shock)
            if shock is None:
                shock = 0.0
            asset_pnl = dollar_value * shock
            per_asset[symbol] = asset_pnl
            total_pnl += asset_pnl

        # If only __portfolio__ shock and no positions, apply to total value
        if not positions and portfolio_shock is not None:
            total_pnl = portfolio_value * portfolio_shock

        pct_return = total_pnl / portfolio_value if portfolio_value > 0 else 0.0

        return StressResult(
            scenario_name=scenario.name,
            portfolio_pnl=total_pnl,
            portfolio_return=pct_return,
            per_asset=per_asset,
        )

    # ── Concentration ──────────────────────────────────────────────────

    @staticmethod
    def _compute_concentration(
        positions: dict[str, float],
        portfolio_value: float,
    ) -> ConcentrationMetrics | None:
        if not positions or portfolio_value <= 0:
            return None

        abs_values = np.array([abs(v) for v in positions.values()])
        total = abs_values.sum()
        if total == 0:
            return None

        weights = abs_values / total
        weights_sorted = np.sort(weights)[::-1]

        hhi = float(np.sum(weights**2))
        effective_n = 1.0 / hhi if hhi > 0 else 0.0
        top1 = float(weights_sorted[0])
        top5 = float(weights_sorted[:5].sum())

        return ConcentrationMetrics(
            hhi=hhi,
            effective_n=effective_n,
            top1_weight=top1,
            top5_weight=top5,
            n_positions=len(positions),
        )

    # ── Volatility and drawdown ────────────────────────────────────────

    def _annualised_volatility(self, returns: pd.Series) -> float:
        if len(returns) < 2:
            return 0.0
        return float(np.std(returns, ddof=1) * np.sqrt(self._trading_days))

    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        if len(returns) < 1:
            return 0.0
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return float(-drawdown.min())


# ── Helper: inverse normal CDF approximation ─────────────────────────────────

def _inv_norm(p: float) -> float:
    """Approximate inverse of the standard normal CDF.

    Uses the Abramowitz & Stegun rational approximation (formula 26.2.23).
    Accurate to ~4.5e-4 for 0 < p < 1.
    """
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0

    if p > 0.5:
        return -_inv_norm(1.0 - p)

    t = np.sqrt(-2.0 * np.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3))
