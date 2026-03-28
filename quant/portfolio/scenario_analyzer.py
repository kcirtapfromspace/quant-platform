"""Portfolio scenario (what-if) analyzer.

Evaluates hypothetical portfolio changes against a covariance matrix to
answer questions like "what happens to risk if we add/remove a position,
shift weight, or face a specific market shock?"

Scenario types:

  * **Weight change** — move weight from one position to another.
  * **Add / remove position** — introduce or exit an asset.
  * **Market shock** — apply a user-defined return vector and measure P&L.
  * **Volatility shock** — scale the covariance matrix and recompute risk.

Key outputs:

  * Before / after portfolio volatility and tracking error.
  * Marginal risk contribution of affected positions.
  * Shock P&L and conditional tail impact.

Usage::

    from quant.portfolio.scenario_analyzer import (
        ScenarioAnalyzer,
        ScenarioConfig,
        WeightChangeScenario,
    )

    analyzer = ScenarioAnalyzer(covariance_matrix)
    result = analyzer.weight_change(
        current_weights, {"AAPL": 0.05, "MSFT": -0.03},
    )
    print(result.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    """Configuration for scenario analysis.

    Attributes:
        annualise:        If True, report annualised volatility.
        trading_days:     Annualisation factor.
        var_confidence:   VaR confidence level for shock analysis.
    """

    annualise: bool = True
    trading_days: int = TRADING_DAYS
    var_confidence: float = 0.95


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RiskDelta:
    """Before/after risk comparison for a single scenario.

    Attributes:
        vol_before:       Portfolio volatility before change.
        vol_after:        Portfolio volatility after change.
        vol_change:       Absolute change in volatility.
        vol_change_pct:   Percentage change in volatility.
        tracking_error:   Volatility of the weight-change vector
                          (annualised if configured).
    """

    vol_before: float
    vol_after: float
    vol_change: float
    vol_change_pct: float
    tracking_error: float


@dataclass(frozen=True, slots=True)
class MarginalRisk:
    """Marginal risk contribution of a position.

    Attributes:
        symbol:          Asset symbol.
        weight:          Portfolio weight.
        marginal_vol:    Marginal contribution to portfolio volatility
                         (∂σ_p / ∂w_i).
        risk_contrib:    Risk contribution: w_i × marginal_vol.
    """

    symbol: str
    weight: float
    marginal_vol: float
    risk_contrib: float


@dataclass
class ScenarioResult:
    """Complete scenario analysis result.

    Attributes:
        name:             Human-readable scenario description.
        risk_delta:       Before / after risk comparison.
        marginal_risks:   Marginal risk for affected positions (after).
        weights_before:   Weights before change.
        weights_after:    Weights after change.
    """

    name: str
    risk_delta: RiskDelta
    marginal_risks: list[MarginalRisk] = field(default_factory=list)
    weights_before: dict[str, float] = field(default_factory=dict)
    weights_after: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary."""
        rd = self.risk_delta
        lines = [
            f"Scenario: {self.name}",
            "=" * 60,
            "",
            f"Vol before     : {rd.vol_before:.2%}",
            f"Vol after      : {rd.vol_after:.2%}",
            f"Vol change     : {rd.vol_change:+.2%} ({rd.vol_change_pct:+.1%})",
            f"Tracking error : {rd.tracking_error:.2%}",
        ]

        if self.marginal_risks:
            lines.append("")
            lines.append("Marginal risk (affected positions):")
            for mr in sorted(
                self.marginal_risks, key=lambda x: abs(x.risk_contrib), reverse=True,
            ):
                lines.append(
                    f"  {mr.symbol:<10s}: w={mr.weight:+.4f}  "
                    f"MRC={mr.marginal_vol:+.4f}  RC={mr.risk_contrib:+.6f}",
                )

        return "\n".join(lines)


@dataclass
class ShockResult:
    """Result of a market shock scenario.

    Attributes:
        name:           Scenario description.
        portfolio_pnl:  P&L as fraction of portfolio value.
        position_pnls:  Per-position P&L contribution.
        vol_before:     Portfolio vol before shock (for context).
    """

    name: str
    portfolio_pnl: float
    position_pnls: dict[str, float] = field(default_factory=dict)
    vol_before: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Shock Scenario: {self.name}",
            "=" * 60,
            "",
            f"Portfolio P&L  : {self.portfolio_pnl:+.2%}",
            f"Portfolio vol  : {self.vol_before:.2%}",
            "",
            "Position P&L (top contributors):",
        ]

        sorted_pnl = sorted(
            self.position_pnls.items(), key=lambda x: abs(x[1]), reverse=True,
        )
        for sym, pnl in sorted_pnl[:10]:
            lines.append(f"  {sym:<10s}: {pnl:+.4%}")
        if len(sorted_pnl) > 10:
            lines.append(f"  ... and {len(sorted_pnl) - 10} more")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class ScenarioAnalyzer:
    """Portfolio scenario (what-if) analyzer.

    All scenarios require a covariance matrix that covers the universe of
    assets appearing in any portfolio weight or shock vector.

    Args:
        covariance: N x N covariance matrix as pd.DataFrame (indexed by
                    asset symbols).  Should be annualised if ``config.annualise``
                    is False, or daily if True (the analyzer will annualise).
        config:     Analysis configuration.
    """

    def __init__(
        self,
        covariance: pd.DataFrame,
        config: ScenarioConfig | None = None,
    ) -> None:
        self._cov = covariance.values.astype(np.float64)
        self._symbols = list(covariance.columns)
        self._sym_idx = {s: i for i, s in enumerate(self._symbols)}
        self._config = config or ScenarioConfig()

    @property
    def config(self) -> ScenarioConfig:
        return self._config

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    # ── Weight change scenario ────────────────────────────────────

    def weight_change(
        self,
        current_weights: dict[str, float],
        changes: dict[str, float],
        name: str = "Weight change",
    ) -> ScenarioResult:
        """Evaluate the impact of changing portfolio weights.

        Args:
            current_weights: Current portfolio weights ``{symbol: weight}``.
            changes:         Weight changes ``{symbol: delta_weight}``.
                             Positive = increase, negative = decrease.
            name:            Scenario description.

        Returns:
            :class:`ScenarioResult` with risk comparison and marginal risks.
        """
        new_weights = dict(current_weights)
        for sym, dw in changes.items():
            new_weights[sym] = new_weights.get(sym, 0.0) + dw

        w_before = self._weights_to_vector(current_weights)
        w_after = self._weights_to_vector(new_weights)
        w_delta = w_after - w_before

        vol_before = self._portfolio_vol(w_before)
        vol_after = self._portfolio_vol(w_after)
        te = self._portfolio_vol(w_delta)

        vol_change = vol_after - vol_before
        vol_pct = vol_change / vol_before if vol_before > 1e-15 else 0.0

        # Marginal risk for affected positions
        affected = set(changes.keys())
        marginals = self._marginal_risks(w_after, affected)

        return ScenarioResult(
            name=name,
            risk_delta=RiskDelta(
                vol_before=vol_before,
                vol_after=vol_after,
                vol_change=vol_change,
                vol_change_pct=vol_pct,
                tracking_error=te,
            ),
            marginal_risks=marginals,
            weights_before=dict(current_weights),
            weights_after={s: w for s, w in new_weights.items() if abs(w) > 1e-15},
        )

    # ── Add position scenario ─────────────────────────────────────

    def add_position(
        self,
        current_weights: dict[str, float],
        symbol: str,
        weight: float,
        fund_from: str | None = None,
    ) -> ScenarioResult:
        """Evaluate adding a new position.

        Args:
            current_weights: Current portfolio weights.
            symbol:          Asset to add.
            weight:          Target weight for the new position.
            fund_from:       If provided, reduce this position's weight
                             to fund the new position.

        Returns:
            :class:`ScenarioResult` with risk comparison.
        """
        changes: dict[str, float] = {symbol: weight}
        if fund_from is not None:
            changes[fund_from] = -weight
        return self.weight_change(
            current_weights, changes, name=f"Add {symbol} @ {weight:.2%}",
        )

    # ── Remove position scenario ──────────────────────────────────

    def remove_position(
        self,
        current_weights: dict[str, float],
        symbol: str,
        redistribute: bool = False,
    ) -> ScenarioResult:
        """Evaluate removing a position.

        Args:
            current_weights: Current portfolio weights.
            symbol:          Asset to remove.
            redistribute:    If True, redistribute the removed weight
                             pro-rata across remaining positions.

        Returns:
            :class:`ScenarioResult` with risk comparison.
        """
        w = current_weights.get(symbol, 0.0)
        changes: dict[str, float] = {symbol: -w}

        if redistribute and abs(w) > 1e-15:
            remaining = {
                s: v for s, v in current_weights.items()
                if s != symbol and abs(v) > 1e-15
            }
            total_remaining = sum(remaining.values())
            if abs(total_remaining) > 1e-15:
                for s, v in remaining.items():
                    changes[s] = changes.get(s, 0.0) + w * (v / total_remaining)

        return self.weight_change(
            current_weights, changes, name=f"Remove {symbol}",
        )

    # ── Market shock scenario ─────────────────────────────────────

    def market_shock(
        self,
        weights: dict[str, float],
        shock_returns: dict[str, float],
        name: str = "Market shock",
    ) -> ShockResult:
        """Evaluate portfolio impact of a market shock.

        Args:
            weights:        Current portfolio weights.
            shock_returns:  Hypothetical returns ``{symbol: return}``.
                            E.g. {"SPY": -0.05} for a 5% drop.
            name:           Scenario description.

        Returns:
            :class:`ShockResult` with portfolio and position P&L.
        """
        w_vec = self._weights_to_vector(weights)
        vol = self._portfolio_vol(w_vec)

        # Position-level P&L
        pnls: dict[str, float] = {}
        total_pnl = 0.0
        for sym, w in weights.items():
            ret = shock_returns.get(sym, 0.0)
            pnl = w * ret
            if abs(pnl) > 1e-15:
                pnls[sym] = pnl
            total_pnl += pnl

        return ShockResult(
            name=name,
            portfolio_pnl=total_pnl,
            position_pnls=pnls,
            vol_before=vol,
        )

    # ── Volatility shock scenario ─────────────────────────────────

    def vol_shock(
        self,
        weights: dict[str, float],
        vol_multiplier: float,
        name: str | None = None,
    ) -> ScenarioResult:
        """Evaluate portfolio risk under scaled volatility.

        Scales the entire covariance matrix by ``vol_multiplier²`` and
        recomputes portfolio risk.

        Args:
            weights:         Current portfolio weights.
            vol_multiplier:  Volatility scaling factor (e.g. 1.5 = 50%
                             increase in all volatilities).
            name:            Scenario description.

        Returns:
            :class:`ScenarioResult` with before/after risk.
        """
        if name is None:
            name = f"Vol shock x{vol_multiplier:.1f}"

        w_vec = self._weights_to_vector(weights)
        vol_before = self._portfolio_vol(w_vec)

        # Scale covariance
        vol_after = vol_before * vol_multiplier

        vol_change = vol_after - vol_before
        vol_pct = vol_change / vol_before if vol_before > 1e-15 else 0.0

        return ScenarioResult(
            name=name,
            risk_delta=RiskDelta(
                vol_before=vol_before,
                vol_after=vol_after,
                vol_change=vol_change,
                vol_change_pct=vol_pct,
                tracking_error=0.0,
            ),
            weights_before=dict(weights),
            weights_after=dict(weights),
        )

    # ── Internal helpers ──────────────────────────────────────────

    def _weights_to_vector(self, weights: dict[str, float]) -> np.ndarray:
        """Convert weight dict to numpy vector aligned with covariance."""
        w = np.zeros(len(self._symbols))
        for sym, val in weights.items():
            idx = self._sym_idx.get(sym)
            if idx is not None:
                w[idx] = val
        return w

    def _portfolio_vol(self, w: np.ndarray) -> float:
        """Compute portfolio volatility from weight vector."""
        var = float(w @ self._cov @ w)
        if var < 0:
            var = 0.0
        vol = math.sqrt(var)
        if self._config.annualise:
            vol *= math.sqrt(self._config.trading_days)
        return vol

    def _marginal_risks(
        self,
        w: np.ndarray,
        symbols: set[str],
    ) -> list[MarginalRisk]:
        """Compute marginal risk contributions for selected symbols."""
        port_var = float(w @ self._cov @ w)
        if port_var < 1e-30:
            return []

        port_vol = math.sqrt(port_var)
        ann = math.sqrt(self._config.trading_days) if self._config.annualise else 1.0

        # Marginal vol: (Σw)_i / σ_p
        sigma_w = self._cov @ w
        results = []
        for sym in symbols:
            idx = self._sym_idx.get(sym)
            if idx is None:
                continue
            mrc = float(sigma_w[idx]) / port_vol * ann
            rc = w[idx] * mrc
            results.append(MarginalRisk(
                symbol=sym,
                weight=float(w[idx]),
                marginal_vol=mrc,
                risk_contrib=rc,
            ))
        return results
