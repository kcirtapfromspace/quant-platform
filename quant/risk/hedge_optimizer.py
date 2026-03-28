"""Portfolio hedge optimiser.

Evaluates, selects, and sizes hedges that reduce portfolio tail risk
at minimum cost.  Designed for the CIO to answer questions like:

  * "Which of these instruments best hedges my current portfolio?"
  * "How much of each hedge do I need to cut portfolio VaR by 20%?"
  * "What's the cost/benefit of hedging my top risk contributors?"

Key concepts:

  * **Minimum-variance hedge ratio**: β from regressing portfolio returns
    on hedge instrument returns (OLS or exponentially weighted).
  * **Hedge effectiveness**: R² of the hedging regression — what fraction
    of portfolio variance is explained by the hedge.
  * **Risk reduction profile**: how portfolio vol, VaR, and CVaR change
    as hedge ratio varies from 0 to full hedge.
  * **Cost-adjusted selection**: ranks candidates by risk reduction per
    unit of expected carry/drag.

Usage::

    from quant.risk.hedge_optimizer import HedgeOptimizer, HedgeConfig

    optimizer = HedgeOptimizer(HedgeConfig())
    candidates = optimizer.evaluate_candidates(
        portfolio_returns, hedge_returns, carry_costs,
    )
    for c in candidates:
        print(c.summary())
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HedgeConfig:
    """Configuration for hedge optimisation.

    Attributes:
        var_confidence:   VaR confidence level (e.g. 0.95 = 95% VaR).
        annualise:        Whether to annualise risk metrics.
        trading_days:     Trading days per year.
        n_profile_points: Number of points on the risk reduction profile.
        min_observations: Minimum overlapping returns for hedge evaluation.
    """

    var_confidence: float = 0.95
    annualise: bool = True
    trading_days: int = 252
    n_profile_points: int = 20
    min_observations: int = 60


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HedgeCandidate:
    """Evaluation of a single hedge instrument.

    Attributes:
        symbol:           Hedge instrument identifier.
        hedge_ratio:      Minimum-variance hedge ratio (β).
        effectiveness:    R² of hedging regression (0 to 1).
        correlation:      Correlation between portfolio and hedge returns.
        vol_reduction_pct: Portfolio vol reduction at full hedge ratio.
        var_reduction_pct: Portfolio VaR reduction at full hedge ratio.
        carry_cost:       Annual expected carry/drag of hedge (bps).
        risk_per_cost:    Vol reduction / carry cost — higher is better.
        n_observations:   Number of overlapping return observations.
    """

    symbol: str
    hedge_ratio: float
    effectiveness: float
    correlation: float
    vol_reduction_pct: float
    var_reduction_pct: float
    carry_cost: float
    risk_per_cost: float
    n_observations: int

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Hedge: {self.symbol}",
            f"  Hedge ratio       : {self.hedge_ratio:.4f}",
            f"  Effectiveness (R²): {self.effectiveness:.2%}",
            f"  Correlation       : {self.correlation:+.3f}",
            f"  Vol reduction     : {self.vol_reduction_pct:.1%}",
            f"  VaR reduction     : {self.var_reduction_pct:.1%}",
            f"  Carry cost (bps)  : {self.carry_cost:.1f}",
            f"  Risk/cost ratio   : {self.risk_per_cost:.2f}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RiskProfilePoint:
    """A single point on the hedge ratio vs risk curve.

    Attributes:
        hedge_fraction: Fraction of full hedge ratio applied (0 to 1).
        portfolio_vol:  Annualised portfolio vol at this hedge level.
        portfolio_var:  Portfolio VaR at this hedge level.
        portfolio_cvar: Portfolio CVaR at this hedge level.
    """

    hedge_fraction: float
    portfolio_vol: float
    portfolio_var: float
    portfolio_cvar: float


@dataclass
class RiskProfile:
    """Risk reduction profile for a specific hedge.

    Attributes:
        symbol:     Hedge instrument.
        hedge_ratio: Full minimum-variance hedge ratio.
        points:     List of profile points from 0% to 100% hedge.
    """

    symbol: str
    hedge_ratio: float
    points: list[RiskProfilePoint]

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Risk Profile: {self.symbol} (β={self.hedge_ratio:.4f})",
            "=" * 55,
            "",
            f"{'Hedge%':>8} {'Vol':>8} {'VaR':>8} {'CVaR':>8}",
            "-" * 40,
        ]
        for p in self.points:
            lines.append(
                f"{p.hedge_fraction:>7.0%} {p.portfolio_vol:>7.2%} "
                f"{p.portfolio_var:>7.2%} {p.portfolio_cvar:>7.2%}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------


class HedgeOptimizer:
    """Evaluates and ranks hedge instruments for a portfolio.

    Args:
        config: Hedge optimisation parameters.
    """

    def __init__(self, config: HedgeConfig | None = None) -> None:
        self._config = config or HedgeConfig()

    @property
    def config(self) -> HedgeConfig:
        return self._config

    def evaluate_candidates(
        self,
        portfolio_returns: pd.Series,
        hedge_returns: pd.DataFrame,
        carry_costs_bps: dict[str, float] | None = None,
    ) -> list[HedgeCandidate]:
        """Evaluate multiple hedge instruments and rank by effectiveness.

        Args:
            portfolio_returns: Daily portfolio return series.
            hedge_returns:     DataFrame with one column per hedge
                instrument, same date index as portfolio.
            carry_costs_bps:   Annual carry cost per instrument (in bps).
                Defaults to 0 for all instruments.

        Returns:
            List of :class:`HedgeCandidate` sorted by effectiveness
            (highest first).

        Raises:
            ValueError: If fewer than min_observations overlap.
        """
        carry = carry_costs_bps or {}
        candidates: list[HedgeCandidate] = []

        for col in hedge_returns.columns:
            cand = self._evaluate_single(
                portfolio_returns,
                hedge_returns[col],
                str(col),
                carry.get(str(col), 0.0),
            )
            if cand is not None:
                candidates.append(cand)

        candidates.sort(key=lambda c: c.effectiveness, reverse=True)
        return candidates

    def risk_profile(
        self,
        portfolio_returns: pd.Series,
        hedge_returns: pd.Series,
    ) -> RiskProfile:
        """Compute risk reduction profile across hedge ratios.

        Args:
            portfolio_returns: Daily portfolio return series.
            hedge_returns:     Daily hedge instrument return series.

        Returns:
            :class:`RiskProfile` with points from 0% to 100% hedge.

        Raises:
            ValueError: If fewer than min_observations overlap.
        """
        combined = pd.concat(
            [portfolio_returns.rename("p"), hedge_returns.rename("h")],
            axis=1, sort=True,
        ).dropna()

        cfg = self._config
        if len(combined) < cfg.min_observations:
            raise ValueError(
                f"Need at least {cfg.min_observations} overlapping periods, "
                f"got {len(combined)}"
            )

        p = combined["p"].values
        h = combined["h"].values
        beta = self._ols_beta(p, h)
        ann = np.sqrt(cfg.trading_days) if cfg.annualise else 1.0

        points: list[RiskProfilePoint] = []
        n_pts = cfg.n_profile_points
        for i in range(n_pts + 1):
            frac = i / n_pts
            hedged = p - frac * beta * h
            vol = float(np.std(hedged, ddof=1)) * ann
            var_val = self._historical_var(hedged, cfg.var_confidence)
            cvar_val = self._historical_cvar(hedged, cfg.var_confidence)
            points.append(RiskProfilePoint(
                hedge_fraction=frac,
                portfolio_vol=vol,
                portfolio_var=var_val,
                portfolio_cvar=cvar_val,
            ))

        return RiskProfile(
            symbol=str(hedge_returns.name or "hedge"),
            hedge_ratio=beta,
            points=points,
        )

    def optimal_hedge_size(
        self,
        portfolio_returns: pd.Series,
        hedge_returns: pd.Series,
        target_vol_reduction_pct: float = 0.20,
    ) -> float:
        """Find hedge fraction that achieves a target vol reduction.

        Uses bisection on the risk profile to find the hedge fraction
        that reduces portfolio vol by approximately the target percentage.

        Args:
            portfolio_returns: Daily portfolio return series.
            hedge_returns:     Daily hedge instrument return series.
            target_vol_reduction_pct: Desired vol reduction as fraction
                (e.g. 0.20 = 20% reduction).

        Returns:
            Hedge fraction (0 to 1) that achieves the target, or 1.0
            if the target is not achievable.
        """
        profile = self.risk_profile(portfolio_returns, hedge_returns)
        if not profile.points:
            return 0.0

        base_vol = profile.points[0].portfolio_vol
        if base_vol < 1e-15:
            return 0.0

        target_vol = base_vol * (1.0 - target_vol_reduction_pct)

        # Find first point that achieves target
        for pt in profile.points:
            if pt.portfolio_vol <= target_vol:
                return pt.hedge_fraction

        return 1.0  # full hedge insufficient

    # ── Internal ──────────────────────────────────────────────────

    def _evaluate_single(
        self,
        portfolio_returns: pd.Series,
        hedge_returns: pd.Series,
        symbol: str,
        carry_bps: float,
    ) -> HedgeCandidate | None:
        """Evaluate a single hedge instrument."""
        combined = pd.concat(
            [portfolio_returns.rename("p"), hedge_returns.rename("h")],
            axis=1, sort=True,
        ).dropna()

        cfg = self._config
        if len(combined) < cfg.min_observations:
            return None

        p = combined["p"].values
        h = combined["h"].values
        n = len(p)

        beta = self._ols_beta(p, h)
        corr = float(np.corrcoef(p, h)[0, 1])
        r_sq = corr ** 2

        ann = np.sqrt(cfg.trading_days) if cfg.annualise else 1.0

        # Vol reduction at full hedge
        unhedged_vol = float(np.std(p, ddof=1)) * ann
        hedged = p - beta * h
        hedged_vol = float(np.std(hedged, ddof=1)) * ann
        vol_reduction = (unhedged_vol - hedged_vol) / unhedged_vol if unhedged_vol > 1e-15 else 0.0

        # VaR reduction
        unhedged_var = self._historical_var(p, cfg.var_confidence)
        hedged_var = self._historical_var(hedged, cfg.var_confidence)
        var_reduction = (unhedged_var - hedged_var) / unhedged_var if unhedged_var > 1e-15 else 0.0

        # Risk per cost
        risk_per_cost = vol_reduction / (carry_bps / 10000) if carry_bps > 0 else float("inf") if vol_reduction > 0 else 0.0

        return HedgeCandidate(
            symbol=symbol,
            hedge_ratio=beta,
            effectiveness=r_sq,
            correlation=corr,
            vol_reduction_pct=vol_reduction,
            var_reduction_pct=var_reduction,
            carry_cost=carry_bps,
            risk_per_cost=risk_per_cost,
            n_observations=n,
        )

    @staticmethod
    def _ols_beta(y: np.ndarray, x: np.ndarray) -> float:
        """OLS hedge ratio: β = cov(y,x) / var(x)."""
        cov = float(np.cov(y, x, ddof=1)[0, 1])
        var_x = float(np.var(x, ddof=1))
        return cov / var_x if var_x > 1e-15 else 0.0

    @staticmethod
    def _historical_var(returns: np.ndarray, confidence: float) -> float:
        """Historical VaR (positive = loss)."""
        cutoff = np.percentile(returns, (1 - confidence) * 100)
        return float(-cutoff)

    @staticmethod
    def _historical_cvar(returns: np.ndarray, confidence: float) -> float:
        """Historical CVaR / Expected Shortfall (positive = loss)."""
        cutoff = np.percentile(returns, (1 - confidence) * 100)
        tail = returns[returns <= cutoff]
        return float(-np.mean(tail)) if len(tail) > 0 else float(-cutoff)
