"""Dynamic capital allocation across strategy sleeves.

Provides risk-aware alternatives to static or equal-weight sleeve
capital allocation.  Plugs into the multi-strategy backtester's
rebalance loop and the orchestrator's capital weight assignment.

Three allocation methods:

  * **Equal weight**: ``1/N`` baseline.
  * **Inverse volatility**: allocate more capital to lower-vol strategies
    so each sleeve contributes roughly equal risk.
  * **Risk parity**: iterative equal risk contribution — each sleeve's
    marginal contribution to total portfolio risk is equalised.

Usage::

    from quant.portfolio.capital_allocator import (
        CapitalAllocator,
        AllocationConfig,
    )

    allocator = CapitalAllocator(AllocationConfig(method="inv_vol"))
    weights = allocator.allocate(sleeve_returns)
    # weights = {"momentum": 0.35, "mean_rev": 0.65}
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class AllocationMethod(enum.Enum):
    """Sleeve capital allocation method."""

    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOL = "inv_vol"
    RISK_PARITY = "risk_parity"


@dataclass
class AllocationConfig:
    """Configuration for dynamic capital allocation.

    Attributes:
        method: Allocation method.
        vol_lookback: Days of recent returns for vol estimation.
        min_weight: Floor weight per sleeve (prevents zero allocation).
        max_weight: Ceiling weight per sleeve.
        risk_parity_iterations: Max iterations for risk parity solver.
        risk_parity_tol: Convergence tolerance for risk parity.
    """

    method: AllocationMethod = AllocationMethod.INVERSE_VOL
    vol_lookback: int = 63
    min_weight: float = 0.05
    max_weight: float = 0.80
    risk_parity_iterations: int = 100
    risk_parity_tol: float = 1e-8


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """Result of a capital allocation computation.

    Attributes:
        weights: {sleeve_name: capital_weight} summing to 1.0.
        sleeve_vols: {sleeve_name: annualized_vol}.
        method: Method used.
    """

    weights: dict[str, float]
    sleeve_vols: dict[str, float]
    method: AllocationMethod


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------


class CapitalAllocator:
    """Dynamic capital allocator for multi-strategy portfolios."""

    def __init__(self, config: AllocationConfig | None = None) -> None:
        self._config = config or AllocationConfig()

    @property
    def config(self) -> AllocationConfig:
        return self._config

    def allocate(
        self,
        sleeve_returns: dict[str, pd.Series],
    ) -> AllocationResult:
        """Compute capital weights from recent sleeve returns.

        Args:
            sleeve_returns: {sleeve_name: daily_returns_series}.

        Returns:
            :class:`AllocationResult` with normalised weights.
        """
        names = list(sleeve_returns.keys())
        n = len(names)

        if n == 0:
            return AllocationResult(
                weights={},
                sleeve_vols={},
                method=self._config.method,
            )

        if n == 1:
            return AllocationResult(
                weights={names[0]: 1.0},
                sleeve_vols={names[0]: self._estimate_vol(sleeve_returns[names[0]])},
                method=self._config.method,
            )

        # Estimate per-sleeve volatility
        vols = {name: self._estimate_vol(rets) for name, rets in sleeve_returns.items()}

        method = self._config.method

        if method == AllocationMethod.EQUAL_WEIGHT:
            raw = dict.fromkeys(names, 1.0 / n)
        elif method == AllocationMethod.INVERSE_VOL:
            raw = self._inverse_vol(names, vols)
        elif method == AllocationMethod.RISK_PARITY:
            raw = self._risk_parity(names, vols, sleeve_returns)
        else:
            raw = dict.fromkeys(names, 1.0 / n)

        # Apply min/max bounds and re-normalise
        clamped = self._clamp_and_normalise(raw)

        return AllocationResult(
            weights=clamped,
            sleeve_vols=vols,
            method=method,
        )

    # ── Allocation methods ────────────────────────────────────────────

    @staticmethod
    def _inverse_vol(
        names: list[str], vols: dict[str, float]
    ) -> dict[str, float]:
        """Allocate inversely proportional to volatility."""
        inv = {}
        for name in names:
            v = vols[name]
            inv[name] = 1.0 / v if v > 1e-12 else 1e6  # cap for near-zero vol
        total = sum(inv.values())
        return {name: inv[name] / total for name in names}

    def _risk_parity(
        self,
        names: list[str],
        vols: dict[str, float],
        sleeve_returns: dict[str, pd.Series],
    ) -> dict[str, float]:
        """Equal risk contribution via iterative solver.

        Uses the Newton-style approach: at each iteration, adjust
        weights proportionally to the gap between target and actual
        risk contribution.
        """
        n = len(names)
        lookback = self._config.vol_lookback

        # Build covariance matrix from recent returns
        rets_df = pd.DataFrame({
            name: sleeve_returns[name].iloc[-lookback:]
            for name in names
        }).dropna()

        if len(rets_df) < 5:
            # Not enough data — fall back to inverse vol
            return self._inverse_vol(names, vols)

        cov = rets_df.cov().values * TRADING_DAYS_PER_YEAR
        n_assets = cov.shape[0]

        # Initialise with inverse-vol weights
        w = np.array([1.0 / max(vols[name], 1e-12) for name in names])
        w = w / w.sum()

        target_rc = 1.0 / n_assets

        for _ in range(self._config.risk_parity_iterations):
            sigma_w = cov @ w
            port_vol = float(np.sqrt(w @ sigma_w))
            if port_vol < 1e-12:
                break

            # Marginal risk contribution
            mrc = sigma_w / port_vol
            # Risk contribution (weight × marginal)
            rc = w * mrc
            total_rc = rc.sum()
            if abs(total_rc) < 1e-12:
                break

            rc_pct = rc / total_rc

            # Check convergence
            if np.max(np.abs(rc_pct - target_rc)) < self._config.risk_parity_tol:
                break

            # Update: scale inversely to over/under contribution
            adjustment = target_rc / np.maximum(rc_pct, 1e-12)
            w = w * adjustment
            w = np.maximum(w, 1e-12)
            w = w / w.sum()

        return {names[i]: float(w[i]) for i in range(n)}

    def _clamp_and_normalise(
        self, weights: dict[str, float]
    ) -> dict[str, float]:
        """Apply min/max bounds and re-normalise to sum to 1.0.

        Iterates clamping until all weights respect bounds after
        normalization.
        """
        cfg = self._config
        w = dict(weights)
        for _ in range(20):
            total = sum(w.values())
            if total > 0:
                w = {name: v / total for name, v in w.items()}
            else:
                n = len(w)
                return dict.fromkeys(w, 1.0 / n)

            violated = False
            for name in w:
                if w[name] < cfg.min_weight:
                    w[name] = cfg.min_weight
                    violated = True
                elif w[name] > cfg.max_weight:
                    w[name] = cfg.max_weight
                    violated = True
            if not violated:
                return w

        # Final normalization
        total = sum(w.values())
        if total > 0:
            return {name: v / total for name, v in w.items()}
        n = len(w)
        return dict.fromkeys(w, 1.0 / n)

    def _estimate_vol(self, returns: pd.Series) -> float:
        """Annualized volatility from recent returns."""
        lookback = self._config.vol_lookback
        recent = returns.iloc[-lookback:] if len(returns) > lookback else returns
        if len(recent) < 2:
            return 0.0
        return float(recent.std() * math.sqrt(TRADING_DAYS_PER_YEAR))
