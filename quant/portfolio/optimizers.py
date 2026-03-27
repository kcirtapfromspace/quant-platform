"""Portfolio optimisation methods.

Implements four standard portfolio construction approaches:
  1. Mean-Variance (Markowitz) — maximise expected return for a given risk level.
  2. Minimum Variance — minimise portfolio variance regardless of expected return.
  3. Risk Parity — equalise risk contribution from each asset.
  4. Maximum Diversification — maximise the diversification ratio.

All optimisers accept a covariance matrix and optional expected returns
(alpha scores), and return a dict of {symbol: weight}.  Constraints are
applied via :class:`~quant.portfolio.constraints.PortfolioConstraints`.
"""
from __future__ import annotations

import abc
import enum
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.portfolio.constraints import PortfolioConstraints


class OptimizationMethod(enum.Enum):
    MEAN_VARIANCE = "mean_variance"
    MINIMUM_VARIANCE = "minimum_variance"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Output of a portfolio optimisation.

    Attributes:
        weights:      Dict of {symbol: weight}.
        method:       Optimisation method used.
        risk:         Portfolio volatility (annualised).
        expected_return: Portfolio expected return (annualised), if available.
        diversification_ratio: Weighted-average vol / portfolio vol.
    """

    weights: dict[str, float]
    method: OptimizationMethod
    risk: float
    expected_return: float
    diversification_ratio: float


class BaseOptimizer(abc.ABC):
    """Abstract portfolio optimiser."""

    @abc.abstractmethod
    def optimize(
        self,
        symbols: list[str],
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None,
        constraints: PortfolioConstraints,
    ) -> OptimizationResult:
        """Compute optimal weights.

        Args:
            symbols:          Ordered list of asset symbols.
            cov_matrix:       N×N covariance matrix (annualised).
            expected_returns: Expected return per asset (annualised).
            constraints:      Portfolio constraints to enforce.

        Returns:
            OptimizationResult with weights and diagnostics.
        """


def _portfolio_stats(
    weights: np.ndarray, cov: np.ndarray, mu: np.ndarray | None
) -> tuple[float, float, float]:
    """Compute portfolio vol, return, and diversification ratio."""
    port_var = float(weights @ cov @ weights)
    port_vol = np.sqrt(max(port_var, 0.0))
    port_ret = float(weights @ mu) if mu is not None else 0.0

    asset_vols = np.sqrt(np.diag(cov))
    weighted_avg_vol = float(np.abs(weights) @ asset_vols)
    div_ratio = weighted_avg_vol / port_vol if port_vol > 1e-12 else 1.0

    return port_vol, port_ret, div_ratio


def _apply_constraints(
    raw_weights: np.ndarray,
    symbols: list[str],
    constraints: PortfolioConstraints,
) -> np.ndarray:
    """Apply bound and gross-exposure constraints via clipping and rescaling."""
    eff_min = constraints.effective_min()
    w = np.clip(raw_weights, eff_min, constraints.max_weight)

    gross = np.sum(np.abs(w))
    if gross > constraints.max_gross_exposure and gross > 1e-12:
        w *= constraints.max_gross_exposure / gross

    return w


class MeanVarianceOptimizer(BaseOptimizer):
    """Classical Markowitz mean-variance optimisation.

    Uses an analytical solution: w* = Σ⁻¹ μ (scaled to satisfy constraints).
    For ill-conditioned covariance matrices, a shrinkage ridge is applied.

    Args:
        risk_aversion: Risk aversion parameter λ.  Higher values penalise
            variance more, producing more conservative allocations.
        shrinkage: Ridge regularisation added to the covariance diagonal.
    """

    def __init__(
        self, risk_aversion: float = 1.0, shrinkage: float = 1e-6
    ) -> None:
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        self._risk_aversion = risk_aversion
        self._shrinkage = shrinkage

    def optimize(
        self,
        symbols: list[str],
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None,
        constraints: PortfolioConstraints,
    ) -> OptimizationResult:
        n = len(symbols)
        cov = cov_matrix.loc[symbols, symbols].values.astype(float)
        cov += np.eye(n) * self._shrinkage

        if expected_returns is not None:
            mu = expected_returns.reindex(symbols).fillna(0.0).values.astype(float)
        else:
            mu = np.ones(n) / n

        # Analytical: w* ∝ Σ⁻¹ μ / λ
        cov_inv = np.linalg.inv(cov)
        raw = cov_inv @ mu / self._risk_aversion

        # Normalise so |w| sums to gross exposure limit
        w = _apply_constraints(raw, symbols, constraints)

        # Renormalise to sum to 1 (or max_gross_exposure for levered)
        abs_sum = np.sum(np.abs(w))
        if abs_sum > 1e-12:
            w = w / abs_sum * constraints.max_gross_exposure
            w = _apply_constraints(w, symbols, constraints)

        vol, ret, div = _portfolio_stats(w, cov, mu)
        return OptimizationResult(
            weights=dict(zip(symbols, w.tolist(), strict=True)),
            method=OptimizationMethod.MEAN_VARIANCE,
            risk=vol,
            expected_return=ret,
            diversification_ratio=div,
        )


class MinimumVarianceOptimizer(BaseOptimizer):
    """Global minimum-variance portfolio.

    Ignores expected returns entirely; minimises portfolio variance.

    w* = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)

    Args:
        shrinkage: Ridge regularisation.
    """

    def __init__(self, shrinkage: float = 1e-6) -> None:
        self._shrinkage = shrinkage

    def optimize(
        self,
        symbols: list[str],
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None,
        constraints: PortfolioConstraints,
    ) -> OptimizationResult:
        n = len(symbols)
        cov = cov_matrix.loc[symbols, symbols].values.astype(float)
        cov += np.eye(n) * self._shrinkage

        cov_inv = np.linalg.inv(cov)
        ones = np.ones(n)
        raw = cov_inv @ ones
        raw = raw / (ones @ raw)  # normalise to sum to 1

        w = _apply_constraints(raw, symbols, constraints)

        # Renormalise
        w_sum = np.sum(w)
        if abs(w_sum) > 1e-12:
            w = w / np.sum(np.abs(w)) * constraints.max_gross_exposure
            w = _apply_constraints(w, symbols, constraints)

        mu = (
            expected_returns.reindex(symbols).fillna(0.0).values.astype(float)
            if expected_returns is not None
            else None
        )
        vol, ret, div = _portfolio_stats(w, cov, mu)
        return OptimizationResult(
            weights=dict(zip(symbols, w.tolist(), strict=True)),
            method=OptimizationMethod.MINIMUM_VARIANCE,
            risk=vol,
            expected_return=ret,
            diversification_ratio=div,
        )


class RiskParityOptimizer(BaseOptimizer):
    """Risk parity (equal risk contribution) portfolio.

    Iteratively solves for weights where each asset contributes equally to
    total portfolio risk.  Uses the Spinu (2013) closed-form approximation
    for the basic case.

    Risk contribution of asset i:  RC_i = w_i * (Σ w)_i / σ_p

    Args:
        max_iterations: Maximum iterations for the iterative solver.
        tolerance:      Convergence tolerance on risk contribution uniformity.
        shrinkage:      Ridge regularisation.
    """

    def __init__(
        self,
        max_iterations: int = 500,
        tolerance: float = 1e-8,
        shrinkage: float = 1e-6,
    ) -> None:
        self._max_iter = max_iterations
        self._tol = tolerance
        self._shrinkage = shrinkage

    def optimize(
        self,
        symbols: list[str],
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None,
        constraints: PortfolioConstraints,
    ) -> OptimizationResult:
        n = len(symbols)
        cov = cov_matrix.loc[symbols, symbols].values.astype(float)
        cov += np.eye(n) * self._shrinkage

        # Target: equal risk budget (1/n each)
        budget = np.ones(n) / n

        # Initialise with inverse-volatility weights
        vols = np.sqrt(np.diag(cov))
        w = (1.0 / vols)
        w = w / np.sum(w)

        for _ in range(self._max_iter):
            sigma_w = cov @ w
            port_var = float(w @ sigma_w)
            port_vol = np.sqrt(max(port_var, 1e-20))

            # Marginal risk contribution
            mrc = sigma_w / port_vol
            # Risk contribution
            rc = w * mrc
            rc_total = np.sum(rc)
            if rc_total < 1e-20:
                break
            rc_pct = rc / rc_total

            # Update: w_new ∝ budget / mrc
            w_new = budget / (mrc + 1e-20)
            w_new = w_new / np.sum(w_new)

            if np.max(np.abs(rc_pct - budget)) < self._tol:
                w = w_new
                break
            w = w_new

        w = _apply_constraints(w, symbols, constraints)
        # Renormalise
        abs_sum = np.sum(np.abs(w))
        if abs_sum > 1e-12:
            w = w / abs_sum * constraints.max_gross_exposure
            w = _apply_constraints(w, symbols, constraints)

        mu = (
            expected_returns.reindex(symbols).fillna(0.0).values.astype(float)
            if expected_returns is not None
            else None
        )
        vol, ret, div = _portfolio_stats(w, cov, mu)
        return OptimizationResult(
            weights=dict(zip(symbols, w.tolist(), strict=True)),
            method=OptimizationMethod.RISK_PARITY,
            risk=vol,
            expected_return=ret,
            diversification_ratio=div,
        )


class MaxDiversificationOptimizer(BaseOptimizer):
    """Maximum diversification portfolio.

    Maximises the diversification ratio:
        DR = (wᵀ σ) / √(wᵀ Σ w)
    where σ is the vector of asset volatilities.

    Equivalent to solving: w* ∝ Σ⁻¹ σ

    Args:
        shrinkage: Ridge regularisation.
    """

    def __init__(self, shrinkage: float = 1e-6) -> None:
        self._shrinkage = shrinkage

    def optimize(
        self,
        symbols: list[str],
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None,
        constraints: PortfolioConstraints,
    ) -> OptimizationResult:
        n = len(symbols)
        cov = cov_matrix.loc[symbols, symbols].values.astype(float)
        cov += np.eye(n) * self._shrinkage

        asset_vols = np.sqrt(np.diag(cov))
        cov_inv = np.linalg.inv(cov)
        raw = cov_inv @ asset_vols
        raw = raw / np.sum(np.abs(raw))  # normalise

        w = _apply_constraints(raw, symbols, constraints)
        abs_sum = np.sum(np.abs(w))
        if abs_sum > 1e-12:
            w = w / abs_sum * constraints.max_gross_exposure
            w = _apply_constraints(w, symbols, constraints)

        mu = (
            expected_returns.reindex(symbols).fillna(0.0).values.astype(float)
            if expected_returns is not None
            else None
        )
        vol, ret, div = _portfolio_stats(w, cov, mu)
        return OptimizationResult(
            weights=dict(zip(symbols, w.tolist(), strict=True)),
            method=OptimizationMethod.MAX_DIVERSIFICATION,
            risk=vol,
            expected_return=ret,
            diversification_ratio=div,
        )


def get_optimizer(method: OptimizationMethod, **kwargs) -> BaseOptimizer:
    """Factory function to create an optimiser by method name."""
    registry: dict[OptimizationMethod, type[BaseOptimizer]] = {
        OptimizationMethod.MEAN_VARIANCE: MeanVarianceOptimizer,
        OptimizationMethod.MINIMUM_VARIANCE: MinimumVarianceOptimizer,
        OptimizationMethod.RISK_PARITY: RiskParityOptimizer,
        OptimizationMethod.MAX_DIVERSIFICATION: MaxDiversificationOptimizer,
    }
    cls = registry.get(method)
    if cls is None:
        raise ValueError(f"Unknown optimization method: {method}")
    return cls(**kwargs)
