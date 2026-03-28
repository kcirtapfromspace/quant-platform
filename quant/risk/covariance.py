"""Robust covariance estimation with shrinkage methods.

Provides multiple covariance estimation approaches for portfolio optimization
and risk management.  Sample covariance is poorly conditioned when the ratio
p/T is non-trivial; shrinkage estimators trade a small bias for dramatically
lower estimation error.

Methods:

  * **Sample** — standard unbiased sample covariance (baseline).
  * **Ledoit-Wolf** — linear shrinkage toward scaled identity μI.
  * **OAS** — Oracle Approximating Shrinkage (Chen et al. 2010).
  * **Exponential** — exponentially weighted with configurable half-life.

Key outputs:

  * Covariance and correlation matrices (optionally annualised).
  * Shrinkage intensity for shrinkage methods.
  * Condition number for numerical quality assessment.

Usage::

    from quant.risk.covariance import (
        CovarianceEstimator,
        CovarianceConfig,
        EstimationMethod,
    )

    estimator = CovarianceEstimator(CovarianceConfig(
        method=EstimationMethod.LEDOIT_WOLF,
    ))
    result = estimator.estimate(daily_returns_df)
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EstimationMethod(Enum):
    """Covariance estimation method."""

    SAMPLE = auto()
    LEDOIT_WOLF = auto()
    OAS = auto()
    EXPONENTIAL = auto()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CovarianceConfig:
    """Configuration for covariance estimation.

    Attributes:
        method:           Estimation method.
        min_observations: Minimum rows required.
        halflife:         Half-life in periods for exponential weighting.
        annualise:        If True, scale covariance by ``trading_days``.
        trading_days:     Annualisation factor.
    """

    method: EstimationMethod = EstimationMethod.LEDOIT_WOLF
    min_observations: int = 30
    halflife: int = 63
    annualise: bool = True
    trading_days: int = TRADING_DAYS


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class CovarianceResult:
    """Covariance estimation result.

    Attributes:
        covariance:          N x N covariance matrix.
        correlation:         N x N correlation matrix.
        volatilities:        Per-asset volatility (annualised if configured).
        method:              Estimation method used.
        shrinkage_intensity: Shrinkage coefficient in [0, 1].  0 for
                             non-shrinkage methods.
        condition_number:    Ratio of largest to smallest eigenvalue.
        n_observations:      Number of return observations used.
        n_assets:            Number of assets.
    """

    covariance: pd.DataFrame
    correlation: pd.DataFrame
    volatilities: pd.Series
    method: EstimationMethod
    shrinkage_intensity: float
    condition_number: float
    n_observations: int
    n_assets: int

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Covariance Estimation ({self.method.name})",
            "=" * 60,
            "",
            f"Assets         : {self.n_assets}",
            f"Observations   : {self.n_observations}",
            f"Condition #    : {self.condition_number:.1f}",
            f"Shrinkage      : {self.shrinkage_intensity:.4f}",
            "",
            "Volatilities (top 5 by vol):",
        ]

        top = self.volatilities.sort_values(ascending=False).head(5)
        for sym, vol in top.items():
            lines.append(f"  {sym:<12s}: {vol:.2%}")
        if self.n_assets > 5:
            lines.append(f"  ... and {self.n_assets - 5} more")

        lines.append("")
        lines.append("Correlation range:")
        corr_vals = self.correlation.values
        mask = ~np.eye(self.n_assets, dtype=bool)
        if mask.any():
            off_diag = corr_vals[mask]
            lines.append(f"  Min  : {off_diag.min():+.3f}")
            lines.append(f"  Max  : {off_diag.max():+.3f}")
            lines.append(f"  Mean : {off_diag.mean():+.3f}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class CovarianceEstimator:
    """Robust covariance estimator with multiple methods.

    Args:
        config: Estimation configuration.
    """

    def __init__(self, config: CovarianceConfig | None = None) -> None:
        self._config = config or CovarianceConfig()

    @property
    def config(self) -> CovarianceConfig:
        return self._config

    def estimate(self, returns: pd.DataFrame) -> CovarianceResult:
        """Estimate covariance matrix from a returns DataFrame.

        Args:
            returns: T x N DataFrame of asset returns (rows = dates,
                     columns = assets).

        Returns:
            :class:`CovarianceResult` with covariance, correlation, and
            diagnostics.

        Raises:
            ValueError: If fewer than ``min_observations`` rows or < 2 assets.
        """
        r = returns.dropna()
        n_obs, n_cols = r.shape

        if self._config.min_observations > n_obs:
            msg = (
                f"Need at least {self._config.min_observations} observations, "
                f"got {n_obs}"
            )
            raise ValueError(msg)
        if n_cols < 2:
            raise ValueError(f"Need at least 2 assets, got {n_cols}")

        x_mat = r.values.astype(np.float64)
        symbols = list(r.columns)

        method = self._config.method
        _dispatch = {
            EstimationMethod.SAMPLE: self._sample,
            EstimationMethod.LEDOIT_WOLF: self._ledoit_wolf,
            EstimationMethod.OAS: self._oas,
            EstimationMethod.EXPONENTIAL: self._exponential,
        }
        estimator_fn = _dispatch.get(method)
        if estimator_fn is None:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)

        cov, shrinkage = estimator_fn(x_mat)

        # Annualise
        if self._config.annualise:
            cov = cov * self._config.trading_days

        # Volatilities and correlation
        vols = np.sqrt(np.diag(cov))
        inv_vols = np.where(vols > 1e-15, 1.0 / vols, 0.0)
        d_inv = np.diag(inv_vols)
        corr = d_inv @ cov @ d_inv
        np.fill_diagonal(corr, 1.0)
        corr = (corr + corr.T) / 2.0  # enforce symmetry

        # Condition number
        eigvals = np.linalg.eigvalsh(cov)
        cond = float(eigvals[-1] / max(abs(eigvals[0]), 1e-15))

        return CovarianceResult(
            covariance=pd.DataFrame(cov, index=symbols, columns=symbols),
            correlation=pd.DataFrame(corr, index=symbols, columns=symbols),
            volatilities=pd.Series(vols, index=symbols, name="volatility"),
            method=method,
            shrinkage_intensity=float(shrinkage),
            condition_number=cond,
            n_observations=n_obs,
            n_assets=n_cols,
        )

    # ── Estimation methods ─────────────────────────────────────────

    @staticmethod
    def _sample(x_mat: np.ndarray) -> tuple[np.ndarray, float]:
        """Standard sample covariance with Bessel correction (ddof=1)."""
        cov = np.cov(x_mat, rowvar=False, ddof=1)
        return cov, 0.0

    @staticmethod
    def _ledoit_wolf(x_mat: np.ndarray) -> tuple[np.ndarray, float]:
        """Ledoit-Wolf linear shrinkage toward scaled identity.

        Implements Ledoit & Wolf (2004) "A well-conditioned estimator
        for large-dimensional covariance matrices" with target F = μI.

        The shrinkage intensity is computed with 1/n normalisation (as
        published), then the final estimator is rescaled to ddof=1.
        """
        n, p = x_mat.shape
        xc = x_mat - x_mat.mean(axis=0)

        # Sample covariance — 1/n normalisation for LW formulae
        s = xc.T @ xc / n

        # Target: μI where μ = average eigenvalue
        mu = np.trace(s) / p

        # Squared Frobenius distance  ||S − μI||² / p
        delta = s.copy()
        np.fill_diagonal(delta, delta.diagonal() - mu)
        delta_sq = np.sum(delta ** 2) / p

        if delta_sq < 1e-15:
            # Already proportional to identity — shrink fully
            return s * n / (n - 1), 1.0

        # β — estimation variability
        # β = (1/(n²p)) Σ_t ||x_t x_t' − S||²_F
        #   = (1/(n²p)) [Σ_t (||x_t||²)² − n ||S||²_F]
        x2 = xc ** 2
        norms_sq = np.sum(x2, axis=1)
        beta = (np.sum(norms_sq ** 2) / n - np.sum(s ** 2)) / (n * p)
        beta = max(beta, 0.0)

        # Optimal shrinkage intensity
        shrinkage = min(beta / delta_sq, 1.0)

        # Shrunk estimator, rescaled to ddof=1
        cov = (1.0 - shrinkage) * s + shrinkage * mu * np.eye(p)
        cov *= n / (n - 1)

        return cov, shrinkage

    @staticmethod
    def _oas(x_mat: np.ndarray) -> tuple[np.ndarray, float]:
        """Oracle Approximating Shrinkage toward scaled identity.

        Implements Chen, Wiesel, Eldar & Hero (2010) with a closed-form
        shrinkage intensity that approximates the oracle.  Uses 1/n
        normalisation internally, rescaled to ddof=1 for the output.
        """
        n, p = x_mat.shape
        xc = x_mat - x_mat.mean(axis=0)
        s = xc.T @ xc / n

        mu = np.trace(s) / p
        tr_s2 = np.sum(s ** 2)       # tr(S²) = ||S||²_F for symmetric S
        tr_s_sq = np.trace(s) ** 2    # (tr S)²

        # OAS shrinkage (Theorem 1 of Chen et al. 2010)
        rho_num = (1.0 - 2.0 / p) * tr_s2 + tr_s_sq
        rho_den = (n + 1.0 - 2.0 / p) * (tr_s2 - tr_s_sq / p)

        shrinkage = (
            1.0 if abs(rho_den) < 1e-15
            else max(0.0, min(rho_num / rho_den, 1.0))
        )

        cov = (1.0 - shrinkage) * s + shrinkage * mu * np.eye(p)
        cov *= n / (n - 1)

        return cov, shrinkage

    def _exponential(self, x_mat: np.ndarray) -> tuple[np.ndarray, float]:
        """Exponentially weighted covariance matrix.

        Uses an exponential decay kernel with the configured half-life.
        More recent observations receive higher weight, adapting the
        estimate to regime changes.
        """
        n, _p = x_mat.shape
        halflife = self._config.halflife

        # Decay factor
        lam = 0.5 ** (1.0 / halflife)

        # Weights: w_t = λ^(n−1−t), newest observation gets w = 1
        exponents = np.arange(n - 1, -1, -1, dtype=np.float64)
        weights = lam ** exponents
        weights /= weights.sum()

        # Weighted mean
        w_mean = weights @ x_mat

        # Centred observations
        xc = x_mat - w_mean

        # Weighted covariance: Σ w_t · xc_t · xc_t'
        xw = xc * np.sqrt(weights)[:, np.newaxis]
        cov = xw.T @ xw

        # Bias correction (weighted Bessel): 1 / (1 − Σ w²)
        sum_w2 = np.sum(weights ** 2)
        if sum_w2 < 1.0 - 1e-15:
            cov /= 1.0 - sum_w2

        return cov, 0.0
