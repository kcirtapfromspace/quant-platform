"""Statistical factor risk model for covariance estimation.

Estimates the asset covariance matrix via a PCA-based factor model, producing
more stable and lower-noise estimates than the raw sample covariance.  The
output plugs directly into the portfolio optimiser interface
(:class:`~quant.portfolio.optimizers.BaseOptimizer`) as the ``cov_matrix``
parameter.

Model decomposition::

    Σ = B · F · Bᵀ + D

Where:
  * **B** (N × K) — factor loadings (eigenvectors scaled by √eigenvalue).
  * **F** (K × K) — factor covariance matrix (identity in PCA space).
  * **D** (N × N) — diagonal specific (idiosyncratic) variance matrix.

Features:
  * PCA-based factor extraction from asset return cross-section.
  * Automatic factor count selection via explained-variance threshold.
  * Ledoit-Wolf shrinkage for numerical stability.
  * Systematic vs idiosyncratic risk decomposition per asset.
  * Full covariance reconstruction as a ``pd.DataFrame``.
  * Annualisation to match optimiser expectations (252 trading days).

Usage::

    from quant.risk.factor_model import FactorRiskModel, FactorModelConfig

    model = FactorRiskModel(FactorModelConfig(n_factors=5))
    result = model.estimate(daily_returns)
    cov = result.covariance  # pd.DataFrame, plug into optimiser
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FactorModelConfig:
    """Configuration for the statistical factor risk model.

    Attributes:
        n_factors:              Number of PCA factors to extract.  If ``None``,
                                auto-select using ``variance_threshold``.
        variance_threshold:     Minimum cumulative explained variance for
                                auto factor selection (0–1).  Ignored if
                                ``n_factors`` is set.
        max_factors:            Hard upper bound on factor count (safety cap).
        shrinkage_intensity:    Ledoit-Wolf shrinkage intensity (0 = sample,
                                1 = structured target).  If ``None``, computed
                                automatically via the Ledoit-Wolf formula.
        min_observations:       Minimum number of return observations needed.
        annualise:              Whether to annualise the output covariance.
        demean:                 Whether to demean returns before PCA.
    """

    n_factors: int | None = None
    variance_threshold: float = 0.80
    max_factors: int = 20
    shrinkage_intensity: float | None = None
    min_observations: int = 60
    annualise: bool = True
    demean: bool = True


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactorInfo:
    """Description of one extracted factor."""

    index: int
    eigenvalue: float
    explained_variance_ratio: float
    cumulative_variance_ratio: float


@dataclass
class FactorModelResult:
    """Complete factor risk model estimation.

    Attributes:
        covariance:             Estimated N×N covariance matrix (pd.DataFrame).
        loadings:               N×K factor loading matrix (pd.DataFrame).
        factor_covariance:      K×K factor covariance matrix.
        specific_variance:      Per-asset idiosyncratic variance (pd.Series).
        factors:                Metadata for each extracted factor.
        n_assets:               Number of assets.
        n_factors:              Number of factors used.
        n_observations:         Number of return observations.
        total_variance_explained: Fraction of total variance captured by factors.
        systematic_risk_pct:    Per-asset fraction of risk that is systematic.
        shrinkage_intensity:    Applied shrinkage intensity.
        symbols:                Asset symbols (ordered).
    """

    covariance: pd.DataFrame = field(repr=False)
    loadings: pd.DataFrame = field(repr=False)
    factor_covariance: np.ndarray = field(repr=False)
    specific_variance: pd.Series = field(repr=False)
    factors: list[FactorInfo] = field(repr=False)
    n_assets: int = 0
    n_factors: int = 0
    n_observations: int = 0
    total_variance_explained: float = 0.0
    systematic_risk_pct: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), repr=False)
    shrinkage_intensity: float = 0.0
    symbols: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Factor Risk Model ({self.n_assets} assets, {self.n_factors} factors, "
            f"{self.n_observations} obs)",
            "=" * 70,
            "",
            f"Total variance explained : {self.total_variance_explained:.1%}",
            f"Shrinkage intensity      : {self.shrinkage_intensity:.3f}",
            "",
            f"{'Factor':>8s} {'Eigenvalue':>12s} {'Var%':>8s} {'Cum%':>8s}",
            "-" * 40,
        ]
        for f in self.factors:
            lines.append(
                f"{f.index:>8d} {f.eigenvalue:>12.4f} "
                f"{f.explained_variance_ratio:>7.1%} "
                f"{f.cumulative_variance_ratio:>7.1%}"
            )

        # Top/bottom systematic risk assets
        if len(self.systematic_risk_pct) > 0:
            sorted_sys = self.systematic_risk_pct.sort_values(ascending=False)
            lines.extend(["", "Systematic risk fraction (top 5):"])
            for sym, val in sorted_sys.head(5).items():
                lines.append(f"  {sym:<12s}: {val:.1%}")
            if len(sorted_sys) > 5:
                lines.append("  ...")
                for sym, val in sorted_sys.tail(3).items():
                    lines.append(f"  {sym:<12s}: {val:.1%}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class FactorRiskModel:
    """Statistical factor risk model using PCA decomposition.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: FactorModelConfig | None = None) -> None:
        self._config = config or FactorModelConfig()

    @property
    def config(self) -> FactorModelConfig:
        return self._config

    def estimate(self, returns: pd.DataFrame) -> FactorModelResult:
        """Estimate the factor risk model from asset returns.

        Args:
            returns: Daily asset returns (DatetimeIndex × symbols).

        Returns:
            :class:`FactorModelResult` with covariance and decomposition.

        Raises:
            ValueError: If fewer than ``min_observations`` rows or < 2 assets.
        """
        cfg = self._config

        returns = returns.dropna(how="all")
        n_obs, n_assets = returns.shape

        if n_obs < cfg.min_observations:
            raise ValueError(
                f"Need at least {cfg.min_observations} observations, "
                f"got {n_obs}"
            )
        if n_assets < 2:
            raise ValueError(
                f"Need at least 2 assets, got {n_assets}"
            )

        symbols = list(returns.columns)
        data = returns.fillna(0.0).values.astype(np.float64)

        # Demean
        if cfg.demean:
            data = data - data.mean(axis=0)

        # Sample covariance
        sample_cov = (data.T @ data) / (n_obs - 1)

        # PCA via eigendecomposition of the sample covariance
        eigenvalues, eigenvectors = np.linalg.eigh(sample_cov)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Clamp negative eigenvalues (numerical noise)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # Select number of factors
        total_var = eigenvalues.sum()
        if total_var < 1e-15:
            # Degenerate case — all returns are zero/constant
            n_factors = 1
        elif cfg.n_factors is not None:
            n_factors = min(cfg.n_factors, n_assets - 1, cfg.max_factors)
        else:
            cum_var = np.cumsum(eigenvalues) / total_var
            n_factors = int(np.searchsorted(cum_var, cfg.variance_threshold)) + 1
            n_factors = min(n_factors, n_assets - 1, cfg.max_factors)
        n_factors = max(1, n_factors)

        # Factor loadings: loadings = V_k · diag(√λ_k)
        # where V_k are the top-k eigenvectors, λ_k the top-k eigenvalues
        sqrt_eig = np.sqrt(eigenvalues[:n_factors])
        loadings_mat = eigenvectors[:, :n_factors] * sqrt_eig  # (N, K)

        # Factor covariance in PCA space is identity (orthogonal factors)
        fcov = np.eye(n_factors)

        # Systematic covariance: loadings · loadingsᵀ
        systematic_cov = loadings_mat @ loadings_mat.T

        # Specific (idiosyncratic) variance: diag(Σ_sample - Σ_systematic)
        residual_var = np.diag(sample_cov) - np.diag(systematic_cov)
        residual_var = np.maximum(residual_var, 0.0)
        diag_specific = np.diag(residual_var)

        # Reconstructed covariance: Σ = loadings·loadingsᵀ + D
        reconstructed_cov = systematic_cov + diag_specific

        # Ledoit-Wolf shrinkage toward the structured (factor) estimate
        shrinkage = self._compute_shrinkage(
            data, sample_cov, reconstructed_cov, cfg
        )
        final_cov = (1 - shrinkage) * sample_cov + shrinkage * reconstructed_cov

        # Annualise
        if cfg.annualise:
            final_cov *= TRADING_DAYS
            residual_var_out = residual_var * TRADING_DAYS
        else:
            residual_var_out = residual_var

        # Factor info
        variance_explained = eigenvalues[:n_factors].sum() / total_var if total_var > 1e-15 else 0.0
        cum_ratios = np.cumsum(eigenvalues[:n_factors]) / total_var if total_var > 1e-15 else np.zeros(n_factors)
        factor_infos = [
            FactorInfo(
                index=i + 1,
                eigenvalue=float(eigenvalues[i]),
                explained_variance_ratio=float(eigenvalues[i] / total_var) if total_var > 1e-15 else 0.0,
                cumulative_variance_ratio=float(cum_ratios[i]),
            )
            for i in range(n_factors)
        ]

        # Systematic risk fraction per asset
        total_asset_var = np.diag(sample_cov)
        systematic_asset_var = np.diag(systematic_cov)
        systematic_pct = np.where(
            total_asset_var > 1e-15,
            systematic_asset_var / total_asset_var,
            0.0,
        )

        cov_df = pd.DataFrame(final_cov, index=symbols, columns=symbols)
        loadings_df = pd.DataFrame(
            loadings_mat, index=symbols,
            columns=[f"F{i+1}" for i in range(n_factors)],
        )
        spec_var_series = pd.Series(residual_var_out, index=symbols, name="specific_variance")
        sys_pct_series = pd.Series(systematic_pct, index=symbols, name="systematic_pct")

        return FactorModelResult(
            covariance=cov_df,
            loadings=loadings_df,
            factor_covariance=fcov,
            specific_variance=spec_var_series,
            factors=factor_infos,
            n_assets=n_assets,
            n_factors=n_factors,
            n_observations=n_obs,
            total_variance_explained=float(variance_explained),
            systematic_risk_pct=sys_pct_series,
            shrinkage_intensity=shrinkage,
            symbols=symbols,
        )

    @staticmethod
    def _compute_shrinkage(
        data: np.ndarray,
        sample_cov: np.ndarray,
        target: np.ndarray,
        cfg: FactorModelConfig,
    ) -> float:
        """Compute Ledoit-Wolf shrinkage intensity.

        If ``cfg.shrinkage_intensity`` is set, returns that directly.
        Otherwise uses the Oracle Approximating Shrinkage (OAS) formula.
        """
        if cfg.shrinkage_intensity is not None:
            return np.clip(cfg.shrinkage_intensity, 0.0, 1.0)

        n, p = data.shape
        if n < 2 or p < 2:
            return 0.5

        # Simplified Ledoit-Wolf:
        # δ* = min(1, (sum of squared estimation errors) / (sum of squared deviations))
        delta = target - sample_cov
        delta_sq_sum = float((delta**2).sum())

        if delta_sq_sum < 1e-15:
            return 1.0  # sample == target, shrinkage is irrelevant

        # Estimate numerator: sum of Var(s_ij) ≈ (1/n²) Σ_t (x_t x_t' - S)²
        # Simplified: use the squared Frobenius norm heuristic
        centered = data - data.mean(axis=0)
        s2 = 0.0
        for t in range(n):
            x = centered[t : t + 1]  # (1, p)
            outer = x.T @ x
            diff = outer - sample_cov
            s2 += float((diff**2).sum())
        s2 /= n * n

        rho = s2 / delta_sq_sum
        return float(np.clip(rho, 0.0, 1.0))
