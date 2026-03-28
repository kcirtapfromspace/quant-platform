"""Forward-looking portfolio risk decomposition.

Decomposes portfolio risk into factor (systematic) and specific
(idiosyncratic) components at both position and factor level, using
Euler's homogeneous risk decomposition theorem.

Risk decomposition model::

    σ²_p = w' Σ w = w' (B F B' + D) w

    Factor risk²:   w' B F B' w
    Specific risk²: w' D w

    Marginal risk:  MRC_i = (Σ w)_i / σ_p
    Risk contrib:   RC_i  = w_i · MRC_i

Key property: ΣRC_i = σ_p  (Euler's theorem).

Key outputs:

  * **PositionRisk** — per-asset risk contribution, marginal risk, and
    factor/specific split.
  * **FactorRiskContrib** — per-factor risk contribution from the PCA
    factor model.
  * **RiskDecompositionResult** — complete portfolio risk breakdown with
    concentration metrics.

Usage::

    from quant.risk.risk_decomposition import (
        RiskDecomposer,
        DecompositionConfig,
    )

    decomposer = RiskDecomposer()
    result = decomposer.decompose(
        weights=portfolio_weights,
        covariance=cov_matrix,
        factor_loadings=loadings_df,
        factor_covariance=factor_cov,
        specific_variance=specific_var,
    )
    print(result.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DecompositionConfig:
    """Configuration for risk decomposition.

    Attributes:
        top_n:  Number of top risk contributors to highlight.
    """

    top_n: int = 10


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionRisk:
    """Risk attribution for a single position.

    Attributes:
        symbol:                     Asset ticker.
        weight:                     Portfolio weight.
        risk_contribution:          Euler risk contribution (sums to σ_p).
        risk_share:                 RC_i / σ_p  (sums to 1.0).
        marginal_risk:              ∂σ_p / ∂w_i.
        factor_risk_contribution:   Risk contribution from systematic factors.
        specific_risk_contribution: Risk contribution from idiosyncratic variance.
    """

    symbol: str
    weight: float
    risk_contribution: float
    risk_share: float
    marginal_risk: float
    factor_risk_contribution: float
    specific_risk_contribution: float


@dataclass(frozen=True, slots=True)
class FactorRiskContrib:
    """Risk contribution from one latent factor.

    Attributes:
        factor:             Factor identifier (e.g. ``"F0"``).
        risk_contribution:  Factor's contribution to portfolio volatility.
        risk_share:         Fraction of total factor risk from this factor.
        portfolio_exposure: Portfolio's net exposure to this factor (θ_k = B_k'w).
    """

    factor: str
    risk_contribution: float
    risk_share: float
    portfolio_exposure: float


@dataclass
class RiskDecompositionResult:
    """Complete portfolio risk decomposition.

    Attributes:
        total_risk:             Portfolio volatility (σ_p).
        factor_risk:            Systematic risk component.
        specific_risk:          Idiosyncratic risk component.
        factor_risk_pct:        Factor variance / total variance.
        positions:              Per-position risk attribution.
        factors:                Per-factor risk attribution (if factor model provided).
        top_risk_contributors:  Positions with largest absolute risk contribution.
        hhi:                    HHI of risk shares (concentration).
        n_positions:            Number of positions with non-zero weight.
    """

    total_risk: float = 0.0
    factor_risk: float = 0.0
    specific_risk: float = 0.0
    factor_risk_pct: float = 0.0
    positions: list[PositionRisk] = field(default_factory=list, repr=False)
    factors: list[FactorRiskContrib] = field(default_factory=list, repr=False)
    top_risk_contributors: list[PositionRisk] = field(default_factory=list)
    hhi: float = 0.0
    n_positions: int = 0

    def summary(self) -> str:
        """Return a human-readable risk decomposition summary."""
        lines = [
            f"Risk Decomposition ({self.n_positions} positions)",
            "=" * 60,
            "",
            f"Total portfolio risk  : {self.total_risk:.4f}",
            f"  Factor (systematic) : {self.factor_risk:.4f}"
            f" ({self.factor_risk_pct:.1%} of variance)",
            f"  Specific (idio)     : {self.specific_risk:.4f}"
            f" ({1.0 - self.factor_risk_pct:.1%} of variance)",
            "",
            f"Risk concentration (HHI): {self.hhi:.4f}",
            "",
            "Top risk contributors (by |RC|):",
        ]
        for p in self.top_risk_contributors:
            lines.append(
                f"  {p.symbol:<10s}: RC={p.risk_contribution:+.4f} "
                f"(share={p.risk_share:+.1%}, wt={p.weight:+.2%})"
            )

        if self.factors:
            lines.extend(["", "Factor risk decomposition:"])
            for f in sorted(self.factors, key=lambda x: abs(x.risk_contribution), reverse=True):
                lines.append(
                    f"  {f.factor:<8s}: RC={f.risk_contribution:+.4f} "
                    f"(share={f.risk_share:+.1%}, exposure={f.portfolio_exposure:+.4f})"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------


class RiskDecomposer:
    """Forward-looking portfolio risk decomposer.

    Uses Euler's theorem to attribute portfolio volatility to individual
    positions and latent risk factors.

    Args:
        config: Decomposition configuration.
    """

    def __init__(self, config: DecompositionConfig | None = None) -> None:
        self._config = config or DecompositionConfig()

    @property
    def config(self) -> DecompositionConfig:
        return self._config

    def decompose(
        self,
        weights: dict[str, float],
        covariance: pd.DataFrame,
        factor_loadings: pd.DataFrame | None = None,
        factor_covariance: np.ndarray | None = None,
        specific_variance: pd.Series | None = None,
    ) -> RiskDecompositionResult:
        """Decompose portfolio risk into position and factor contributions.

        Args:
            weights:            Portfolio weights ``{symbol: weight}``.
            covariance:         N×N covariance matrix (pd.DataFrame).
            factor_loadings:    N×K factor loading matrix from
                                :class:`~quant.risk.factor_model.FactorModelResult`.
                                If provided with ``factor_covariance`` and
                                ``specific_variance``, enables factor-level
                                decomposition.
            factor_covariance:  K×K factor covariance matrix.
            specific_variance:  Per-asset idiosyncratic variance (pd.Series).

        Returns:
            :class:`RiskDecompositionResult` with full risk attribution.

        Raises:
            ValueError: If weights contain symbols not in the covariance matrix
                        or fewer than 1 position.
        """
        cfg = self._config

        # Filter to non-zero weights
        active = {s: w for s, w in weights.items() if abs(w) > 1e-12}
        if not active:
            return RiskDecompositionResult()

        symbols = sorted(active.keys())

        # Validate symbols exist in covariance
        missing = set(symbols) - set(covariance.columns)
        if missing:
            raise ValueError(
                f"Symbols not in covariance matrix: {sorted(missing)}"
            )

        n = len(symbols)
        w = np.array([active[s] for s in symbols])
        cov = covariance.loc[symbols, symbols].values

        # Total portfolio variance and risk
        port_var = float(w @ cov @ w)
        port_risk = math.sqrt(max(port_var, 0.0))

        if port_risk < 1e-15:
            # Degenerate case: zero risk
            positions = [
                PositionRisk(
                    symbol=s, weight=active[s],
                    risk_contribution=0.0, risk_share=0.0,
                    marginal_risk=0.0,
                    factor_risk_contribution=0.0,
                    specific_risk_contribution=0.0,
                )
                for s in symbols
            ]
            return RiskDecompositionResult(
                positions=positions, n_positions=n,
            )

        # Marginal risk contribution: MRC = Σw / σ_p
        sigma_w = cov @ w
        mrc = sigma_w / port_risk

        # Risk contribution: RC_i = w_i × MRC_i
        rc = w * mrc

        # Factor / specific decomposition
        has_factor_model = (
            factor_loadings is not None
            and factor_covariance is not None
            and specific_variance is not None
        )

        factor_risk_var = 0.0
        specific_risk_var = 0.0
        factor_rc = np.zeros(n)
        specific_rc = np.zeros(n)
        factor_contribs: list[FactorRiskContrib] = []

        if has_factor_model:
            # Align factor loadings to our symbol order
            fl = factor_loadings.loc[symbols].values  # N × K
            fc = factor_covariance  # K × K
            sv = np.array([specific_variance[s] for s in symbols])  # N

            # Specific risk from annualised specific variance: w'Dw
            specific_risk_var = float((w ** 2) @ sv)

            # Factor risk by difference — correctly handles annualisation
            # and shrinkage since Σ may differ from BFB' + D
            factor_risk_var = max(port_var - specific_risk_var, 0.0)

            # Per-position specific RC: w_i × (Dw)_i / σ_p = w_i² × d_i / σ_p
            specific_sigma_w = sv * w  # N
            specific_rc = w * specific_sigma_w / port_risk

            # Per-position factor RC by difference: total RC - specific RC
            factor_rc = rc - specific_rc

            # Per-factor risk contribution using relative loadings shares
            # θ = B'w (portfolio factor exposures, non-annualised space)
            theta = fl.T @ w  # K
            fc_theta = fc @ theta  # K
            raw_factor_var = float(theta @ fc_theta)
            factor_risk_total = math.sqrt(max(factor_risk_var, 0.0))
            n_factors = len(theta)

            for k in range(n_factors):
                frc = theta[k] * fc_theta[k]
                # Share of systematic variance from this factor
                frc_share = frc / raw_factor_var if raw_factor_var > 1e-15 else 0.0
                # Absolute risk contribution scaled to annualised factor risk
                frc_abs = frc_share * factor_risk_total
                factor_contribs.append(FactorRiskContrib(
                    factor=f"F{k}",
                    risk_contribution=frc_abs,
                    risk_share=frc_share,
                    portfolio_exposure=float(theta[k]),
                ))

        factor_risk = math.sqrt(max(factor_risk_var, 0.0))
        specific_risk = math.sqrt(max(specific_risk_var, 0.0))
        factor_pct = factor_risk_var / port_var if port_var > 1e-15 else 0.0

        # Build position-level results
        positions: list[PositionRisk] = []
        for i, sym in enumerate(symbols):
            risk_share = rc[i] / port_risk if port_risk > 1e-15 else 0.0
            positions.append(PositionRisk(
                symbol=sym,
                weight=float(w[i]),
                risk_contribution=float(rc[i]),
                risk_share=float(risk_share),
                marginal_risk=float(mrc[i]),
                factor_risk_contribution=float(factor_rc[i]),
                specific_risk_contribution=float(specific_rc[i]),
            ))

        # Sort by absolute risk contribution
        positions.sort(key=lambda p: abs(p.risk_contribution), reverse=True)

        # Top contributors
        top = positions[:cfg.top_n]

        # Concentration: HHI of risk shares
        shares = [abs(p.risk_share) for p in positions]
        total_abs_share = sum(shares)
        hhi = 0.0
        if total_abs_share > 1e-15:
            norm_shares = [s / total_abs_share for s in shares]
            hhi = sum(s * s for s in norm_shares)

        return RiskDecompositionResult(
            total_risk=port_risk,
            factor_risk=factor_risk,
            specific_risk=specific_risk,
            factor_risk_pct=factor_pct,
            positions=positions,
            factors=factor_contribs,
            top_risk_contributors=top,
            hhi=hhi,
            n_positions=n,
        )
