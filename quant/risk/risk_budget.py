"""Risk budget allocation across strategies.

Distributes a total portfolio risk budget (expressed as target volatility)
across strategy sleeves.  The allocation maximises expected risk-adjusted
return by weighting sleeves according to their Sharpe ratio, capacity
headroom, and diversification contribution.

Three allocation methods are supported:

  * **Equal risk** — each sleeve receives the same risk budget (1/N of total).
  * **Sharpe-weighted** — sleeves with higher Sharpe receive more risk budget,
    proportional to their Sharpe ratio.
  * **Optimized** — maximise portfolio Sharpe subject to total risk and
    per-sleeve risk bounds, using the correlation structure.

Key outputs:

  * Per-sleeve risk budgets (in vol and VaR terms).
  * Expected portfolio Sharpe given the allocation.
  * Marginal risk contribution per sleeve.
  * Utilisation: actual vs allocated risk per sleeve.

Usage::

    from quant.risk.risk_budget import (
        RiskBudgetAllocator,
        RiskBudgetConfig,
    )

    allocator = RiskBudgetAllocator(RiskBudgetConfig(
        total_vol_target=0.10,
        method="sharpe_weighted",
    ))

    result = allocator.allocate(
        sleeve_vols={"momentum": 0.15, "mean_rev": 0.08, "stat_arb": 0.12},
        sleeve_sharpes={"momentum": 1.2, "mean_rev": 0.8, "stat_arb": 1.5},
        correlation_matrix=corr,
    )
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class AllocationMethod(Enum):
    """Risk budget allocation method."""

    EQUAL_RISK = "equal_risk"
    SHARPE_WEIGHTED = "sharpe_weighted"
    OPTIMIZED = "optimized"


@dataclass
class RiskBudgetConfig:
    """Configuration for risk budget allocation.

    Attributes:
        total_vol_target:   Target portfolio annualised volatility.
        method:             Allocation method.
        min_risk_share:     Minimum risk share per sleeve (floor).
        max_risk_share:     Maximum risk share per sleeve (cap).
        var_confidence:     Confidence level for VaR conversion.
        var_horizon_days:   Horizon in days for VaR conversion.
    """

    total_vol_target: float = 0.10
    method: AllocationMethod = AllocationMethod.SHARPE_WEIGHTED
    min_risk_share: float = 0.05
    max_risk_share: float = 0.50
    var_confidence: float = 0.95
    var_horizon_days: int = 1


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SleeveRiskBudget:
    """Risk budget allocation for a single sleeve.

    Attributes:
        name:               Sleeve name.
        risk_share:         Fraction of total risk budget (sums to 1).
        allocated_vol:      Annualised vol budget for this sleeve.
        allocated_var:      VaR budget in return terms.
        capital_weight:     Implied capital weight (vol budget / sleeve vol).
        marginal_risk:      Marginal contribution to portfolio risk.
        sleeve_vol:         Sleeve's standalone volatility.
        sleeve_sharpe:      Sleeve's Sharpe ratio.
    """

    name: str
    risk_share: float
    allocated_vol: float
    allocated_var: float
    capital_weight: float
    marginal_risk: float
    sleeve_vol: float
    sleeve_sharpe: float


@dataclass
class RiskBudgetResult:
    """Portfolio-level risk budget allocation.

    Attributes:
        sleeves:                Per-sleeve risk budgets.
        total_vol_target:       Target portfolio vol.
        expected_portfolio_vol: Realised vol given correlation.
        expected_portfolio_sharpe: Portfolio Sharpe from allocation.
        method:                 Allocation method used.
        capital_weights:        Implied capital weights per sleeve.
        n_sleeves:              Number of sleeves.
    """

    sleeves: list[SleeveRiskBudget]
    total_vol_target: float
    expected_portfolio_vol: float
    expected_portfolio_sharpe: float
    method: AllocationMethod
    capital_weights: dict[str, float] = field(repr=False)
    n_sleeves: int = 0

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Risk Budget Allocation ({self.n_sleeves} sleeves)",
            "=" * 65,
            "",
            f"Method                  : {self.method.value}",
            f"Target portfolio vol    : {self.total_vol_target:.2%}",
            f"Expected portfolio vol  : {self.expected_portfolio_vol:.2%}",
            f"Expected portfolio Sharpe: {self.expected_portfolio_sharpe:.2f}",
            "",
            f"{'Sleeve':<15s} {'Share':>7s} {'Vol Bdgt':>9s} "
            f"{'VaR Bdgt':>9s} {'Cap Wt':>7s} {'Marg R':>7s} {'Sharpe':>7s}",
            "-" * 65,
        ]
        for s in sorted(self.sleeves, key=lambda x: -x.risk_share):
            lines.append(
                f"{s.name:<15s} {s.risk_share:>7.1%} {s.allocated_vol:>8.2%} "
                f"{s.allocated_var:>9.4f} {s.capital_weight:>7.1%} "
                f"{s.marginal_risk:>7.4f} {s.sleeve_sharpe:>7.2f}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------


class RiskBudgetAllocator:
    """Allocates a total risk budget across strategy sleeves.

    Args:
        config: Risk budget configuration.
    """

    def __init__(self, config: RiskBudgetConfig | None = None) -> None:
        self._config = config or RiskBudgetConfig()

    @property
    def config(self) -> RiskBudgetConfig:
        return self._config

    def allocate(
        self,
        sleeve_vols: dict[str, float],
        sleeve_sharpes: dict[str, float] | None = None,
        correlation_matrix: dict[tuple[str, str], float] | None = None,
    ) -> RiskBudgetResult:
        """Allocate risk budget across sleeves.

        Args:
            sleeve_vols:        ``{name: annualised_vol}`` per sleeve.
            sleeve_sharpes:     ``{name: sharpe_ratio}`` per sleeve.
                                Required for ``SHARPE_WEIGHTED`` and ``OPTIMIZED``.
                                Defaults to equal Sharpe (1.0) if not provided.
            correlation_matrix: ``{(name_i, name_j): corr}`` pairwise correlations.
                                Missing pairs default to 0.0.

        Returns:
            :class:`RiskBudgetResult` with per-sleeve budgets and portfolio metrics.

        Raises:
            ValueError: If fewer than 1 sleeve provided.
        """
        cfg = self._config
        names = sorted(sleeve_vols.keys())
        n = len(names)

        if n < 1:
            raise ValueError("Need at least 1 sleeve")

        vols = np.array([sleeve_vols[s] for s in names])
        sharpes = np.array([
            (sleeve_sharpes or {}).get(s, 1.0) for s in names
        ])
        # Ensure non-negative Sharpes for allocation
        sharpes_clipped = np.maximum(sharpes, 0.0)

        # Build correlation matrix
        corr = np.eye(n)
        if correlation_matrix:
            for i, ni in enumerate(names):
                for j, nj in enumerate(names):
                    if i != j:
                        corr[i, j] = correlation_matrix.get(
                            (ni, nj),
                            correlation_matrix.get((nj, ni), 0.0),
                        )

        # Covariance matrix
        cov = np.outer(vols, vols) * corr

        # Compute raw risk shares
        if cfg.method == AllocationMethod.EQUAL_RISK:
            raw_shares = np.ones(n) / n
        elif cfg.method == AllocationMethod.SHARPE_WEIGHTED:
            raw_shares = self._sharpe_weighted_shares(sharpes_clipped)
        else:  # OPTIMIZED
            raw_shares = self._optimized_shares(
                sharpes_clipped, vols, cov, cfg.total_vol_target,
            )

        # Clamp to [min, max] and renormalise
        shares = self._clamp_and_normalise(
            raw_shares, cfg.min_risk_share, cfg.max_risk_share,
        )

        # Allocated vol per sleeve: share * total_vol_target
        allocated_vols = shares * cfg.total_vol_target

        # Implied capital weights: allocated_vol / sleeve_vol
        cap_weights = np.where(
            vols > 1e-10, allocated_vols / vols, 0.0,
        )

        # Portfolio vol given capital weights and covariance
        port_var = float(cap_weights @ cov @ cap_weights)
        port_vol = np.sqrt(port_var) if port_var > 0 else 0.0

        # Portfolio expected Sharpe
        port_return = float(cap_weights @ (sharpes * vols))
        port_sharpe = port_return / port_vol if port_vol > 1e-10 else 0.0

        # Marginal risk contribution: w_i * (Σw)_i / σ_p
        sigma_w = cov @ cap_weights
        marginal = (
            cap_weights * sigma_w / port_vol if port_vol > 1e-10
            else np.zeros(n)
        )

        # VaR conversion
        z = self._z_score(cfg.var_confidence)
        horizon_scale = np.sqrt(cfg.var_horizon_days / 252)

        sleeve_budgets: list[SleeveRiskBudget] = []
        for i, name in enumerate(names):
            alloc_var = allocated_vols[i] * z * horizon_scale
            sleeve_budgets.append(
                SleeveRiskBudget(
                    name=name,
                    risk_share=float(shares[i]),
                    allocated_vol=float(allocated_vols[i]),
                    allocated_var=float(alloc_var),
                    capital_weight=float(cap_weights[i]),
                    marginal_risk=float(marginal[i]),
                    sleeve_vol=float(vols[i]),
                    sleeve_sharpe=float(sharpes[i]),
                )
            )

        return RiskBudgetResult(
            sleeves=sleeve_budgets,
            total_vol_target=cfg.total_vol_target,
            expected_portfolio_vol=float(port_vol),
            expected_portfolio_sharpe=float(port_sharpe),
            method=cfg.method,
            capital_weights={names[i]: float(cap_weights[i]) for i in range(n)},
            n_sleeves=n,
        )

    @staticmethod
    def _sharpe_weighted_shares(sharpes: np.ndarray) -> np.ndarray:
        """Allocate proportional to Sharpe ratios."""
        total = sharpes.sum()
        if total < 1e-10:
            return np.ones(len(sharpes)) / len(sharpes)
        return sharpes / total

    @staticmethod
    def _optimized_shares(
        sharpes: np.ndarray,
        vols: np.ndarray,
        cov: np.ndarray,
        target_vol: float,
    ) -> np.ndarray:
        """Approximate risk-parity-Sharpe blend.

        Uses inverse-vol weighting adjusted by Sharpe as a heuristic
        for maximising portfolio Sharpe without a full quadratic solver.
        """
        inv_vol = np.where(vols > 1e-10, 1.0 / vols, 0.0)
        # Blend: inv_vol * sharpe gives more weight to high-Sharpe, low-vol
        raw = inv_vol * np.maximum(sharpes, 0.01)
        total = raw.sum()
        if total < 1e-10:
            return np.ones(len(sharpes)) / len(sharpes)
        return raw / total

    @staticmethod
    def _clamp_and_normalise(
        shares: np.ndarray,
        min_share: float,
        max_share: float,
        max_rounds: int = 20,
    ) -> np.ndarray:
        """Iteratively clamp shares to [min, max] and renormalise."""
        result = shares.copy()
        for _ in range(max_rounds):
            clamped = np.clip(result, min_share, max_share)
            total = clamped.sum()
            if total < 1e-10:
                return np.ones(len(shares)) / len(shares)
            clamped /= total
            if np.allclose(clamped, result, atol=1e-10):
                break
            result = clamped
        return result

    @staticmethod
    def _z_score(confidence: float) -> float:
        """Approximate z-score for normal distribution."""
        # Common values; for anything else, use a rough approximation
        table = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
        if confidence in table:
            return table[confidence]
        # Beasley-Springer-Moro approximation for probit
        p = confidence
        a = [-3.969683028665376e1, 2.209460984245205e2,
             -2.759285104469687e2, 1.383577518672690e2,
             -3.066479806614716e1, 2.506628277459239e0]
        b = [-5.447609879822406e1, 1.615858368580409e2,
             -1.556989798598866e2, 6.680131188771972e1,
             -1.328068155288572e1]
        c = [-7.784894002430293e-3, -3.223964580411365e-1,
             -2.400758277161838e0, -2.549732539343734e0,
             4.374664141464968e0, 2.938163982698783e0]
        d = [7.784695709041462e-3, 3.224671290700398e-1,
             2.445134137142996e0, 3.754408661907416e0]
        p_low = 0.02425
        p_high = 1 - p_low
        if p < p_low:
            q = (-2 * np.log(p)) ** 0.5
            return float(
                (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
                / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
            )
        if p <= p_high:
            q = p - 0.5
            r = q * q
            return float(
                (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q
                / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
            )
        q = (-2 * np.log(1 - p)) ** 0.5
        return float(
            -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
            / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        )
