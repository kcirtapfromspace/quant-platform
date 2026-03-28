"""Transaction-cost-aware portfolio optimizer.

Extends mean-variance optimisation to include a quadratic transaction cost
penalty directly in the objective function, producing target weights that
already account for the cost of rebalancing from the current portfolio.

Objective::

    maximise  μᵀw − (λ/2) wᵀΣw − (κ/2)(w − w₀)ᵀ Φ (w − w₀)

Where:
  * **μ** — expected return vector (alpha scores).
  * **Σ** — covariance matrix.
  * **λ** — risk aversion parameter.
  * **κ** — cost penalty multiplier.
  * **Φ** — diagonal trading cost matrix (per-asset cost coefficients).
  * **w₀** — current portfolio holdings.

The trading cost matrix Φ encodes per-asset costs:
  * **Linear component**: ``2 · c_linear`` (quadratic approximation of
    proportional costs).
  * **Impact component**: ``η · σ_i · √(AUM / ADV_i)`` (square-root
    market impact scaling).

Closed-form solution::

    w* = (λΣ + κΦ)⁻¹ (μ + κΦ w₀)

This naturally shrinks the optimal portfolio toward current holdings with
strength proportional to κ — higher cost penalty means less trading.

Usage::

    from quant.portfolio.cost_aware_optimizer import (
        CostAwareOptimizer,
        CostAwareConfig,
    )

    optimizer = CostAwareOptimizer(CostAwareConfig(
        risk_aversion=1.0,
        cost_penalty=5.0,
    ))
    result = optimizer.optimize(
        symbols=symbols,
        cov_matrix=cov_df,
        expected_returns=alpha_scores,
        constraints=constraints,
        current_weights=current_holdings,
    )
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant.portfolio.constraints import PortfolioConstraints

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CostAwareConfig:
    """Configuration for cost-aware portfolio optimisation.

    Attributes:
        risk_aversion:      Risk aversion parameter λ (higher = more conservative).
        cost_penalty:       Multiplier κ on the quadratic trading cost penalty.
        linear_cost_bps:    Linear transaction cost in basis points (per unit traded).
        impact_coefficient: Market impact coefficient η (dimensionless).
        aum:                Assets under management (USD), for impact sizing.
        shrinkage:          Ridge regularisation on the covariance diagonal.
    """

    risk_aversion: float = 1.0
    cost_penalty: float = 5.0
    linear_cost_bps: float = 5.0
    impact_coefficient: float = 0.1
    aum: float = 100_000_000.0
    shrinkage: float = 1e-6


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeCost:
    """Estimated transaction cost for one asset."""

    symbol: str
    trade_weight: float
    linear_cost: float
    impact_cost: float
    total_cost: float


@dataclass
class CostAwareResult:
    """Result of cost-aware portfolio optimisation.

    Attributes:
        weights:            Optimal target weights {symbol: weight}.
        current_weights:    Starting weights {symbol: weight}.
        expected_return:    Portfolio expected return (pre-cost).
        risk:               Portfolio volatility (annualised).
        total_cost:         Estimated total transaction cost (as fraction of AUM).
        net_expected_return: Expected return minus cost penalty.
        trade_costs:        Per-asset cost breakdown.
        turnover:           One-way turnover (sum of |Δw|).
        n_trades:           Number of assets with non-zero trades.
    """

    weights: dict[str, float]
    current_weights: dict[str, float]
    expected_return: float = 0.0
    risk: float = 0.0
    total_cost: float = 0.0
    net_expected_return: float = 0.0
    trade_costs: list[TradeCost] = field(repr=False, default_factory=list)
    turnover: float = 0.0
    n_trades: int = 0

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Cost-Aware Portfolio Optimization ({len(self.weights)} assets)",
            "=" * 65,
            "",
            f"Expected return (pre-cost) : {self.expected_return:+.4f}",
            f"Portfolio risk (ann.)      : {self.risk:.4f}",
            f"Total transaction cost     : {self.total_cost:.6f}",
            f"Net expected return        : {self.net_expected_return:+.4f}",
            "",
            f"One-way turnover           : {self.turnover:.4f}",
            f"Number of trades           : {self.n_trades}",
            "",
            f"{'Symbol':<10s} {'Target':>8s} {'Current':>8s} {'Trade':>8s} "
            f"{'Lin$':>8s} {'Imp$':>8s} {'Tot$':>8s}",
            "-" * 65,
        ]
        for tc in sorted(self.trade_costs, key=lambda t: abs(t.trade_weight), reverse=True)[:10]:
            lines.append(
                f"{tc.symbol:<10s} "
                f"{self.weights.get(tc.symbol, 0):>+8.4f} "
                f"{self.current_weights.get(tc.symbol, 0):>+8.4f} "
                f"{tc.trade_weight:>+8.4f} "
                f"{tc.linear_cost:>8.6f} "
                f"{tc.impact_cost:>8.6f} "
                f"{tc.total_cost:>8.6f}"
            )
        if len(self.trade_costs) > 10:
            lines.append(f"  ... and {len(self.trade_costs) - 10} more assets")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class CostAwareOptimizer:
    """Transaction-cost-aware portfolio optimizer.

    Uses a quadratic trading cost penalty to shrink the optimal portfolio
    toward current holdings, balancing alpha capture against trading costs.

    Args:
        config: Optimisation configuration.
    """

    def __init__(self, config: CostAwareConfig | None = None) -> None:
        self._config = config or CostAwareConfig()

    @property
    def config(self) -> CostAwareConfig:
        return self._config

    def optimize(
        self,
        symbols: list[str],
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None,
        constraints: PortfolioConstraints,
        current_weights: dict[str, float] | None = None,
        adv: dict[str, float] | None = None,
    ) -> CostAwareResult:
        """Optimise portfolio weights accounting for transaction costs.

        Args:
            symbols:          Ordered list of asset symbols.
            cov_matrix:       N×N covariance matrix (annualised).
            expected_returns: Expected return per asset (annualised).
            constraints:      Portfolio constraints to enforce.
            current_weights:  Current portfolio holdings {symbol: weight}.
                              Defaults to zero (cash start).
            adv:              Average daily volume per asset in USD.
                              Used for impact cost.  If ``None``, impact is
                              computed without ADV scaling.

        Returns:
            :class:`CostAwareResult` with optimal weights and cost analysis.

        Raises:
            ValueError: If no symbols provided.
        """
        cfg = self._config

        if not symbols:
            raise ValueError("Need at least 1 symbol")

        n = len(symbols)
        cov = cov_matrix.loc[symbols, symbols].values.astype(float)
        cov += np.eye(n) * cfg.shrinkage

        if expected_returns is not None:
            mu = expected_returns.reindex(symbols).fillna(0.0).values.astype(float)
        else:
            mu = np.zeros(n)

        w0 = np.array([
            (current_weights or {}).get(s, 0.0) for s in symbols
        ], dtype=float)

        adv_arr = np.array([
            (adv or {}).get(s, 0.0) for s in symbols
        ], dtype=float)

        asset_vols = np.sqrt(np.diag(cov))

        # Build diagonal trading cost matrix Φ
        phi = self._build_cost_matrix(asset_vols, adv_arr, cfg)

        # Closed-form: w* = (λΣ + κΦ)⁻¹ (μ + κΦw₀)
        lhs = cfg.risk_aversion * cov + cfg.cost_penalty * np.diag(phi)
        rhs = mu + cfg.cost_penalty * phi * w0
        w = np.linalg.solve(lhs, rhs)

        # Apply constraints
        w = self._apply_constraints(w, constraints)

        # Compute final metrics
        port_var = float(w @ cov @ w)
        port_risk = np.sqrt(max(port_var, 0.0))
        port_ret = float(w @ mu)

        # Per-asset trade costs (actual linear + impact, not the quadratic proxy)
        trade_costs = self._compute_trade_costs(
            symbols, w, w0, asset_vols, adv_arr, cfg,
        )
        total_cost = sum(tc.total_cost for tc in trade_costs)

        trades = w - w0
        turnover = float(np.sum(np.abs(trades)))
        n_trades_count = int(np.sum(np.abs(trades) > 1e-8))

        weights_dict = dict(zip(symbols, w.tolist(), strict=True))
        current_dict = dict(zip(symbols, w0.tolist(), strict=True))

        return CostAwareResult(
            weights=weights_dict,
            current_weights=current_dict,
            expected_return=port_ret,
            risk=port_risk,
            total_cost=total_cost,
            net_expected_return=port_ret - cfg.cost_penalty * total_cost,
            trade_costs=trade_costs,
            turnover=turnover,
            n_trades=n_trades_count,
        )

    @staticmethod
    def _build_cost_matrix(
        asset_vols: np.ndarray,
        adv: np.ndarray,
        cfg: CostAwareConfig,
    ) -> np.ndarray:
        """Build the diagonal trading cost vector Φ.

        Each element φ_i captures the per-unit-weight cost of trading asset i:
          φ_i = 2·c_linear + η·σ_i·√(AUM/ADV_i)
        """
        n = len(asset_vols)
        phi = np.full(n, 2.0 * cfg.linear_cost_bps / 10_000)

        for i in range(n):
            if adv[i] > 0:
                phi[i] += (
                    cfg.impact_coefficient
                    * asset_vols[i]
                    * np.sqrt(cfg.aum / adv[i])
                )

        return phi

    @staticmethod
    def _apply_constraints(
        w: np.ndarray,
        constraints: PortfolioConstraints,
    ) -> np.ndarray:
        """Apply portfolio constraints via clipping and rescaling."""
        eff_min = constraints.effective_min()
        w = np.clip(w, eff_min, constraints.max_weight)

        gross = np.sum(np.abs(w))
        if gross > constraints.max_gross_exposure and gross > 1e-12:
            w *= constraints.max_gross_exposure / gross
            w = np.clip(w, eff_min, constraints.max_weight)

        return w

    @staticmethod
    def _compute_trade_costs(
        symbols: list[str],
        w: np.ndarray,
        w0: np.ndarray,
        asset_vols: np.ndarray,
        adv: np.ndarray,
        cfg: CostAwareConfig,
    ) -> list[TradeCost]:
        """Compute per-asset transaction cost breakdown."""
        costs: list[TradeCost] = []
        linear_frac = cfg.linear_cost_bps / 10_000

        for i, sym in enumerate(symbols):
            delta = w[i] - w0[i]
            abs_delta = abs(delta)

            lin_cost = linear_frac * abs_delta

            imp_cost = 0.0
            if abs_delta > 1e-12 and adv[i] > 0:
                participation = abs_delta * cfg.aum / adv[i]
                imp_cost = cfg.impact_coefficient * asset_vols[i] * np.sqrt(participation)

            costs.append(TradeCost(
                symbol=sym,
                trade_weight=float(delta),
                linear_cost=float(lin_cost),
                impact_cost=float(imp_cost),
                total_cost=float(lin_cost + imp_cost),
            ))

        return costs
