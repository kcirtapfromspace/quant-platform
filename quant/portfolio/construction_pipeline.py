"""End-to-end portfolio construction pipeline.

Chains the quant research and portfolio optimisation components into a
single callable that takes raw signal scores and produces target weights:

    signal scores → alpha model → risk model → cost-aware optimizer → weights

Components:

  1. **Alpha model** — converts signals to expected returns
     (:class:`~quant.research.alpha_model.AlphaModel`).
  2. **Risk model** — estimates the covariance matrix
     (:class:`~quant.risk.factor_model.FactorRiskModel`).
  3. **Optimizer** — solves for optimal weights with cost awareness
     (:class:`~quant.portfolio.cost_aware_optimizer.CostAwareOptimizer`).

The pipeline accepts pre-fitted components so it can be reused across
rebalance cycles with different signal snapshots.

Usage::

    from quant.portfolio.construction_pipeline import (
        ConstructionPipeline,
        PipelineConfig,
    )

    pipeline = ConstructionPipeline(PipelineConfig())
    result = pipeline.construct(
        signal_scores=signal_series,
        returns_history=returns_df,
        current_weights=current_holdings,
        adv=adv_dict,
    )
    target_weights = result.target_weights
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.cost_aware_optimizer import (
    CostAwareConfig,
    CostAwareOptimizer,
    CostAwareResult,
)
from quant.research.alpha_model import (
    AlphaModel,
    AlphaModelConfig,
    AlphaModelResult,
)
from quant.risk.factor_model import (
    FactorModelConfig,
    FactorModelResult,
    FactorRiskModel,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for the portfolio construction pipeline.

    Attributes:
        alpha_config:       Alpha model configuration.
        risk_config:        Factor risk model configuration.
        optimizer_config:   Cost-aware optimizer configuration.
        constraints:        Portfolio constraints.
    """

    alpha_config: AlphaModelConfig = field(default_factory=AlphaModelConfig)
    risk_config: FactorModelConfig = field(default_factory=FactorModelConfig)
    optimizer_config: CostAwareConfig = field(default_factory=CostAwareConfig)
    constraints: PortfolioConstraints = field(
        default_factory=lambda: PortfolioConstraints(long_only=True),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Result of the portfolio construction pipeline.

    Attributes:
        target_weights:     Optimal target weights {symbol: weight}.
        alpha_result:       Alpha model output.
        risk_result:        Factor risk model output.
        optimizer_result:   Cost-aware optimizer output.
        n_assets:           Number of assets in the universe.
        expected_return:    Portfolio expected return.
        risk:               Portfolio risk (annualised vol).
        total_cost:         Estimated transaction cost.
        turnover:           One-way turnover from current to target.
    """

    target_weights: dict[str, float]
    alpha_result: AlphaModelResult = field(repr=False)
    risk_result: FactorModelResult = field(repr=False)
    optimizer_result: CostAwareResult = field(repr=False)
    n_assets: int = 0
    expected_return: float = 0.0
    risk: float = 0.0
    total_cost: float = 0.0
    turnover: float = 0.0

    def summary(self) -> str:
        """Return a human-readable pipeline summary."""
        lines = [
            f"Portfolio Construction Pipeline ({self.n_assets} assets)",
            "=" * 60,
            "",
            "Alpha Model:",
            f"  Method         : {self.alpha_result.method.value}",
            f"  Forecast spread: {self.alpha_result.forecast_spread:+.4f}",
            "",
            "Risk Model:",
            f"  Factors        : {self.risk_result.n_factors}",
            f"  Var explained  : {self.risk_result.total_variance_explained:.1%}",
            "",
            "Optimizer:",
            f"  Expected return: {self.expected_return:+.4f}",
            f"  Portfolio risk : {self.risk:.4f}",
            f"  Total cost     : {self.total_cost:.6f}",
            f"  Turnover       : {self.turnover:.4f}",
            "",
            "Target Weights (top 10):",
        ]
        sorted_w = sorted(
            self.target_weights.items(), key=lambda x: abs(x[1]), reverse=True,
        )
        for sym, w in sorted_w[:10]:
            lines.append(f"  {sym:<10s}: {w:+.4f}")
        if len(sorted_w) > 10:
            lines.append(f"  ... and {len(sorted_w) - 10} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ConstructionPipeline:
    """End-to-end portfolio construction pipeline.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._alpha_model = AlphaModel(self._config.alpha_config)
        self._risk_model = FactorRiskModel(self._config.risk_config)
        self._optimizer = CostAwareOptimizer(self._config.optimizer_config)

    @property
    def config(self) -> PipelineConfig:
        return self._config

    def construct(
        self,
        signal_scores: pd.Series,
        returns_history: pd.DataFrame,
        current_weights: dict[str, float] | None = None,
        adv: dict[str, float] | None = None,
        asset_volatilities: pd.Series | None = None,
    ) -> PipelineResult:
        """Run the full construction pipeline.

        Args:
            signal_scores:      Raw signal scores per asset (pd.Series).
            returns_history:    Historical daily returns (DatetimeIndex × symbols).
                                Used by the risk model for covariance estimation.
            current_weights:    Current holdings {symbol: weight}.
            adv:                Average daily volume per asset (USD).
            asset_volatilities: Annualised volatility per asset.  If ``None``,
                                computed from ``returns_history``.

        Returns:
            :class:`PipelineResult` with target weights and diagnostics.

        Raises:
            ValueError: If inputs are insufficient for any pipeline stage.
        """
        # Intersect universes: only trade assets present in both signals and returns
        signal_symbols = set(signal_scores.dropna().index)
        return_symbols = set(returns_history.columns)
        universe = sorted(signal_symbols & return_symbols)

        if len(universe) < 2:
            raise ValueError(
                f"Need at least 2 assets in common between signals and returns, "
                f"got {len(universe)}"
            )

        # Filter to common universe
        scores = signal_scores.reindex(universe).fillna(0.0)
        returns = returns_history[universe]

        # Compute volatilities if not provided
        if asset_volatilities is not None:
            vols = asset_volatilities.reindex(universe).fillna(0.20)
        else:
            daily_std = returns.std()
            vols = daily_std * (252 ** 0.5)
            vols = vols.fillna(0.20)

        # Step 1: Alpha model — signals → expected returns
        alpha_result = self._alpha_model.forecast(scores, vols)

        # Step 2: Risk model — returns → covariance matrix
        risk_result = self._risk_model.estimate(returns)

        # Step 3: Cost-aware optimizer — (expected returns, cov) → weights
        optimizer_result = self._optimizer.optimize(
            symbols=universe,
            cov_matrix=risk_result.covariance,
            expected_returns=alpha_result.expected_returns,
            constraints=self._config.constraints,
            current_weights=current_weights,
            adv=adv,
        )

        return PipelineResult(
            target_weights=optimizer_result.weights,
            alpha_result=alpha_result,
            risk_result=risk_result,
            optimizer_result=optimizer_result,
            n_assets=len(universe),
            expected_return=optimizer_result.expected_return,
            risk=optimizer_result.risk,
            total_cost=optimizer_result.total_cost,
            turnover=optimizer_result.turnover,
        )
