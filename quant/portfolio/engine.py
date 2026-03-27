"""Portfolio construction engine: the main orchestrator.

Ties together alpha combination, covariance estimation, portfolio optimisation,
constraint enforcement, and rebalancing into a single pipeline.

Usage::

    from quant.portfolio import PortfolioEngine, PortfolioConfig

    config = PortfolioConfig(
        optimization_method=OptimizationMethod.RISK_PARITY,
        constraints=PortfolioConstraints(long_only=True, max_weight=0.10),
        rebalance_threshold=0.02,
    )
    engine = PortfolioEngine(config)
    result = engine.construct(
        alpha_scores=alpha_dict,
        returns_history=returns_df,
        current_weights=current_dict,
        portfolio_value=1_000_000,
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from quant.portfolio.alpha import AlphaScore
from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.optimizers import (
    BaseOptimizer,
    OptimizationMethod,
    OptimizationResult,
    get_optimizer,
)
from quant.portfolio.rebalancer import RebalanceEngine, RebalanceResult


@dataclass
class PortfolioConfig:
    """Configuration for the portfolio construction engine.

    Attributes:
        optimization_method: Which optimizer to use.
        constraints:         Portfolio constraints.
        rebalance_threshold: Minimum total turnover to trigger a rebalance.
            If the turnover required is below this threshold, the current
            portfolio is kept unchanged.
        cov_lookback_days:   Number of trading days for covariance estimation.
        cov_shrinkage:       Ledoit-Wolf shrinkage intensity (0 = sample cov,
            1 = diagonal target). Set to None for automatic estimation.
        min_trade_weight:    Dead band for individual trades.
        optimizer_kwargs:    Extra kwargs passed to the optimizer constructor.
    """

    optimization_method: OptimizationMethod = OptimizationMethod.RISK_PARITY
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    rebalance_threshold: float = 0.01
    cov_lookback_days: int = 252
    cov_shrinkage: float | None = None
    min_trade_weight: float = 0.001
    optimizer_kwargs: dict = field(default_factory=dict)


@dataclass
class ConstructionResult:
    """Output of the portfolio construction pipeline.

    Attributes:
        optimization:     Raw optimizer output (weights, risk, etc.).
        rebalance:        Rebalance result (trade list, turnover).
        covariance:       Estimated covariance matrix used.
        alpha_vector:     Alpha scores used as expected returns.
        rebalance_triggered: Whether the rebalance threshold was met.
    """

    optimization: OptimizationResult
    rebalance: RebalanceResult
    covariance: pd.DataFrame
    alpha_vector: pd.Series
    rebalance_triggered: bool


class PortfolioEngine:
    """Main portfolio construction pipeline.

    Steps:
    1. Convert alpha scores to an expected-return vector.
    2. Estimate the covariance matrix from historical returns.
    3. Run the chosen optimizer to compute target weights.
    4. Check rebalance threshold — skip if turnover is trivial.
    5. Generate trade list via the rebalancer.
    """

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self._config = config or PortfolioConfig()
        self._optimizer: BaseOptimizer = get_optimizer(
            self._config.optimization_method,
            **self._config.optimizer_kwargs,
        )
        self._rebalancer = RebalanceEngine(
            min_trade_weight=self._config.min_trade_weight
        )

    @property
    def config(self) -> PortfolioConfig:
        return self._config

    def construct(
        self,
        alpha_scores: dict[str, AlphaScore],
        returns_history: pd.DataFrame,
        current_weights: dict[str, float] | None = None,
        portfolio_value: float = 1_000_000.0,
    ) -> ConstructionResult:
        """Run the full portfolio construction pipeline.

        Args:
            alpha_scores:    {symbol: AlphaScore} from the alpha combiner.
            returns_history: DataFrame of daily asset returns
                             (index=date, columns=symbols).
            current_weights: {symbol: weight} of current portfolio. None for
                             initial construction (all cash).
            portfolio_value: Total portfolio value in dollars.

        Returns:
            ConstructionResult with optimizer output, trades, and diagnostics.
        """
        if current_weights is None:
            current_weights = {}

        symbols = sorted(alpha_scores.keys())
        if not symbols:
            logger.warning("Portfolio construction: empty universe — returning no trades")
            empty_opt = OptimizationResult(
                weights={},
                method=self._config.optimization_method,
                risk=0.0,
                expected_return=0.0,
                diversification_ratio=1.0,
            )
            empty_reb = self._rebalancer.rebalance(
                current_weights, {}, portfolio_value, self._config.constraints
            )
            return ConstructionResult(
                optimization=empty_opt,
                rebalance=empty_reb,
                covariance=pd.DataFrame(),
                alpha_vector=pd.Series(dtype=float),
                rebalance_triggered=False,
            )

        # 1. Build alpha vector (expected returns proxy)
        alpha_vector = pd.Series(
            {sym: alpha_scores[sym].score for sym in symbols},
            dtype=float,
        )

        # 2. Estimate covariance matrix
        available_symbols = [s for s in symbols if s in returns_history.columns]
        if not available_symbols:
            raise ValueError(
                "No overlap between alpha universe and returns_history columns"
            )

        cov_matrix = self._estimate_covariance(
            returns_history[available_symbols]
        )

        # Filter to symbols with both alpha and covariance data
        symbols = available_symbols
        alpha_vector = alpha_vector.reindex(symbols).fillna(0.0)

        # 3. Optimise
        opt_result = self._optimizer.optimize(
            symbols=symbols,
            cov_matrix=cov_matrix,
            expected_returns=alpha_vector,
            constraints=self._config.constraints,
        )

        # 4. Check rebalance threshold
        all_syms = set(opt_result.weights) | set(current_weights)
        raw_turnover = sum(
            abs(opt_result.weights.get(s, 0.0) - current_weights.get(s, 0.0))
            for s in all_syms
        )
        rebalance_triggered = raw_turnover >= self._config.rebalance_threshold

        # 5. Generate trades
        if rebalance_triggered:
            target_weights = opt_result.weights
        else:
            target_weights = current_weights
            logger.debug(
                "Portfolio: turnover {:.4f} below threshold {:.4f} — holding",
                raw_turnover,
                self._config.rebalance_threshold,
            )

        rebalance_result = self._rebalancer.rebalance(
            current_weights=current_weights,
            target_weights=target_weights,
            portfolio_value=portfolio_value,
            constraints=self._config.constraints,
        )

        logger.info(
            "Portfolio constructed: {} assets | vol={:.1%} | turnover={:.1%} | {} trades",
            len(symbols),
            opt_result.risk,
            rebalance_result.turnover,
            len(rebalance_result.trades),
        )

        return ConstructionResult(
            optimization=opt_result,
            rebalance=rebalance_result,
            covariance=cov_matrix,
            alpha_vector=alpha_vector,
            rebalance_triggered=rebalance_triggered,
        )

    def _estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate the covariance matrix with optional Ledoit-Wolf shrinkage.

        Uses the trailing cov_lookback_days of data.  If shrinkage is specified,
        blends the sample covariance with a diagonal target.
        """
        lookback = self._config.cov_lookback_days
        recent = returns.iloc[-lookback:] if len(returns) > lookback else returns

        sample_cov = recent.cov() * TRADING_DAYS_PER_YEAR

        if self._config.cov_shrinkage is not None:
            delta = self._config.cov_shrinkage
        else:
            # Automatic Ledoit-Wolf-style shrinkage estimate
            delta = self._ledoit_wolf_shrinkage(recent)

        if delta > 0:
            target = np.diag(np.diag(sample_cov.values))
            shrunk = (1 - delta) * sample_cov.values + delta * target
            return pd.DataFrame(
                shrunk, index=sample_cov.index, columns=sample_cov.columns
            )

        return sample_cov

    @staticmethod
    def _ledoit_wolf_shrinkage(returns: pd.DataFrame) -> float:
        """Simplified Ledoit-Wolf optimal shrinkage intensity.

        Returns a shrinkage coefficient in [0, 1].
        """
        n_obs, n_assets = returns.shape
        if n_obs < 2 or n_assets < 2:
            return 0.5

        x_centered = returns.values - returns.values.mean(axis=0)
        sample = (x_centered.T @ x_centered) / n_obs

        # Shrinkage target: diagonal of sample cov
        target = np.diag(np.diag(sample))

        # Frobenius norms
        delta = sample - target
        d2 = np.sum(delta ** 2) / n_assets

        # Estimate optimal intensity
        # Simplified: use ratio of off-diagonal to total variance
        total_var = np.sum(sample ** 2) / n_assets
        if total_var < 1e-12:
            return 0.5

        intensity = min(1.0, max(0.0, d2 / total_var * (1.0 / n_obs)))
        return intensity


TRADING_DAYS_PER_YEAR = 252
