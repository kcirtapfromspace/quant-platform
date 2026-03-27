"""Strategy ranking and comparison — multi-dimensional strategy evaluation.

Provides a unified framework for comparing strategies across the dimensions
that matter to the CIO:

  * **Risk-adjusted return**: Sharpe ratio, Sortino ratio, Calmar ratio.
  * **Alpha quality**: magnitude, persistence (t-stat), residual vol.
  * **Drawdown behaviour**: max drawdown, time to recovery, tail risk.
  * **Execution quality**: slippage cost, severe event frequency.
  * **Composite score**: weighted blend across all dimensions.

Each strategy is scored on each metric, normalised to [0, 1], then combined
into a composite score for ranking.

Usage::

    from quant.portfolio.strategy_ranking import StrategyRanker, RankingConfig

    ranker = StrategyRanker()
    result = ranker.rank(strategies)
    print(result.summary())
    print(result.rankings[0].name, result.rankings[0].composite_score)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RankingConfig:
    """Weights for composite strategy ranking.

    Each weight controls how much that dimension contributes to the
    composite score.  Weights are normalised internally so they need not
    sum to 1.0.

    Attributes:
        w_sharpe:         Weight for risk-adjusted return (Sharpe).
        w_sortino:        Weight for downside-risk-adjusted return (Sortino).
        w_alpha:          Weight for residual alpha magnitude.
        w_alpha_persistence: Weight for alpha t-statistic (persistence).
        w_max_drawdown:   Weight for maximum drawdown (lower is better).
        w_calmar:         Weight for Calmar ratio (return / max DD).
        w_execution:      Weight for execution quality score.
        w_consistency:    Weight for return consistency (positive month %).
    """

    w_sharpe: float = 0.20
    w_sortino: float = 0.10
    w_alpha: float = 0.15
    w_alpha_persistence: float = 0.10
    w_max_drawdown: float = 0.10
    w_calmar: float = 0.10
    w_execution: float = 0.10
    w_consistency: float = 0.15


# ---------------------------------------------------------------------------
# Strategy input
# ---------------------------------------------------------------------------


@dataclass
class StrategyMetrics:
    """Raw performance metrics for one strategy.

    Populate these from backtests, live results, or the strategy monitor.
    All fields are optional — missing data is handled gracefully with
    neutral scores.

    Attributes:
        name:               Strategy identifier.
        sharpe_ratio:       Annualised Sharpe ratio.
        sortino_ratio:      Annualised Sortino ratio.
        total_return:       Cumulative return (e.g. 0.15 = 15%).
        annualised_return:  Annualised return.
        max_drawdown:       Maximum drawdown (positive, e.g. 0.10 = 10%).
        calmar_ratio:       Annualised return / max drawdown.
        alpha:              Annualised residual alpha.
        alpha_t_stat:       t-statistic for alpha.
        residual_vol:       Annualised idiosyncratic volatility.
        execution_quality:  Execution quality score (0–1) from tracker.
        positive_months_pct: Fraction of months with positive returns (0–1).
        n_months:           Number of months of data.
    """

    name: str
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    total_return: float | None = None
    annualised_return: float | None = None
    max_drawdown: float | None = None
    calmar_ratio: float | None = None
    alpha: float | None = None
    alpha_t_stat: float | None = None
    residual_vol: float | None = None
    execution_quality: float | None = None
    positive_months_pct: float | None = None
    n_months: int | None = None


# ---------------------------------------------------------------------------
# Ranking output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyRank:
    """Scored and ranked strategy.

    Attributes:
        name:              Strategy name.
        rank:              1-based rank (1 = best).
        composite_score:   Weighted composite score (0–1).
        dimension_scores:  Per-dimension normalised scores (0–1).
    """

    name: str
    rank: int
    composite_score: float
    dimension_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class RankingResult:
    """Complete ranking output.

    Attributes:
        rankings:  Strategies sorted by composite score (best first).
        config:    Ranking configuration used.
    """

    rankings: list[StrategyRank] = field(default_factory=list)
    config: RankingConfig = field(default_factory=RankingConfig)

    @property
    def n_strategies(self) -> int:
        return len(self.rankings)

    def top(self, n: int = 5) -> list[StrategyRank]:
        """Return the top N strategies."""
        return self.rankings[:n]

    def by_name(self, name: str) -> StrategyRank | None:
        """Look up a strategy by name."""
        for r in self.rankings:
            if r.name == name:
                return r
        return None

    def summary(self) -> str:
        """Human-readable ranking summary."""
        if not self.rankings:
            return "Strategy Ranking: no strategies evaluated"

        lines = [
            "Strategy Ranking",
            "=" * 70,
            f"  {'Rank':<6}{'Strategy':<25}{'Score':<10}{'Sharpe':<10}"
            f"{'Alpha':<10}{'Exec':<10}",
            "-" * 70,
        ]

        for r in self.rankings:
            sharpe_s = f"{r.dimension_scores.get('sharpe', 0):.2f}"
            alpha_s = f"{r.dimension_scores.get('alpha', 0):.2f}"
            exec_s = f"{r.dimension_scores.get('execution', 0):.2f}"
            lines.append(
                f"  {r.rank:<6}{r.name:<25}{r.composite_score:<10.3f}"
                f"{sharpe_s:<10}{alpha_s:<10}{exec_s:<10}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------


class StrategyRanker:
    """Rank strategies across multiple performance dimensions.

    Args:
        config: Ranking configuration with dimension weights.
    """

    def __init__(self, config: RankingConfig | None = None) -> None:
        self._config = config or RankingConfig()

    def rank(self, strategies: list[StrategyMetrics]) -> RankingResult:
        """Score and rank a set of strategies.

        Args:
            strategies: Raw metrics for each strategy.

        Returns:
            :class:`RankingResult` with strategies sorted best-to-worst.
        """
        if not strategies:
            return RankingResult(config=self._config)

        # Score each strategy across all dimensions
        scored: list[tuple[str, float, dict[str, float]]] = []
        for s in strategies:
            dim_scores = self._dimension_scores(s, strategies)
            composite = self._composite(dim_scores)
            scored.append((s.name, composite, dim_scores))

        # Sort by composite score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        rankings = [
            StrategyRank(
                name=name,
                rank=i + 1,
                composite_score=score,
                dimension_scores=dims,
            )
            for i, (name, score, dims) in enumerate(scored)
        ]

        return RankingResult(rankings=rankings, config=self._config)

    # ── Dimension scoring ──────────────────────────────────────────

    def _dimension_scores(
        self, strategy: StrategyMetrics, all_strategies: list[StrategyMetrics]
    ) -> dict[str, float]:
        """Compute normalised [0, 1] scores for each dimension."""
        return {
            "sharpe": _norm_sharpe(strategy.sharpe_ratio),
            "sortino": _norm_sortino(strategy.sortino_ratio),
            "alpha": _norm_alpha(strategy.alpha),
            "alpha_persistence": _norm_t_stat(strategy.alpha_t_stat),
            "max_drawdown": _norm_drawdown(strategy.max_drawdown),
            "calmar": _norm_calmar(strategy.calmar_ratio),
            "execution": _norm_execution(strategy.execution_quality),
            "consistency": _norm_consistency(strategy.positive_months_pct),
        }

    def _composite(self, dim_scores: dict[str, float]) -> float:
        """Compute weighted composite score from dimension scores."""
        cfg = self._config
        weights = {
            "sharpe": cfg.w_sharpe,
            "sortino": cfg.w_sortino,
            "alpha": cfg.w_alpha,
            "alpha_persistence": cfg.w_alpha_persistence,
            "max_drawdown": cfg.w_max_drawdown,
            "calmar": cfg.w_calmar,
            "execution": cfg.w_execution,
            "consistency": cfg.w_consistency,
        }

        total_w = sum(weights.values())
        if total_w <= 0:
            return 0.5

        score = sum(
            dim_scores.get(dim, 0.5) * w for dim, w in weights.items()
        )
        return score / total_w


# ---------------------------------------------------------------------------
# Normalisation functions: raw metric → [0, 1]
# ---------------------------------------------------------------------------


def _norm_sharpe(v: float | None) -> float:
    """Normalise Sharpe ratio. 0 → 0.5, 2.0 → 0.9, -2.0 → 0.1."""
    if v is None:
        return 0.5
    return _sigmoid(v, midpoint=0.0, scale=1.0)


def _norm_sortino(v: float | None) -> float:
    """Normalise Sortino ratio. Similar to Sharpe but wider range."""
    if v is None:
        return 0.5
    return _sigmoid(v, midpoint=0.0, scale=0.7)


def _norm_alpha(v: float | None) -> float:
    """Normalise annualised alpha. 0 → 0.5, +0.05 → ~0.75."""
    if v is None:
        return 0.5
    return _sigmoid(v * 20, midpoint=0.0, scale=1.0)


def _norm_t_stat(v: float | None) -> float:
    """Normalise alpha t-statistic. 0 → 0.5, 2.0 → ~0.88."""
    if v is None:
        return 0.5
    return _sigmoid(v, midpoint=0.0, scale=1.0)


def _norm_drawdown(v: float | None) -> float:
    """Normalise max drawdown (lower is better). 0 → 1.0, 0.5 → ~0.1."""
    if v is None:
        return 0.5
    # Invert: low drawdown = high score
    return 1.0 - _sigmoid(v * 10, midpoint=1.5, scale=1.0)


def _norm_calmar(v: float | None) -> float:
    """Normalise Calmar ratio. 0 → 0.5, 2.0 → ~0.88."""
    if v is None:
        return 0.5
    return _sigmoid(v, midpoint=0.0, scale=1.0)


def _norm_execution(v: float | None) -> float:
    """Normalise execution quality score (already 0–1)."""
    if v is None:
        return 0.5
    return max(0.0, min(1.0, v))


def _norm_consistency(v: float | None) -> float:
    """Normalise positive months percentage. 0.5 → 0.5, 0.7 → ~0.75."""
    if v is None:
        return 0.5
    return max(0.0, min(1.0, v))


def _sigmoid(x: float, midpoint: float = 0.0, scale: float = 1.0) -> float:
    """Logistic sigmoid mapping (-inf, inf) → (0, 1).

    Args:
        x: Input value.
        midpoint: Value of x that maps to 0.5.
        scale: Controls steepness (higher = steeper).

    Returns:
        Value in (0, 1).
    """
    z = (x - midpoint) * scale
    # Clamp to avoid overflow
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + math.exp(-z))
