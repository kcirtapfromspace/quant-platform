"""Multi-strategy comparison for CIO decision support.

Compares multiple strategy configurations side-by-side across risk-adjusted
return, drawdown, turnover, and capacity metrics.  Produces a ranked
leaderboard and pairwise comparison to help the CIO select and allocate
capital across strategies.

The comparator accepts pre-computed performance summaries (not raw backtest
reports) so it can compare any combination of strategies without coupling
to a specific backtest engine.

Usage::

    from quant.backtest.strategy_comparison import (
        StrategyComparator,
        StrategySummary,
    )

    summaries = [
        StrategySummary(name="MomentumV2", sharpe=1.8, cagr=0.12, ...),
        StrategySummary(name="MeanRev", sharpe=1.2, cagr=0.08, ...),
    ]
    result = StrategyComparator().compare(summaries)
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


@dataclass
class StrategySummary:
    """Pre-computed performance summary for one strategy.

    Attributes:
        name:               Strategy identifier.
        sharpe:             Annualised Sharpe ratio.
        cagr:               Compound annual growth rate.
        max_drawdown:       Maximum drawdown (negative number).
        annualised_vol:     Annualised volatility.
        calmar:             CAGR / abs(max_drawdown).
        avg_turnover:       Average one-way turnover per rebalance.
        total_return:       Cumulative total return.
        win_rate:           Fraction of profitable periods.
        profit_factor:      Gross profits / gross losses.
        n_trades:           Total number of trades.
        capacity_aum:       Estimated AUM capacity (optional).
        correlation_to_benchmark: Correlation to benchmark (optional).
    """

    name: str
    sharpe: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    annualised_vol: float = 0.0
    calmar: float = 0.0
    avg_turnover: float = 0.0
    total_return: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    capacity_aum: float | None = None
    correlation_to_benchmark: float | None = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class RankMetric(Enum):
    """Metric used for primary ranking."""

    SHARPE = "sharpe"
    CALMAR = "calmar"
    CAGR = "cagr"
    TOTAL_RETURN = "total_return"


@dataclass
class ComparisonConfig:
    """Configuration for strategy comparison.

    Attributes:
        rank_by:            Primary ranking metric.
        min_sharpe:         Minimum Sharpe to be considered viable.
        max_drawdown_limit: Strategies with worse drawdown are flagged.
        max_turnover_limit: Strategies with higher turnover are flagged.
    """

    rank_by: RankMetric = RankMetric.SHARPE
    min_sharpe: float = 0.0
    max_drawdown_limit: float = -0.30
    max_turnover_limit: float = 1.0


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyRanking:
    """Ranking of a single strategy.

    Attributes:
        name:       Strategy name.
        rank:       1-based rank (1 = best).
        score:      Value of the ranking metric.
        is_viable:  Passes minimum thresholds.
        flags:      Warning flags (e.g. "HIGH_DRAWDOWN").
    """

    name: str
    rank: int
    score: float
    is_viable: bool
    flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PairwiseComparison:
    """Head-to-head comparison between two strategies."""

    strategy_a: str
    strategy_b: str
    sharpe_diff: float
    cagr_diff: float
    drawdown_diff: float
    turnover_diff: float
    winner: str


@dataclass
class ComparisonResult:
    """Complete multi-strategy comparison.

    Attributes:
        rankings:           Ranked list of strategies.
        pairwise:           Pairwise comparisons.
        best_strategy:      Name of the top-ranked strategy.
        n_strategies:       Number of strategies compared.
        n_viable:           Number of viable strategies.
        metric_table:       Full metric table as list of dicts.
    """

    rankings: list[StrategyRanking]
    pairwise: list[PairwiseComparison] = field(repr=False, default_factory=list)
    best_strategy: str = ""
    n_strategies: int = 0
    n_viable: int = 0
    metric_table: list[dict] = field(repr=False, default_factory=list)

    def summary(self) -> str:
        """Return a human-readable comparison summary."""
        lines = [
            f"Strategy Comparison ({self.n_strategies} strategies, {self.n_viable} viable)",
            "=" * 80,
            "",
            f"Best strategy: {self.best_strategy}",
            "",
            f"{'Rank':>4s} {'Strategy':<20s} {'Sharpe':>7s} {'CAGR':>7s} "
            f"{'MaxDD':>7s} {'Calmar':>7s} {'Turn':>7s} {'Viable':>7s}",
            "-" * 80,
        ]
        for r in self.rankings:
            m = next(
                (mt for mt in self.metric_table if mt["name"] == r.name), {},
            )
            viable_str = "YES" if r.is_viable else "NO"
            flags_str = f" [{', '.join(r.flags)}]" if r.flags else ""
            lines.append(
                f"{r.rank:>4d} {r.name:<20s} "
                f"{m.get('sharpe', 0):>+7.2f} {m.get('cagr', 0):>7.2%} "
                f"{m.get('max_drawdown', 0):>7.2%} "
                f"{m.get('calmar', 0):>7.2f} "
                f"{m.get('avg_turnover', 0):>7.2%} "
                f"{viable_str:>7s}{flags_str}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


class StrategyComparator:
    """Compares multiple strategies and produces a ranked leaderboard.

    Args:
        config: Comparison configuration.
    """

    def __init__(self, config: ComparisonConfig | None = None) -> None:
        self._config = config or ComparisonConfig()

    @property
    def config(self) -> ComparisonConfig:
        return self._config

    def compare(
        self,
        strategies: list[StrategySummary],
    ) -> ComparisonResult:
        """Compare and rank strategies.

        Args:
            strategies: List of strategy performance summaries.

        Returns:
            :class:`ComparisonResult` with rankings and pairwise analysis.

        Raises:
            ValueError: If fewer than 1 strategy provided.
        """
        cfg = self._config

        if not strategies:
            raise ValueError("Need at least 1 strategy")

        # Build metric table
        table: list[dict] = []
        for s in strategies:
            table.append({
                "name": s.name,
                "sharpe": s.sharpe,
                "cagr": s.cagr,
                "max_drawdown": s.max_drawdown,
                "annualised_vol": s.annualised_vol,
                "calmar": s.calmar,
                "avg_turnover": s.avg_turnover,
                "total_return": s.total_return,
                "win_rate": s.win_rate,
                "profit_factor": s.profit_factor,
                "n_trades": s.n_trades,
                "capacity_aum": s.capacity_aum,
                "correlation_to_benchmark": s.correlation_to_benchmark,
            })

        # Rank
        rank_key = cfg.rank_by.value
        sorted_table = sorted(table, key=lambda m: -m.get(rank_key, 0))

        rankings: list[StrategyRanking] = []
        for i, m in enumerate(sorted_table):
            flags = self._compute_flags(m, cfg)
            is_viable = (
                m["sharpe"] >= cfg.min_sharpe
                and "HIGH_DRAWDOWN" not in flags
            )
            rankings.append(StrategyRanking(
                name=m["name"],
                rank=i + 1,
                score=m.get(rank_key, 0),
                is_viable=is_viable,
                flags=tuple(flags),
            ))

        # Pairwise comparisons
        pairwise: list[PairwiseComparison] = []
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                a, b = strategies[i], strategies[j]
                sharpe_diff = a.sharpe - b.sharpe
                cagr_diff = a.cagr - b.cagr
                dd_diff = a.max_drawdown - b.max_drawdown  # Less negative = better
                turn_diff = a.avg_turnover - b.avg_turnover

                # Winner by ranking metric
                a_score = getattr(a, rank_key, 0)
                b_score = getattr(b, rank_key, 0)
                winner = a.name if a_score >= b_score else b.name

                pairwise.append(PairwiseComparison(
                    strategy_a=a.name,
                    strategy_b=b.name,
                    sharpe_diff=sharpe_diff,
                    cagr_diff=cagr_diff,
                    drawdown_diff=dd_diff,
                    turnover_diff=turn_diff,
                    winner=winner,
                ))

        best = rankings[0].name if rankings else ""
        n_viable = sum(1 for r in rankings if r.is_viable)

        return ComparisonResult(
            rankings=rankings,
            pairwise=pairwise,
            best_strategy=best,
            n_strategies=len(strategies),
            n_viable=n_viable,
            metric_table=table,
        )

    @staticmethod
    def _compute_flags(
        metrics: dict, cfg: ComparisonConfig,
    ) -> list[str]:
        """Compute warning flags for a strategy."""
        flags: list[str] = []
        if metrics["max_drawdown"] < cfg.max_drawdown_limit:
            flags.append("HIGH_DRAWDOWN")
        if metrics["avg_turnover"] > cfg.max_turnover_limit:
            flags.append("HIGH_TURNOVER")
        if metrics["sharpe"] < cfg.min_sharpe:
            flags.append("LOW_SHARPE")
        return flags
