"""Side-by-side strategy backtest comparison.

Takes multiple :class:`PortfolioBacktestReport` objects and produces a
structured comparison with:

  * **Metric table**: strategies × key metrics for quick visual scanning.
  * **Return correlation matrix**: pairwise correlation of daily returns.
  * **Rolling correlation**: time-varying pairwise correlation.
  * **Drawdown overlap**: how often strategies are in drawdown simultaneously.
  * **Auto-ranking**: populates :class:`StrategyMetrics` for use with the
    :class:`StrategyRanker` (QUA-46) framework.

Usage::

    from quant.backtest.comparison import BacktestComparator

    comparator = BacktestComparator()
    result = comparator.compare([report_a, report_b, report_c])
    print(result.summary())
    ranking = result.auto_rank()
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant.backtest.portfolio_backtest import PortfolioBacktestReport
from quant.portfolio.strategy_ranking import (
    RankingResult,
    StrategyMetrics,
    StrategyRanker,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ComparisonConfig:
    """Configuration for backtest comparison.

    Attributes:
        correlation_window:  Rolling window for correlation calculation.
        align_dates:         If True, align all strategies to common dates.
        annualisation_factor: Trading days per year for annualising metrics.
    """

    correlation_window: int = 63
    align_dates: bool = True
    annualisation_factor: int = 252


# ---------------------------------------------------------------------------
# Per-strategy metric row
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyRow:
    """Comparison metrics for one strategy.

    Attributes:
        name:             Strategy name.
        total_return:     Cumulative return.
        cagr:             Compound annual growth rate.
        sharpe:           Annualised Sharpe ratio.
        volatility:       Annualised volatility.
        max_drawdown:     Maximum drawdown.
        calmar:           Calmar ratio (CAGR / max_drawdown).
        avg_turnover:     Average turnover per rebalance.
        total_costs:      Total transaction costs.
        n_rebalances:     Number of rebalances.
        alpha:            Annualised factor alpha (or None).
        alpha_t_stat:     Alpha t-statistic (or None).
        r_squared:        Factor model R² (or None).
    """

    name: str
    total_return: float
    cagr: float
    sharpe: float
    volatility: float
    max_drawdown: float
    calmar: float
    avg_turnover: float
    total_costs: float
    n_rebalances: int
    alpha: float | None = None
    alpha_t_stat: float | None = None
    r_squared: float | None = None


# ---------------------------------------------------------------------------
# Comparison result
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Complete comparison across multiple strategy backtests.

    Attributes:
        strategies:          Per-strategy metric rows.
        correlation_matrix:  Pairwise return correlation matrix.
        rolling_correlations: Rolling pairwise correlations (DataFrame).
        drawdown_overlap:    Fraction of days both strategies in drawdown,
                            for each pair.
        aligned_returns:     Returns aligned to common dates.
        config:              Configuration used.
    """

    strategies: list[StrategyRow] = field(default_factory=list)
    correlation_matrix: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame()
    )
    rolling_correlations: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame()
    )
    drawdown_overlap: dict[tuple[str, str], float] = field(
        default_factory=dict
    )
    aligned_returns: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame()
    )
    config: ComparisonConfig = field(default_factory=ComparisonConfig)

    @property
    def n_strategies(self) -> int:
        return len(self.strategies)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.strategies]

    def best_by(self, metric: str) -> StrategyRow | None:
        """Return the strategy with the best value for a given metric.

        For max_drawdown, lower is better.  For all other metrics, higher
        is better.
        """
        if not self.strategies:
            return None

        reverse = metric != "max_drawdown"
        valid = [
            s for s in self.strategies if getattr(s, metric, None) is not None
        ]
        if not valid:
            return None

        return sorted(
            valid, key=lambda s: getattr(s, metric, 0), reverse=reverse
        )[0]

    def to_strategy_metrics(self) -> list[StrategyMetrics]:
        """Convert strategy rows to StrategyMetrics for use with StrategyRanker."""
        result: list[StrategyMetrics] = []
        for row in self.strategies:
            result.append(
                StrategyMetrics(
                    name=row.name,
                    sharpe_ratio=row.sharpe,
                    total_return=row.total_return,
                    annualised_return=row.cagr,
                    max_drawdown=row.max_drawdown,
                    calmar_ratio=row.calmar,
                    alpha=row.alpha,
                    alpha_t_stat=row.alpha_t_stat,
                )
            )
        return result

    def auto_rank(self) -> RankingResult:
        """Auto-rank strategies using the StrategyRanker framework (QUA-46)."""
        metrics = self.to_strategy_metrics()
        return StrategyRanker().rank(metrics)

    def summary(self) -> str:
        """Human-readable comparison summary."""
        if not self.strategies:
            return "Backtest Comparison: no strategies"

        lines = [
            "Backtest Comparison",
            "=" * 80,
            "",
            f"  {'Strategy':<20}{'Return':>10}{'CAGR':>10}{'Sharpe':>10}"
            f"{'Vol':>10}{'MaxDD':>10}{'Calmar':>10}",
            "-" * 80,
        ]

        for s in self.strategies:
            lines.append(
                f"  {s.name:<20}{s.total_return:>+9.2%} {s.cagr:>+9.2%} "
                f"{s.sharpe:>9.2f} {s.volatility:>9.2%} "
                f"{s.max_drawdown:>9.2%} {s.calmar:>9.2f}"
            )

        # Correlation matrix
        if not self.correlation_matrix.empty and self.n_strategies > 1:
            lines.append("")
            lines.append("Return Correlations")
            lines.append("-" * 80)
            for name in self.correlation_matrix.index:
                vals = "  ".join(
                    f"{self.correlation_matrix.loc[name, col]:+.2f}"
                    for col in self.correlation_matrix.columns
                )
                lines.append(f"  {name:<20}{vals}")

        # Drawdown overlap
        if self.drawdown_overlap:
            lines.append("")
            lines.append("Drawdown Overlap (fraction of days both in DD)")
            lines.append("-" * 80)
            for (a, b), pct in sorted(self.drawdown_overlap.items()):
                lines.append(f"  {a} vs {b}: {pct:.1%}")

        # Best-by highlights
        lines.append("")
        lines.append("Highlights")
        lines.append("-" * 80)
        for metric, label in [
            ("sharpe", "Best Sharpe"),
            ("cagr", "Best CAGR"),
            ("max_drawdown", "Lowest Drawdown"),
            ("calmar", "Best Calmar"),
        ]:
            best = self.best_by(metric)
            if best:
                val = getattr(best, metric)
                if metric in ("cagr", "max_drawdown", "total_return"):
                    lines.append(f"  {label:<25}: {best.name} ({val:.2%})")
                else:
                    lines.append(f"  {label:<25}: {best.name} ({val:.2f})")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparator engine
# ---------------------------------------------------------------------------


class BacktestComparator:
    """Compare multiple strategy backtest results side-by-side.

    Args:
        config: Comparison configuration.
    """

    def __init__(self, config: ComparisonConfig | None = None) -> None:
        self._config = config or ComparisonConfig()

    def compare(
        self, reports: list[PortfolioBacktestReport]
    ) -> ComparisonResult:
        """Run comparison across multiple backtest reports.

        Args:
            reports: List of backtest reports to compare.

        Returns:
            :class:`ComparisonResult` with metrics, correlations, and rankings.
        """
        if not reports:
            return ComparisonResult(config=self._config)

        # Build metric rows
        rows = [self._build_row(r) for r in reports]

        # Align returns
        aligned = self._align_returns(reports)

        # Correlation matrix
        corr_matrix = pd.DataFrame()
        rolling_corrs = pd.DataFrame()
        if len(reports) > 1 and not aligned.empty:
            corr_matrix = aligned.corr()
            rolling_corrs = self._rolling_correlations(aligned)

        # Drawdown overlap
        overlap = self._drawdown_overlap(reports, aligned)

        return ComparisonResult(
            strategies=rows,
            correlation_matrix=corr_matrix,
            rolling_correlations=rolling_corrs,
            drawdown_overlap=overlap,
            aligned_returns=aligned,
            config=self._config,
        )

    # ── Build metric row ───────────────────────────────────────────

    @staticmethod
    def _build_row(report: PortfolioBacktestReport) -> StrategyRow:
        """Extract comparison metrics from a single report."""
        calmar = (
            report.cagr / report.max_drawdown
            if report.max_drawdown > 1e-8
            else 0.0
        )

        alpha = None
        alpha_t = None
        r_sq = None
        if report.factor_attribution is not None:
            alpha = report.factor_attribution.alpha
            alpha_t = report.factor_attribution.alpha_t_stat
            r_sq = report.factor_attribution.r_squared

        return StrategyRow(
            name=report.name,
            total_return=report.total_return,
            cagr=report.cagr,
            sharpe=report.sharpe_ratio,
            volatility=report.volatility,
            max_drawdown=report.max_drawdown,
            calmar=calmar,
            avg_turnover=report.avg_turnover,
            total_costs=report.total_costs,
            n_rebalances=report.n_rebalances,
            alpha=alpha,
            alpha_t_stat=alpha_t,
            r_squared=r_sq,
        )

    # ── Return alignment ───────────────────────────────────────────

    def _align_returns(
        self, reports: list[PortfolioBacktestReport]
    ) -> pd.DataFrame:
        """Align strategy returns to common dates."""
        series_dict: dict[str, pd.Series] = {}
        for r in reports:
            if r.returns_series is not None and not r.returns_series.empty:
                series_dict[r.name] = r.returns_series

        if not series_dict:
            return pd.DataFrame()

        df = pd.DataFrame(series_dict)

        if self._config.align_dates:
            df = df.dropna()

        return df

    # ── Rolling correlations ───────────────────────────────────────

    def _rolling_correlations(self, aligned: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling pairwise correlations."""
        cols = list(aligned.columns)
        if len(cols) < 2:
            return pd.DataFrame()

        window = self._config.correlation_window
        results: dict[str, pd.Series] = {}

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pair = f"{cols[i]}_vs_{cols[j]}"
                rolling = aligned[cols[i]].rolling(window).corr(
                    aligned[cols[j]]
                )
                results[pair] = rolling

        return pd.DataFrame(results)

    # ── Drawdown overlap ───────────────────────────────────────────

    def _drawdown_overlap(
        self,
        reports: list[PortfolioBacktestReport],
        aligned: pd.DataFrame,
    ) -> dict[tuple[str, str], float]:
        """Compute fraction of days both strategies are in drawdown."""
        if len(reports) < 2 or aligned.empty:
            return {}

        # Build drawdown flags per strategy
        dd_flags: dict[str, pd.Series] = {}
        for r in reports:
            if r.equity_curve is not None and not r.equity_curve.empty:
                # Compute drawdown: running max - current
                cummax = r.equity_curve.cummax()
                dd = (r.equity_curve - cummax) / cummax
                in_dd = dd < -0.001  # more than 0.1% below peak
                dd_flags[r.name] = in_dd.reindex(aligned.index).fillna(False)

        # Pairwise overlap
        names = [r.name for r in reports if r.name in dd_flags]
        overlap: dict[tuple[str, str], float] = {}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                both = dd_flags[a] & dd_flags[b]
                n = len(both)
                overlap[(a, b)] = float(both.sum() / n) if n > 0 else 0.0

        return overlap
