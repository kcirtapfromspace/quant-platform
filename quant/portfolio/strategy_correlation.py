"""Cross-strategy correlation monitor — detects strategy crowding risk.

Monitors pairwise correlations between strategy return streams and alerts
when strategies become too similar.  High inter-strategy correlation means
the portfolio is less diversified than the sleeve count suggests, increasing
tail risk during drawdowns.

The monitor produces a :class:`StrategyCorrelationReport` with:
  * Pairwise correlation matrix between all strategy pairs
  * Average and max inter-strategy correlation
  * Effective number of independent strategies (1 / w'Cw)
  * Crowding alerts for pairs exceeding a threshold

Usage::

    from quant.portfolio.strategy_correlation import (
        StrategyCorrelationConfig,
        StrategyCorrelationMonitor,
    )

    monitor = StrategyCorrelationMonitor(StrategyCorrelationConfig())
    report = monitor.evaluate(
        strategy_returns={"momentum": ret_mom, "mean_rev": ret_mr},
        capital_weights={"momentum": 0.6, "mean_rev": 0.4},
    )
    print(report.summary())

All computations are pure Python — no scipy dependency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ── Configuration ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class StrategyCorrelationConfig:
    """Configuration for cross-strategy correlation monitoring.

    Attributes:
        window: Rolling window for correlation computation (trading days).
            Only the most recent ``window`` overlapping observations are used.
        min_observations: Minimum overlapping data points required before
            correlation is considered meaningful.
        avg_corr_warn: Average pairwise correlation above which the report
            is flagged as *elevated*.
        avg_corr_critical: Average pairwise correlation above which the
            report is flagged as *critical*.
        crowding_threshold: Pairwise correlation threshold to generate a
            crowding alert for a specific strategy pair.
        min_effective_strategies: Minimum effective number of independent
            strategies.  Below this the report is elevated even if average
            correlation is moderate.
    """

    window: int = 63
    min_observations: int = 21
    avg_corr_warn: float = 0.50
    avg_corr_critical: float = 0.70
    crowding_threshold: float = 0.80
    min_effective_strategies: float = 1.5


# ── Report types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CrowdingAlert:
    """Alert for a pair of strategies with dangerously high correlation."""

    strategy_a: str
    strategy_b: str
    correlation: float
    message: str


@dataclass
class StrategyCorrelationReport:
    """Output of a strategy-correlation evaluation.

    Attributes:
        timestamp: When the report was generated.
        n_strategies: Number of strategies evaluated.
        avg_pairwise_corr: Average pairwise correlation across all pairs.
        max_pairwise_corr: Maximum pairwise correlation observed.
        max_corr_pair: The two strategies with the highest correlation.
        effective_strategies: Effective number of independent strategies,
            analogous to effective N for assets but at the strategy level.
            Computed as 1 / (w' C w) where w are capital weights and C is
            the strategy correlation matrix.
        correlation_matrix: ``{strat_a: {strat_b: rho}}``.
        crowding_alerts: Alerts for pairs exceeding the crowding threshold.
        level: Risk level — ``"normal"``, ``"elevated"``, or ``"critical"``.
        n_observations: Number of overlapping return observations used.
    """

    timestamp: datetime
    n_strategies: int = 0
    avg_pairwise_corr: float = 0.0
    max_pairwise_corr: float = 0.0
    max_corr_pair: tuple[str, str] = ("", "")
    effective_strategies: float = 0.0
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    crowding_alerts: list[CrowdingAlert] = field(default_factory=list)
    level: str = "normal"
    n_observations: int = 0

    def summary(self) -> str:
        """Human-readable summary of the correlation report."""
        parts = [
            f"Strategy Correlation ({self.n_strategies} strategies, "
            f"{self.n_observations} obs): "
            f"avg_corr={self.avg_pairwise_corr:.3f} "
            f"max_corr={self.max_pairwise_corr:.3f} "
            f"eff_N={self.effective_strategies:.1f} "
            f"[{self.level.upper()}]",
        ]
        if self.max_corr_pair[0]:
            parts.append(
                f"  Most correlated: {self.max_corr_pair[0]} / "
                f"{self.max_corr_pair[1]} = {self.max_pairwise_corr:.3f}"
            )
        for alert in self.crowding_alerts:
            parts.append(f"  CROWDING: {alert.message}")
        return "\n".join(parts)


# ── Pure-Python math helpers ─────────────────────────────────────────────────


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation between two equal-length return series."""
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    mx, my = _mean(xs[:n]), _mean(ys[:n])
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if dx < 1e-12 or dy < 1e-12:
        return 0.0
    return num / (dx * dy)


# ── Monitor ──────────────────────────────────────────────────────────────────


class StrategyCorrelationMonitor:
    """Monitors inter-strategy correlation to detect crowding risk.

    Accepts strategy return series and computes a pairwise correlation
    matrix across strategies.  Produces a :class:`StrategyCorrelationReport`
    with crowding alerts and an effective-strategy-count metric.

    Args:
        config: Monitoring configuration.  Defaults to sensible thresholds.
    """

    def __init__(self, config: StrategyCorrelationConfig | None = None) -> None:
        self._config = config or StrategyCorrelationConfig()
        # Incremental buffer: {strategy_name: list[float]} for single-observation updates
        self._buffer: dict[str, list[float]] = {}

    @property
    def config(self) -> StrategyCorrelationConfig:
        return self._config

    # ── Primary API: evaluate from full return series ────────────────────

    def evaluate(
        self,
        strategy_returns: dict[str, list[float]],
        capital_weights: dict[str, float] | None = None,
        timestamp: datetime | None = None,
    ) -> StrategyCorrelationReport:
        """Evaluate inter-strategy correlation from return series.

        Args:
            strategy_returns: ``{strategy_name: [daily_returns]}`` where each
                list contains one float per day.  Only overlapping observations
                (matching indices) are used.
            capital_weights: ``{strategy_name: weight}`` — capital allocation
                per strategy.  If None, equal-weight is assumed.
            timestamp: Report timestamp.  Defaults to now (UTC).

        Returns:
            :class:`StrategyCorrelationReport` with correlation metrics and
            crowding alerts.
        """
        ts = timestamp or datetime.now(timezone.utc)
        names = sorted(strategy_returns.keys())
        n = len(names)

        if n < 2:
            return StrategyCorrelationReport(
                timestamp=ts,
                n_strategies=n,
                effective_strategies=float(n),
                level="normal",
            )

        # Determine overlapping observation count (use min length, capped by window)
        lengths = [len(strategy_returns[s]) for s in names]
        n_obs = min(min(lengths), self._config.window)

        if n_obs < self._config.min_observations:
            return StrategyCorrelationReport(
                timestamp=ts,
                n_strategies=n,
                effective_strategies=float(n),
                n_observations=n_obs,
                level="normal",
            )

        # Extract tail of each series (most recent n_obs observations)
        series: dict[str, list[float]] = {}
        for name in names:
            raw = strategy_returns[name]
            series[name] = raw[-n_obs:]

        # Compute pairwise correlation matrix
        corr_matrix: dict[str, dict[str, float]] = {}
        pair_corrs: list[float] = []
        max_corr = -2.0
        max_pair = (names[0], names[1])

        for i, si in enumerate(names):
            corr_matrix[si] = {}
            for j, sj in enumerate(names):
                if i == j:
                    corr_matrix[si][sj] = 1.0
                elif j < i:
                    corr_matrix[si][sj] = corr_matrix[sj][si]
                else:
                    rho = _pearson(series[si], series[sj])
                    corr_matrix[si][sj] = rho
                    pair_corrs.append(rho)
                    if rho > max_corr:
                        max_corr = rho
                        max_pair = (si, sj)

        avg_corr = _mean(pair_corrs) if pair_corrs else 0.0

        # Effective number of independent strategies: 1 / (w' C w)
        if capital_weights is None:
            w_val = 1.0 / n
            weights = dict.fromkeys(names, w_val)
        else:
            # Normalise to active strategies
            active_w = {s: abs(capital_weights.get(s, 0.0)) for s in names}
            total_w = sum(active_w.values())
            if total_w > 0:
                weights = {s: w / total_w for s, w in active_w.items()}
            else:
                weights = dict.fromkeys(names, 1.0 / n)

        wcw = 0.0
        for si in names:
            for sj in names:
                rho = corr_matrix.get(si, {}).get(sj, 0.0)
                wcw += weights[si] * weights[sj] * rho

        effective_n = 1.0 / wcw if wcw > 1e-12 else float(n)

        # Crowding alerts
        alerts: list[CrowdingAlert] = []
        for i, si in enumerate(names):
            for j in range(i + 1, len(names)):
                sj = names[j]
                rho = corr_matrix[si][sj]
                if rho >= self._config.crowding_threshold:
                    alerts.append(
                        CrowdingAlert(
                            strategy_a=si,
                            strategy_b=sj,
                            correlation=rho,
                            message=(
                                f"{si} / {sj} corr={rho:.3f} "
                                f">= threshold {self._config.crowding_threshold:.2f}"
                            ),
                        )
                    )

        # Determine risk level
        cfg = self._config
        if avg_corr >= cfg.avg_corr_critical:
            level = "critical"
        elif avg_corr >= cfg.avg_corr_warn or effective_n < cfg.min_effective_strategies:
            level = "elevated"
        else:
            level = "normal"

        return StrategyCorrelationReport(
            timestamp=ts,
            n_strategies=n,
            avg_pairwise_corr=avg_corr,
            max_pairwise_corr=max_corr if max_corr > -2.0 else 0.0,
            max_corr_pair=max_pair,
            effective_strategies=effective_n,
            correlation_matrix=corr_matrix,
            crowding_alerts=alerts,
            level=level,
            n_observations=n_obs,
        )

    # ── Incremental API: accumulate one observation per cycle ────────────

    def update(self, strategy_returns: dict[str, float]) -> None:
        """Add a single return observation per strategy.

        Call this once per rebalance cycle.  Maintains a rolling buffer
        capped at ``config.window`` observations.

        Args:
            strategy_returns: ``{strategy_name: return}`` — one return
                value per strategy for the current period.
        """
        window = self._config.window
        for name, ret in strategy_returns.items():
            if name not in self._buffer:
                self._buffer[name] = []
            self._buffer[name].append(ret)
            if len(self._buffer[name]) > window:
                self._buffer[name] = self._buffer[name][-window:]

    def evaluate_incremental(
        self,
        capital_weights: dict[str, float] | None = None,
        timestamp: datetime | None = None,
    ) -> StrategyCorrelationReport:
        """Evaluate correlation from the accumulated incremental buffer.

        Uses the returns accumulated via :meth:`update`.

        Args:
            capital_weights: Capital allocation weights per strategy.
            timestamp: Report timestamp.

        Returns:
            :class:`StrategyCorrelationReport`.
        """
        return self.evaluate(
            strategy_returns=self._buffer,
            capital_weights=capital_weights,
            timestamp=timestamp,
        )

    def reset(self) -> None:
        """Clear the incremental buffer."""
        self._buffer.clear()

    @property
    def buffer_depth(self) -> int:
        """Number of observations in the incremental buffer (min across strategies)."""
        if not self._buffer:
            return 0
        return min(len(v) for v in self._buffer.values())
