"""Execution quality tracking — rolling slippage and cost monitoring per strategy.

Bridges the TCA module with the strategy monitor by maintaining rolling
execution quality metrics.  Provides:

  * **Per-strategy rolling slippage**: mean, median, and worst-case
    implementation shortfall over a configurable window.
  * **Execution cost budget**: tracks cumulative slippage cost against a
    per-strategy budget.
  * **Quality score**: composite 0–1 score summarising execution quality
    (1.0 = zero slippage, 0.0 = severe execution drag).

Usage::

    from quant.execution.quality_tracker import ExecutionQualityTracker

    tracker = ExecutionQualityTracker()
    tracker.record(strategy="momentum", slippage_bps=3.5, notional=50_000)
    quality = tracker.quality_score("momentum")
    stats = tracker.strategy_stats("momentum")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class QualityConfig:
    """Configuration for the execution quality tracker.

    Attributes:
        rolling_window:     Number of fills to keep per strategy for
                            rolling statistics.
        cost_budget_bps:    Maximum acceptable average slippage in basis
                            points.  Exceeding this degrades the quality
                            score toward zero.
        severe_slippage_bps: Single-fill slippage above this (in bps) is
                            flagged as a severe event.
    """

    rolling_window: int = 200
    cost_budget_bps: float = 10.0
    severe_slippage_bps: float = 50.0


# ---------------------------------------------------------------------------
# Per-fill record (lightweight)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FillRecord:
    """Minimal per-fill record for quality tracking.

    Attributes:
        slippage_bps: Signed implementation shortfall in basis points
                      (positive = cost).
        notional:     Dollar notional of the fill.
        timestamp:    When the fill occurred.
    """

    slippage_bps: float
    notional: float
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Per-strategy statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyExecStats:
    """Rolling execution quality statistics for one strategy.

    Attributes:
        strategy:              Strategy name.
        n_fills:               Number of fills in the rolling window.
        mean_slippage_bps:     Notional-weighted mean slippage (bps).
        median_slippage_bps:   Median slippage (bps, unweighted).
        max_slippage_bps:      Worst single-fill slippage in window.
        total_dollar_cost:     Cumulative dollar slippage cost.
        total_notional:        Total notional traded.
        severe_count:          Fills exceeding severe_slippage_bps.
        quality_score:         Composite execution quality (0–1).
    """

    strategy: str
    n_fills: int
    mean_slippage_bps: float
    median_slippage_bps: float
    max_slippage_bps: float
    total_dollar_cost: float
    total_notional: float
    severe_count: int
    quality_score: float


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class ExecutionQualityTracker:
    """Track rolling execution quality per strategy.

    Args:
        config: Quality tracker configuration.
    """

    def __init__(self, config: QualityConfig | None = None) -> None:
        self._config = config or QualityConfig()
        self._buffers: dict[str, list[FillRecord]] = {}

    @property
    def config(self) -> QualityConfig:
        return self._config

    def record(
        self,
        strategy: str,
        slippage_bps: float,
        notional: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a fill's execution quality.

        Args:
            strategy: Strategy name.
            slippage_bps: Signed slippage in basis points (positive = cost).
            notional: Dollar notional of the fill.
            timestamp: Fill timestamp (defaults to now).
        """
        fill = FillRecord(
            slippage_bps=slippage_bps,
            notional=notional,
            timestamp=timestamp or datetime.now(timezone.utc),
        )
        buf = self._buffers.setdefault(strategy, [])
        buf.append(fill)
        # Trim to rolling window
        if len(buf) > self._config.rolling_window:
            self._buffers[strategy] = buf[-self._config.rolling_window:]

    def strategy_stats(self, strategy: str) -> StrategyExecStats:
        """Compute rolling execution statistics for a strategy.

        Raises:
            KeyError: If the strategy has never been recorded.
        """
        buf = self._buffers[strategy]
        return self._compute_stats(strategy, buf)

    def quality_score(self, strategy: str) -> float:
        """Get the composite quality score for a strategy (0–1).

        Returns 1.0 for unknown strategies (no data = assume good).
        """
        buf = self._buffers.get(strategy)
        if not buf:
            return 1.0
        return self._compute_stats(strategy, buf).quality_score

    def all_stats(self) -> list[StrategyExecStats]:
        """Get stats for all tracked strategies."""
        return [
            self._compute_stats(name, buf)
            for name, buf in sorted(self._buffers.items())
        ]

    def reset(self, strategy: str) -> None:
        """Clear tracking data for a strategy."""
        self._buffers.pop(strategy, None)

    def reset_all(self) -> None:
        """Clear all tracking data."""
        self._buffers.clear()

    @property
    def strategy_names(self) -> list[str]:
        return sorted(self._buffers)

    # ── Internal ──────────────────────────────────────────────────────

    def _compute_stats(
        self, strategy: str, fills: list[FillRecord]
    ) -> StrategyExecStats:
        """Compute statistics from a list of fill records."""
        cfg = self._config
        n = len(fills)

        if n == 0:
            return StrategyExecStats(
                strategy=strategy,
                n_fills=0,
                mean_slippage_bps=0.0,
                median_slippage_bps=0.0,
                max_slippage_bps=0.0,
                total_dollar_cost=0.0,
                total_notional=0.0,
                severe_count=0,
                quality_score=1.0,
            )

        # Notional-weighted mean slippage
        total_notional = sum(f.notional for f in fills)
        if total_notional > 0:
            weighted_slip = sum(f.slippage_bps * f.notional for f in fills)
            mean_bps = weighted_slip / total_notional
        else:
            mean_bps = 0.0

        # Median (unweighted)
        sorted_slips = sorted(f.slippage_bps for f in fills)
        if n % 2 == 1:
            median_bps = sorted_slips[n // 2]
        else:
            median_bps = (sorted_slips[n // 2 - 1] + sorted_slips[n // 2]) / 2

        # Max (worst case)
        max_bps = max(f.slippage_bps for f in fills)

        # Dollar cost: sum of slippage_bps * notional / 10000
        total_dollar = sum(
            f.slippage_bps * f.notional / 10_000 for f in fills
        )

        # Severe event count
        severe = sum(
            1 for f in fills if f.slippage_bps > cfg.severe_slippage_bps
        )

        # Quality score: 1.0 when mean_bps <= 0 (improvement),
        # decays as mean_bps approaches budget, 0 when 2× budget
        score = self._compute_quality_score(mean_bps, severe, n)

        return StrategyExecStats(
            strategy=strategy,
            n_fills=n,
            mean_slippage_bps=mean_bps,
            median_slippage_bps=median_bps,
            max_slippage_bps=max_bps,
            total_dollar_cost=total_dollar,
            total_notional=total_notional,
            severe_count=severe,
            quality_score=score,
        )

    def _compute_quality_score(
        self, mean_bps: float, severe_count: int, n_fills: int
    ) -> float:
        """Compute composite quality score (0–1).

        Components:
        1. Slippage component: linear decay from 1.0 at 0 bps to 0.0 at
           2× cost_budget_bps.
        2. Severe penalty: reduces score by up to 0.3 based on fraction
           of fills that are severe events.
        """
        budget = self._config.cost_budget_bps

        # Slippage component (0–1)
        if mean_bps <= 0:
            slip_score = 1.0
        elif budget > 0:
            slip_score = max(0.0, 1.0 - mean_bps / (2 * budget))
        else:
            slip_score = 0.0

        # Severe penalty (up to 0.3)
        if n_fills > 0:
            severe_frac = severe_count / n_fills
            severe_penalty = min(0.3, severe_frac)
        else:
            severe_penalty = 0.0

        return max(0.0, min(1.0, slip_score - severe_penalty))

    def summary(self) -> str:
        """Human-readable summary of execution quality across strategies."""
        stats = self.all_stats()
        if not stats:
            return "Execution Quality Tracker: no fills recorded"

        lines = [
            "Execution Quality Tracker",
            "-" * 60,
        ]
        for s in stats:
            lines.append(
                f"  {s.strategy:20s}  fills={s.n_fills:4d}  "
                f"mean={s.mean_slippage_bps:+.1f}bps  "
                f"max={s.max_slippage_bps:+.1f}bps  "
                f"cost=${s.total_dollar_cost:+,.0f}  "
                f"score={s.quality_score:.2f}"
            )
        return "\n".join(lines)
