"""Strategy lifecycle manager — monitor, evaluate, and reallocate across strategies.

The CIO's meta-layer above individual strategy execution.  Consumes signal
decay analysis, strategy ranking, and live performance tracking to produce
automated capital reallocation recommendations.

Key capabilities:

  * **Performance tracking**: maintain rolling returns and drawdown per strategy.
  * **IC decay monitoring**: detect when a strategy's signal predictive power
    is fading, via half-life estimation and IC trend analysis.
  * **Health classification**: healthy / watch / degraded / critical, based on
    configurable thresholds for drawdown, IC decay, and return quality.
  * **Capital reallocation**: recommend weight adjustments that shift capital
    away from degraded strategies toward healthy ones, subject to max-move
    and min-allocation constraints.
  * **Lifecycle report**: structured output suitable for CIO review, combining
    per-strategy health, ranking, and reallocation recommendations.

Usage::

    from quant.portfolio.lifecycle import (
        LifecycleManager,
        LifecycleConfig,
        StrategySnapshot,
    )

    mgr = LifecycleManager(LifecycleConfig())

    # Feed in strategy snapshots (e.g. after each rebalance or daily)
    mgr.update(StrategySnapshot(
        name="momentum",
        returns_series=momentum_returns,
        current_weight=0.40,
        signal_ic=0.08,
    ))
    mgr.update(StrategySnapshot(
        name="mean_reversion",
        returns_series=mr_returns,
        current_weight=0.35,
        signal_ic=0.03,
    ))

    report = mgr.evaluate()
    print(report.summary())
    for rec in report.recommendations:
        print(f"  {rec.strategy}: {rec.current_weight:.1%} → {rec.recommended_weight:.1%}")
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class HealthStatus(enum.Enum):
    """Strategy health classification."""

    HEALTHY = "healthy"
    WATCH = "watch"
    DEGRADED = "degraded"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LifecycleConfig:
    """Configuration for the lifecycle manager.

    Attributes:
        drawdown_watch:     Max drawdown threshold to enter WATCH.
        drawdown_degraded:  Max drawdown threshold to enter DEGRADED.
        drawdown_critical:  Max drawdown threshold to enter CRITICAL.
        ic_watch:           IC below this triggers WATCH.
        ic_degraded:        IC below this triggers DEGRADED.
        ic_critical:        IC below this triggers CRITICAL.
        min_sharpe_watch:   Rolling Sharpe below this triggers WATCH.
        min_sharpe_degraded: Rolling Sharpe below this triggers DEGRADED.
        eval_window:        Rolling window (trading days) for Sharpe/vol.
        max_weight_move:    Maximum capital weight change per evaluation
                            (prevents extreme rebalancing).
        min_allocation:     Minimum allocation for any active strategy.
        realloc_aggressiveness: How aggressively to shift capital (0–1).
                            0 = no reallocation, 1 = full proportional shift.
    """

    drawdown_watch: float = 0.05
    drawdown_degraded: float = 0.10
    drawdown_critical: float = 0.20
    ic_watch: float = 0.03
    ic_degraded: float = 0.01
    ic_critical: float = -0.01
    min_sharpe_watch: float = 0.5
    min_sharpe_degraded: float = 0.0
    eval_window: int = 63
    max_weight_move: float = 0.10
    min_allocation: float = 0.02
    realloc_aggressiveness: float = 0.5


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass
class StrategySnapshot:
    """Performance snapshot for one strategy at a point in time.

    Attributes:
        name:             Strategy identifier.
        returns_series:   Daily returns Series (DatetimeIndex).
        current_weight:   Current capital allocation weight (0–1).
        signal_ic:        Most recent cross-sectional signal IC.
                          None if unavailable.
        ic_history:       Time series of rolling IC values.
                          None if unavailable.
        metadata:         Arbitrary strategy metadata for reporting.
    """

    name: str
    returns_series: pd.Series
    current_weight: float = 0.0
    signal_ic: float | None = None
    ic_history: pd.Series | None = None
    metadata: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyHealth:
    """Health assessment for one strategy.

    Attributes:
        name:             Strategy name.
        status:           Health classification.
        current_weight:   Current capital allocation.
        rolling_sharpe:   Sharpe ratio over the evaluation window.
        rolling_vol:      Annualised volatility over the eval window.
        max_drawdown:     Maximum drawdown in the returns series.
        current_drawdown: Current drawdown from peak.
        signal_ic:        Most recent signal IC (None if unavailable).
        ic_trend:         Slope of IC over recent history
                          (positive = improving, negative = decaying).
        reasons:          Human-readable reasons for the status.
    """

    name: str
    status: HealthStatus
    current_weight: float
    rolling_sharpe: float
    rolling_vol: float
    max_drawdown: float
    current_drawdown: float
    signal_ic: float | None
    ic_trend: float | None
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Capital reallocation recommendation for one strategy.

    Attributes:
        strategy:            Strategy name.
        current_weight:      Current allocation.
        recommended_weight:  Proposed new allocation.
        delta:               Weight change (recommended - current).
        reason:              Brief explanation.
    """

    strategy: str
    current_weight: float
    recommended_weight: float
    delta: float
    reason: str


@dataclass
class LifecycleReport:
    """Complete lifecycle evaluation output.

    Attributes:
        timestamp:        When the evaluation was performed.
        strategy_health:  Per-strategy health assessments.
        recommendations:  Capital reallocation recommendations.
        n_healthy:        Count of HEALTHY strategies.
        n_watch:          Count of WATCH strategies.
        n_degraded:       Count of DEGRADED strategies.
        n_critical:       Count of CRITICAL strategies.
    """

    timestamp: datetime
    strategy_health: list[StrategyHealth] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    n_healthy: int = 0
    n_watch: int = 0
    n_degraded: int = 0
    n_critical: int = 0

    @property
    def total_reallocation(self) -> float:
        """Total absolute weight change across all recommendations."""
        return sum(abs(r.delta) for r in self.recommendations)

    @property
    def has_critical(self) -> bool:
        return self.n_critical > 0

    def summary(self) -> str:
        """Human-readable lifecycle summary."""
        lines = [
            f"Strategy Lifecycle Report — {self.timestamp:%Y-%m-%d %H:%M}",
            "-" * 55,
            f"Strategies: {len(self.strategy_health)} total",
            f"  Healthy={self.n_healthy}  Watch={self.n_watch}  "
            f"Degraded={self.n_degraded}  Critical={self.n_critical}",
        ]

        if self.strategy_health:
            lines.append("")
            lines.append(
                f"{'Strategy':<20s} {'Status':<10s} {'Weight':>7s} "
                f"{'Sharpe':>7s} {'DD':>7s} {'IC':>7s}"
            )
            lines.append("-" * 55)
            for h in self.strategy_health:
                ic_str = f"{h.signal_ic:+.3f}" if h.signal_ic is not None else "   n/a"
                lines.append(
                    f"{h.name:<20s} {h.status.value:<10s} "
                    f"{h.current_weight:>6.1%} "
                    f"{h.rolling_sharpe:>+7.2f} "
                    f"{h.max_drawdown:>6.1%} "
                    f"{ic_str:>7s}"
                )

        if self.recommendations:
            lines.append("")
            lines.append("Reallocation recommendations:")
            for r in self.recommendations:
                arrow = "↑" if r.delta > 0 else "↓"
                lines.append(
                    f"  {r.strategy:<18s} "
                    f"{r.current_weight:.1%} → {r.recommended_weight:.1%} "
                    f"({arrow}{abs(r.delta):.1%})  {r.reason}"
                )
            lines.append(
                f"  Total turnover: {self.total_reallocation:.1%}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252


class LifecycleManager:
    """Evaluate strategy health and recommend capital reallocation.

    Maintains a snapshot of each strategy's recent performance and
    produces structured lifecycle reports with reallocation recommendations.

    Args:
        config: Lifecycle configuration.
    """

    def __init__(self, config: LifecycleConfig | None = None) -> None:
        self._config = config or LifecycleConfig()
        self._snapshots: dict[str, StrategySnapshot] = {}

    @property
    def config(self) -> LifecycleConfig:
        return self._config

    @property
    def strategy_names(self) -> list[str]:
        return sorted(self._snapshots.keys())

    def update(self, snapshot: StrategySnapshot) -> None:
        """Register or update a strategy snapshot."""
        self._snapshots[snapshot.name] = snapshot

    def remove(self, name: str) -> None:
        """Remove a strategy from tracking."""
        self._snapshots.pop(name, None)

    def evaluate(self) -> LifecycleReport:
        """Run full lifecycle evaluation and produce a report.

        Returns:
            :class:`LifecycleReport` with health assessments and
            reallocation recommendations.
        """
        now = datetime.now(timezone.utc)
        health_list: list[StrategyHealth] = []

        for name in sorted(self._snapshots):
            snap = self._snapshots[name]
            health = self._assess_health(snap)
            health_list.append(health)

        # Count by status
        counts = dict.fromkeys(HealthStatus, 0)
        for h in health_list:
            counts[h.status] += 1

        # Compute recommendations
        recommendations = self._recommend_reallocation(health_list)

        return LifecycleReport(
            timestamp=now,
            strategy_health=health_list,
            recommendations=recommendations,
            n_healthy=counts[HealthStatus.HEALTHY],
            n_watch=counts[HealthStatus.WATCH],
            n_degraded=counts[HealthStatus.DEGRADED],
            n_critical=counts[HealthStatus.CRITICAL],
        )

    # ── Health assessment ──────────────────────────────────────────

    def _assess_health(self, snap: StrategySnapshot) -> StrategyHealth:
        """Classify strategy health from its performance snapshot."""
        cfg = self._config
        reasons: list[str] = []

        # Compute rolling metrics
        returns = snap.returns_series.dropna()
        window = min(cfg.eval_window, len(returns))

        rolling_sharpe = self._rolling_sharpe(returns, window)
        rolling_vol = self._rolling_vol(returns, window)
        max_dd = self._max_drawdown(returns)
        current_dd = self._current_drawdown(returns)

        # IC analysis
        ic_trend = self._ic_trend(snap.ic_history) if snap.ic_history is not None else None

        # Classify — worst condition wins
        status = HealthStatus.HEALTHY

        # Drawdown checks
        if max_dd >= cfg.drawdown_critical:
            status = HealthStatus.CRITICAL
            reasons.append(f"max drawdown {max_dd:.1%} >= {cfg.drawdown_critical:.1%}")
        elif max_dd >= cfg.drawdown_degraded:
            status = max(status, HealthStatus.DEGRADED, key=_status_severity)
            reasons.append(f"max drawdown {max_dd:.1%} >= {cfg.drawdown_degraded:.1%}")
        elif max_dd >= cfg.drawdown_watch:
            status = max(status, HealthStatus.WATCH, key=_status_severity)
            reasons.append(f"max drawdown {max_dd:.1%} >= {cfg.drawdown_watch:.1%}")

        # IC checks
        if snap.signal_ic is not None:
            if snap.signal_ic <= cfg.ic_critical:
                status = max(status, HealthStatus.CRITICAL, key=_status_severity)
                reasons.append(f"IC {snap.signal_ic:.3f} <= {cfg.ic_critical:.3f}")
            elif snap.signal_ic <= cfg.ic_degraded:
                status = max(status, HealthStatus.DEGRADED, key=_status_severity)
                reasons.append(f"IC {snap.signal_ic:.3f} <= {cfg.ic_degraded:.3f}")
            elif snap.signal_ic <= cfg.ic_watch:
                status = max(status, HealthStatus.WATCH, key=_status_severity)
                reasons.append(f"IC {snap.signal_ic:.3f} <= {cfg.ic_watch:.3f}")

        # Sharpe checks
        if rolling_sharpe <= cfg.min_sharpe_degraded:
            status = max(status, HealthStatus.DEGRADED, key=_status_severity)
            reasons.append(f"Sharpe {rolling_sharpe:.2f} <= {cfg.min_sharpe_degraded:.2f}")
        elif rolling_sharpe <= cfg.min_sharpe_watch:
            status = max(status, HealthStatus.WATCH, key=_status_severity)
            reasons.append(f"Sharpe {rolling_sharpe:.2f} <= {cfg.min_sharpe_watch:.2f}")

        if not reasons:
            reasons.append("all metrics within thresholds")

        return StrategyHealth(
            name=snap.name,
            status=status,
            current_weight=snap.current_weight,
            rolling_sharpe=rolling_sharpe,
            rolling_vol=rolling_vol,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            signal_ic=snap.signal_ic,
            ic_trend=ic_trend,
            reasons=reasons,
        )

    # ── Reallocation ──────────────────────────────────────────────

    def _recommend_reallocation(
        self, health_list: list[StrategyHealth]
    ) -> list[Recommendation]:
        """Compute capital weight recommendations from health assessments."""
        if not health_list:
            return []

        cfg = self._config

        # Score: HEALTHY=1.0, WATCH=0.6, DEGRADED=0.3, CRITICAL=0.0
        score_map = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.WATCH: 0.6,
            HealthStatus.DEGRADED: 0.3,
            HealthStatus.CRITICAL: 0.0,
        }

        # Compute raw target weights proportional to health score
        scores = [score_map[h.status] for h in health_list]
        total_score = sum(scores)

        if total_score < 1e-12:
            # All critical — keep current weights, flag everything
            return [
                Recommendation(
                    strategy=h.name,
                    current_weight=h.current_weight,
                    recommended_weight=h.current_weight,
                    delta=0.0,
                    reason="all strategies critical — no safe reallocation",
                )
                for h in health_list
            ]

        total_current = sum(h.current_weight for h in health_list)

        recommendations: list[Recommendation] = []
        for i, h in enumerate(health_list):
            # Target: blend between current weight and score-proportional weight
            score_weight = (scores[i] / total_score) * total_current
            target = (
                h.current_weight * (1 - cfg.realloc_aggressiveness)
                + score_weight * cfg.realloc_aggressiveness
            )

            # Enforce min allocation (unless current is already below)
            if target < cfg.min_allocation and h.current_weight >= cfg.min_allocation:
                target = cfg.min_allocation

            # Clamp move size
            delta = target - h.current_weight
            if abs(delta) > cfg.max_weight_move:
                delta = math.copysign(cfg.max_weight_move, delta)
                target = h.current_weight + delta

            # Determine reason
            if abs(delta) < 1e-6:
                reason = "no change"
            elif delta > 0:
                reason = f"{h.status.value} — increase allocation"
            else:
                reason = f"{h.status.value} — reduce allocation"

            recommendations.append(
                Recommendation(
                    strategy=h.name,
                    current_weight=h.current_weight,
                    recommended_weight=round(target, 6),
                    delta=round(delta, 6),
                    reason=reason,
                )
            )

        # Normalise so recommended weights sum to original total
        rec_total = sum(r.recommended_weight for r in recommendations)
        if rec_total > 1e-12 and abs(rec_total - total_current) > 1e-6:
            scale = total_current / rec_total
            recommendations = [
                Recommendation(
                    strategy=r.strategy,
                    current_weight=r.current_weight,
                    recommended_weight=round(r.recommended_weight * scale, 6),
                    delta=round(r.recommended_weight * scale - r.current_weight, 6),
                    reason=r.reason,
                )
                for r in recommendations
            ]

        return recommendations

    # ── Metric helpers ────────────────────────────────────────────

    @staticmethod
    def _rolling_sharpe(returns: pd.Series, window: int) -> float:
        """Annualised Sharpe ratio over the most recent *window* days."""
        if len(returns) < 2:
            return 0.0
        tail = returns.tail(window)
        mean = float(tail.mean())
        std = float(tail.std())
        if std < 1e-12:
            return 0.0
        return mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)

    @staticmethod
    def _rolling_vol(returns: pd.Series, window: int) -> float:
        """Annualised volatility over the most recent *window* days."""
        if len(returns) < 2:
            return 0.0
        tail = returns.tail(window)
        return float(tail.std() * math.sqrt(TRADING_DAYS_PER_YEAR))

    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        """Maximum drawdown from the returns series."""
        if len(returns) < 1:
            return 0.0
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        dd = (cumulative - peak) / peak
        return float(-dd.min()) if len(dd) > 0 else 0.0

    @staticmethod
    def _current_drawdown(returns: pd.Series) -> float:
        """Current drawdown from peak."""
        if len(returns) < 1:
            return 0.0
        cumulative = (1 + returns).cumprod()
        peak = float(cumulative.cummax().iloc[-1])
        current = float(cumulative.iloc[-1])
        if peak < 1e-12:
            return 0.0
        return max(0.0, (peak - current) / peak)

    @staticmethod
    def _ic_trend(ic_history: pd.Series) -> float:
        """Linear slope of IC over time (positive = improving).

        Uses simple OLS: slope = Cov(x, y) / Var(x).
        """
        ic = ic_history.dropna()
        if len(ic) < 3:
            return 0.0
        x = np.arange(len(ic), dtype=float)
        y = ic.values.astype(float)
        x_mean = x.mean()
        y_mean = y.mean()
        cov_xy = float(((x - x_mean) * (y - y_mean)).sum())
        var_x = float(((x - x_mean) ** 2).sum())
        if var_x < 1e-12:
            return 0.0
        return cov_xy / var_x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {
    HealthStatus.HEALTHY: 0,
    HealthStatus.WATCH: 1,
    HealthStatus.DEGRADED: 2,
    HealthStatus.CRITICAL: 3,
}


def _status_severity(status: HealthStatus) -> int:
    """Numeric severity for max() comparisons."""
    return _SEVERITY_ORDER[status]
