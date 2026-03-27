"""Per-strategy performance monitoring and drawdown management.

Tracks the health of individual strategy sleeves and provides graduated
capital scaling responses — from early warning through forced pause.
This sits between the emergency circuit breaker (all-or-nothing halt)
and the regime adapter (market-driven tilts), providing strategy-specific
performance-based capital management.

Response levels:
  1. **HEALTHY** — strategy performing within expectations, full capital.
  2. **WARNING** — early drawdown signal, capital reduced by ``warn_scale``.
  3. **REDUCED** — significant drawdown, capital reduced by ``reduce_scale``.
  4. **PAUSED** — drawdown exceeds pause threshold, capital set to zero.

Recovery: once a paused strategy recovers above ``recovery_threshold``
(expressed as a fraction of its peak recovered), it can be manually or
automatically reinstated.

Usage::

    from quant.risk.strategy_monitor import StrategyMonitor, MonitorConfig

    monitor = StrategyMonitor(MonitorConfig())
    monitor.update("momentum", portfolio_value=1_050_000)
    status = monitor.status("momentum")
    scale = monitor.capital_scale("momentum")  # 0.0 – 1.0

    # In orchestrator:
    effective_weight = base_weight * monitor.capital_scale(sleeve.name)
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class HealthLevel(enum.Enum):
    """Strategy health classification."""

    HEALTHY = "healthy"
    WARNING = "warning"
    REDUCED = "reduced"
    PAUSED = "paused"


@dataclass(frozen=True, slots=True)
class StrategyStatus:
    """Snapshot of a strategy's health.

    Attributes:
        name:               Strategy name.
        health:             Current health level.
        current_value:      Most recent portfolio value for this strategy.
        peak_value:         High water mark.
        drawdown:           Current drawdown (0 = at peak, 0.10 = 10% below).
        max_drawdown:       Worst drawdown observed.
        capital_scale:      Multiplier applied to base capital weight (0–1).
        losing_streak:      Consecutive periods with negative returns.
        rolling_sharpe:     Annualised Sharpe over the rolling window.
        n_updates:          Total update calls.
        last_updated:       Timestamp of most recent update.
    """

    name: str
    health: HealthLevel
    current_value: float
    peak_value: float
    drawdown: float
    max_drawdown: float
    capital_scale: float
    losing_streak: int
    rolling_sharpe: float
    n_updates: int
    last_updated: datetime


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252


@dataclass
class MonitorConfig:
    """Configuration for the strategy performance monitor.

    Attributes:
        warn_drawdown:       Drawdown threshold to enter WARNING (e.g. 0.05 = 5%).
        reduce_drawdown:     Drawdown threshold to enter REDUCED.
        pause_drawdown:      Drawdown threshold to enter PAUSED.
        warn_scale:          Capital multiplier when WARNING (0–1).
        reduce_scale:        Capital multiplier when REDUCED (0–1).
        recovery_threshold:  Fraction of peak that must be recovered before
                             a PAUSED strategy can be reinstated (e.g. 0.95 = must
                             recover to 95% of peak).
        rolling_window:      Number of observations for rolling Sharpe.
        sharpe_floor:        If rolling Sharpe drops below this, enter WARNING
                             regardless of drawdown.
        max_losing_streak:   Consecutive losing periods to trigger REDUCED.
        auto_reinstate:      If True, PAUSED strategies automatically reinstate
                             when recovery_threshold is met. If False, requires
                             explicit ``reinstate()`` call.
    """

    warn_drawdown: float = 0.05
    reduce_drawdown: float = 0.10
    pause_drawdown: float = 0.20
    warn_scale: float = 0.70
    reduce_scale: float = 0.40
    recovery_threshold: float = 0.95
    rolling_window: int = 63
    sharpe_floor: float = -0.50
    max_losing_streak: int = 10
    auto_reinstate: bool = True


# ---------------------------------------------------------------------------
# Internal per-strategy state
# ---------------------------------------------------------------------------


@dataclass
class _StrategyState:
    """Mutable per-strategy tracking state."""

    name: str
    peak_value: float = 0.0
    current_value: float = 0.0
    max_drawdown: float = 0.0
    health: HealthLevel = HealthLevel.HEALTHY
    losing_streak: int = 0
    n_updates: int = 0
    returns_buffer: list[float] = field(default_factory=list)
    last_updated: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    _paused_manually: bool = False


# ---------------------------------------------------------------------------
# Strategy monitor
# ---------------------------------------------------------------------------


class StrategyMonitor:
    """Track per-strategy health and compute capital scaling.

    Call ``update()`` after each rebalance or evaluation cycle for every
    active strategy.  The monitor classifies each strategy into a
    :class:`HealthLevel` and provides a capital multiplier for the
    orchestrator to apply.

    Args:
        config: Monitor configuration.
    """

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self._config = config or MonitorConfig()
        self._strategies: dict[str, _StrategyState] = {}

    @property
    def config(self) -> MonitorConfig:
        return self._config

    def update(self, name: str, portfolio_value: float) -> HealthLevel:
        """Record a new portfolio value for a strategy and classify health.

        Args:
            name: Strategy name (must be consistent across calls).
            portfolio_value: Current portfolio value for this strategy's
                capital allocation (not the whole fund).

        Returns:
            The updated :class:`HealthLevel`.
        """
        state = self._strategies.get(name)
        if state is None:
            state = _StrategyState(name=name, peak_value=portfolio_value)
            self._strategies[name] = state

        prev_value = state.current_value if state.n_updates > 0 else portfolio_value
        state.current_value = portfolio_value
        state.n_updates += 1
        state.last_updated = datetime.now(timezone.utc)

        # Track peak (high water mark)
        if portfolio_value > state.peak_value:
            state.peak_value = portfolio_value

        # Compute return for this period
        if prev_value > 0 and state.n_updates > 1:
            period_return = (portfolio_value - prev_value) / prev_value
            state.returns_buffer.append(period_return)
            # Trim to rolling window
            if len(state.returns_buffer) > self._config.rolling_window:
                state.returns_buffer = state.returns_buffer[
                    -self._config.rolling_window:
                ]

            # Track losing streak
            if period_return < 0:
                state.losing_streak += 1
            else:
                state.losing_streak = 0

        # Compute drawdown
        drawdown = self._compute_drawdown(state)
        state.max_drawdown = max(state.max_drawdown, drawdown)

        # Classify health
        state.health = self._classify(state, drawdown)

        return state.health

    def status(self, name: str) -> StrategyStatus:
        """Get the current status of a strategy.

        Raises:
            KeyError: If the strategy has never been updated.
        """
        state = self._strategies[name]
        dd = self._compute_drawdown(state)
        return StrategyStatus(
            name=state.name,
            health=state.health,
            current_value=state.current_value,
            peak_value=state.peak_value,
            drawdown=dd,
            max_drawdown=state.max_drawdown,
            capital_scale=self._scale_for_health(state.health),
            losing_streak=state.losing_streak,
            rolling_sharpe=self._rolling_sharpe(state),
            n_updates=state.n_updates,
            last_updated=state.last_updated,
        )

    def capital_scale(self, name: str) -> float:
        """Get the capital scaling factor for a strategy (0.0 – 1.0).

        Returns 1.0 for unknown strategies (conservative default: don't
        reduce capital for strategies we haven't seen yet).
        """
        state = self._strategies.get(name)
        if state is None:
            return 1.0
        return self._scale_for_health(state.health)

    def all_statuses(self) -> list[StrategyStatus]:
        """Get status for all tracked strategies."""
        return [self.status(name) for name in sorted(self._strategies)]

    def reinstate(self, name: str) -> bool:
        """Manually reinstate a PAUSED strategy to HEALTHY.

        Returns True if the strategy was paused and is now reinstated.
        """
        state = self._strategies.get(name)
        if state is None:
            return False
        if state.health != HealthLevel.PAUSED:
            return False

        state.health = HealthLevel.HEALTHY
        state._paused_manually = False
        return True

    def reset(self, name: str) -> None:
        """Reset all tracking state for a strategy (e.g. after strategy restart)."""
        if name in self._strategies:
            del self._strategies[name]

    def reset_all(self) -> None:
        """Reset all tracking state."""
        self._strategies.clear()

    @property
    def strategy_names(self) -> list[str]:
        """Names of all tracked strategies."""
        return sorted(self._strategies)

    # ── Private ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_drawdown(state: _StrategyState) -> float:
        """Compute current drawdown as a positive fraction (0 = at peak)."""
        if state.peak_value <= 0:
            return 0.0
        return max(0.0, (state.peak_value - state.current_value) / state.peak_value)

    def _classify(
        self, state: _StrategyState, drawdown: float
    ) -> HealthLevel:
        """Classify strategy health based on drawdown, Sharpe, and streaks."""
        cfg = self._config

        # Auto-reinstate: if paused and recovered enough, reinstate
        if (
            state.health == HealthLevel.PAUSED
            and cfg.auto_reinstate
            and not state._paused_manually
        ):
            recovery_ratio = (
                state.current_value / state.peak_value
                if state.peak_value > 0
                else 0.0
            )
            if recovery_ratio >= cfg.recovery_threshold:
                return HealthLevel.HEALTHY

        # Stay paused until explicitly reinstated (when auto_reinstate is off)
        if state.health == HealthLevel.PAUSED and not cfg.auto_reinstate:
            return HealthLevel.PAUSED

        # PAUSED: severe drawdown
        if drawdown >= cfg.pause_drawdown:
            return HealthLevel.PAUSED

        # REDUCED: significant drawdown or extreme losing streak
        if drawdown >= cfg.reduce_drawdown:
            return HealthLevel.REDUCED
        if state.losing_streak >= cfg.max_losing_streak:
            return HealthLevel.REDUCED

        # WARNING: early drawdown signal or Sharpe floor breach
        if drawdown >= cfg.warn_drawdown:
            return HealthLevel.WARNING

        rolling_sharpe = self._rolling_sharpe(state)
        if (
            len(state.returns_buffer) >= cfg.rolling_window // 2
            and rolling_sharpe < cfg.sharpe_floor
        ):
            return HealthLevel.WARNING

        return HealthLevel.HEALTHY

    def _scale_for_health(self, health: HealthLevel) -> float:
        """Map health level to capital scaling factor."""
        if health == HealthLevel.HEALTHY:
            return 1.0
        if health == HealthLevel.WARNING:
            return self._config.warn_scale
        if health == HealthLevel.REDUCED:
            return self._config.reduce_scale
        # PAUSED
        return 0.0

    @staticmethod
    def _rolling_sharpe(state: _StrategyState) -> float:
        """Compute annualised Sharpe ratio over the returns buffer."""
        returns = state.returns_buffer
        n = len(returns)
        if n < 5:
            return 0.0

        mean_ret = sum(returns) / n
        var = sum((r - mean_ret) ** 2 for r in returns) / (n - 1)
        std = math.sqrt(var) if var > 0 else 0.0

        if std < 1e-12:
            return 0.0

        return (mean_ret / std) * math.sqrt(TRADING_DAYS_PER_YEAR)
