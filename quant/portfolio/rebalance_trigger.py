"""Rebalance trigger engine for portfolio management.

Determines when a portfolio should be rebalanced by evaluating a set of
configurable trigger conditions.  Each trigger fires independently; any
trigger firing is sufficient to recommend a rebalance.

Trigger types:

  * **Calendar** — periodic rebalance on a fixed schedule
    (daily, weekly, monthly, quarterly).
  * **Drift** — current weights have drifted beyond a threshold from
    target weights (measured by max absolute deviation or total
    turnover distance).
  * **Risk** — portfolio risk has exceeded a volatility ceiling.
  * **Signal** — alpha signal has changed enough to warrant a new
    portfolio construction.

Usage::

    from quant.portfolio.rebalance_trigger import (
        RebalanceTrigger,
        TriggerConfig,
    )

    trigger = RebalanceTrigger(TriggerConfig(
        calendar_frequency="weekly",
        max_drift=0.05,
        max_total_drift=0.20,
    ))
    decision = trigger.evaluate(
        current_date=today,
        last_rebalance_date=last_rebal,
        current_weights=current,
        target_weights=target,
    )
    if decision.should_rebalance:
        # Run construction pipeline and execute trades
        ...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class CalendarFrequency(str, Enum):
    """Periodic rebalance schedule."""

    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    NONE = "none"


@dataclass
class TriggerConfig:
    """Configuration for rebalance triggers.

    Attributes:
        calendar_frequency: Periodic rebalance schedule.
        max_drift:          Maximum single-position absolute weight drift
                            before triggering a rebalance.
        max_total_drift:    Maximum total portfolio drift (sum of absolute
                            weight deviations) before triggering.
        max_volatility:     Maximum portfolio annualised volatility; exceeding
                            this triggers a risk rebalance.  ``None`` disables.
        signal_change_threshold: Minimum rank correlation change in signals
                                 to trigger a signal-based rebalance.
                                 ``None`` disables.
        min_days_between:   Minimum calendar days between rebalances to
                            prevent excessive trading.
    """

    calendar_frequency: CalendarFrequency = CalendarFrequency.WEEKLY
    max_drift: float = 0.05
    max_total_drift: float = 0.20
    max_volatility: float | None = None
    signal_change_threshold: float | None = None
    min_days_between: int = 1


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


class TriggerReason(str, Enum):
    """Why a rebalance was triggered."""

    CALENDAR = "calendar"
    POSITION_DRIFT = "position_drift"
    TOTAL_DRIFT = "total_drift"
    RISK_BREACH = "risk_breach"
    SIGNAL_CHANGE = "signal_change"


@dataclass(frozen=True, slots=True)
class TriggerDetail:
    """Detail for one trigger evaluation.

    Attributes:
        reason:     Trigger type.
        fired:      Whether this trigger recommends a rebalance.
        value:      Current metric value.
        threshold:  Threshold that was (or wasn't) breached.
        message:    Human-readable explanation.
    """

    reason: TriggerReason
    fired: bool
    value: float
    threshold: float
    message: str


@dataclass
class TriggerDecision:
    """Complete rebalance trigger decision.

    Attributes:
        should_rebalance:   True if any trigger fired.
        triggers:           Individual trigger evaluations.
        fired_triggers:     Only the triggers that fired.
        days_since_last:    Calendar days since last rebalance.
        max_position_drift: Largest single-position absolute drift.
        total_drift:        Sum of absolute weight deviations.
    """

    should_rebalance: bool = False
    triggers: list[TriggerDetail] = field(default_factory=list)
    fired_triggers: list[TriggerDetail] = field(default_factory=list)
    days_since_last: int = 0
    max_position_drift: float = 0.0
    total_drift: float = 0.0

    def summary(self) -> str:
        """Return a human-readable trigger summary."""
        status = "REBALANCE" if self.should_rebalance else "HOLD"
        lines = [
            f"Rebalance Decision: {status}",
            "=" * 60,
            "",
            f"Days since last rebalance: {self.days_since_last}",
            f"Max position drift       : {self.max_position_drift:.2%}",
            f"Total portfolio drift     : {self.total_drift:.2%}",
            "",
            "Trigger evaluations:",
        ]
        for t in self.triggers:
            flag = ">>>" if t.fired else "   "
            lines.append(f"  {flag} [{t.reason.value}] {t.message}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trigger engine
# ---------------------------------------------------------------------------


class RebalanceTrigger:
    """Evaluates whether a portfolio should be rebalanced.

    Checks calendar schedule, weight drift, risk thresholds, and signal
    changes.  Any single trigger firing is sufficient to recommend a
    rebalance, subject to the minimum days-between cooldown.

    Args:
        config: Trigger configuration.
    """

    def __init__(self, config: TriggerConfig | None = None) -> None:
        self._config = config or TriggerConfig()

    @property
    def config(self) -> TriggerConfig:
        return self._config

    def evaluate(
        self,
        current_date: date,
        last_rebalance_date: date | None = None,
        current_weights: dict[str, float] | None = None,
        target_weights: dict[str, float] | None = None,
        portfolio_volatility: float | None = None,
        signal_rank_correlation: float | None = None,
    ) -> TriggerDecision:
        """Evaluate all rebalance triggers.

        Args:
            current_date:           Today's date.
            last_rebalance_date:    Date of the most recent rebalance.
                                    ``None`` means never rebalanced.
            current_weights:        Current portfolio weights.
            target_weights:         Target portfolio weights from the last
                                    construction.
            portfolio_volatility:   Current annualised portfolio volatility.
            signal_rank_correlation: Rank correlation between current and
                                     previous signal scores (1.0 = identical).

        Returns:
            :class:`TriggerDecision` with rebalance recommendation.
        """
        cfg = self._config

        # Days since last rebalance
        if last_rebalance_date is not None:
            days_since = (current_date - last_rebalance_date).days
        else:
            days_since = 999  # Never rebalanced → always trigger

        # Compute drift metrics
        max_drift, total_drift = self._compute_drift(
            current_weights or {}, target_weights or {},
        )

        triggers: list[TriggerDetail] = []

        # 1. Calendar trigger
        triggers.append(self._check_calendar(current_date, last_rebalance_date, days_since))

        # 2. Position drift trigger
        triggers.append(self._check_position_drift(max_drift))

        # 3. Total drift trigger
        triggers.append(self._check_total_drift(total_drift))

        # 4. Risk trigger
        if cfg.max_volatility is not None:
            triggers.append(self._check_risk(portfolio_volatility))

        # 5. Signal change trigger
        if cfg.signal_change_threshold is not None:
            triggers.append(self._check_signal(signal_rank_correlation))

        # Any trigger fires → rebalance (subject to cooldown)
        any_fired = any(t.fired for t in triggers)
        cooldown_ok = days_since >= cfg.min_days_between

        should_rebalance = any_fired and cooldown_ok
        fired = [t for t in triggers if t.fired]

        return TriggerDecision(
            should_rebalance=should_rebalance,
            triggers=triggers,
            fired_triggers=fired,
            days_since_last=days_since,
            max_position_drift=max_drift,
            total_drift=total_drift,
        )

    # ── Trigger checks ─────────────────────────────────────────────

    def _check_calendar(
        self,
        current_date: date,
        last_rebalance_date: date | None,
        days_since: int,
    ) -> TriggerDetail:
        """Check calendar-based trigger."""
        cfg = self._config
        freq = cfg.calendar_frequency

        if freq == CalendarFrequency.NONE:
            return TriggerDetail(
                reason=TriggerReason.CALENDAR,
                fired=False, value=0, threshold=0,
                message="Calendar trigger disabled",
            )

        threshold_days = _calendar_days(freq)

        if last_rebalance_date is None:
            return TriggerDetail(
                reason=TriggerReason.CALENDAR,
                fired=True, value=float(days_since), threshold=float(threshold_days),
                message=f"Never rebalanced (schedule: {freq.value})",
            )

        fired = days_since >= threshold_days
        return TriggerDetail(
            reason=TriggerReason.CALENDAR,
            fired=fired,
            value=float(days_since),
            threshold=float(threshold_days),
            message=(
                f"{days_since}d since last rebalance "
                f"(threshold: {threshold_days}d / {freq.value})"
            ),
        )

    def _check_position_drift(self, max_drift: float) -> TriggerDetail:
        """Check single-position drift trigger."""
        cfg = self._config
        fired = max_drift > cfg.max_drift
        return TriggerDetail(
            reason=TriggerReason.POSITION_DRIFT,
            fired=fired,
            value=max_drift,
            threshold=cfg.max_drift,
            message=(
                f"Max position drift {max_drift:.2%} "
                f"(threshold: {cfg.max_drift:.2%})"
            ),
        )

    def _check_total_drift(self, total_drift: float) -> TriggerDetail:
        """Check total portfolio drift trigger."""
        cfg = self._config
        fired = total_drift > cfg.max_total_drift
        return TriggerDetail(
            reason=TriggerReason.TOTAL_DRIFT,
            fired=fired,
            value=total_drift,
            threshold=cfg.max_total_drift,
            message=(
                f"Total drift {total_drift:.2%} "
                f"(threshold: {cfg.max_total_drift:.2%})"
            ),
        )

    def _check_risk(self, portfolio_volatility: float | None) -> TriggerDetail:
        """Check risk-based trigger."""
        cfg = self._config
        threshold = cfg.max_volatility or 0.0
        vol = portfolio_volatility if portfolio_volatility is not None else 0.0
        fired = vol > threshold
        return TriggerDetail(
            reason=TriggerReason.RISK_BREACH,
            fired=fired,
            value=vol,
            threshold=threshold,
            message=(
                f"Portfolio vol {vol:.2%} "
                f"(threshold: {threshold:.2%})"
            ),
        )

    def _check_signal(self, rank_corr: float | None) -> TriggerDetail:
        """Check signal-change trigger.

        Fires when the rank correlation between current and previous
        signals drops below (1 - threshold).
        """
        cfg = self._config
        threshold = cfg.signal_change_threshold or 0.0
        corr = rank_corr if rank_corr is not None else 1.0
        # Signal has changed enough when correlation drops below 1 - threshold
        change = 1.0 - corr
        fired = change > threshold
        return TriggerDetail(
            reason=TriggerReason.SIGNAL_CHANGE,
            fired=fired,
            value=change,
            threshold=threshold,
            message=(
                f"Signal change {change:.2%} (1 - rank_corr) "
                f"(threshold: {threshold:.2%})"
            ),
        )

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _compute_drift(
        current: dict[str, float],
        target: dict[str, float],
    ) -> tuple[float, float]:
        """Compute max and total absolute drift between current and target."""
        all_symbols = set(current) | set(target)
        if not all_symbols:
            return 0.0, 0.0

        max_drift = 0.0
        total_drift = 0.0
        for sym in all_symbols:
            drift = abs(current.get(sym, 0.0) - target.get(sym, 0.0))
            max_drift = max(max_drift, drift)
            total_drift += drift

        return max_drift, total_drift


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calendar_days(freq: CalendarFrequency) -> int:
    """Return approximate calendar days for a frequency."""
    return {
        CalendarFrequency.DAILY: 1,
        CalendarFrequency.WEEKLY: 7,
        CalendarFrequency.BIWEEKLY: 14,
        CalendarFrequency.MONTHLY: 30,
        CalendarFrequency.QUARTERLY: 90,
        CalendarFrequency.NONE: 999_999,
    }[freq]
