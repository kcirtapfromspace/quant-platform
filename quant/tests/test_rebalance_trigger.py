"""Tests for rebalance trigger engine (QUA-104)."""
from __future__ import annotations

from datetime import date

from quant.portfolio.rebalance_trigger import (
    CalendarFrequency,
    RebalanceTrigger,
    TriggerConfig,
    TriggerDecision,
    TriggerReason,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TODAY = date(2024, 6, 15)  # Saturday doesn't matter for triggers


def _target() -> dict[str, float]:
    return {"AAPL": 0.30, "GOOG": 0.30, "MSFT": 0.40}


def _current_close() -> dict[str, float]:
    """Weights close to target — small drift."""
    return {"AAPL": 0.29, "GOOG": 0.31, "MSFT": 0.40}


def _current_drifted() -> dict[str, float]:
    """Weights far from target."""
    return {"AAPL": 0.20, "GOOG": 0.40, "MSFT": 0.40}


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_decision(self):
        trigger = RebalanceTrigger()
        result = trigger.evaluate(TODAY, last_rebalance_date=TODAY)
        assert isinstance(result, TriggerDecision)

    def test_triggers_populated(self):
        trigger = RebalanceTrigger()
        result = trigger.evaluate(TODAY, last_rebalance_date=TODAY)
        assert len(result.triggers) >= 3  # Calendar, position drift, total drift

    def test_config_accessible(self):
        cfg = TriggerConfig(max_drift=0.10)
        trigger = RebalanceTrigger(cfg)
        assert trigger.config.max_drift == 0.10

    def test_never_rebalanced_triggers(self):
        """If never rebalanced, calendar should always fire."""
        trigger = RebalanceTrigger()
        result = trigger.evaluate(TODAY)
        assert result.should_rebalance
        calendar_triggers = [
            t for t in result.fired_triggers
            if t.reason == TriggerReason.CALENDAR
        ]
        assert len(calendar_triggers) == 1


# ---------------------------------------------------------------------------
# Calendar trigger
# ---------------------------------------------------------------------------


class TestCalendarTrigger:
    def test_weekly_fires_after_7_days(self):
        cfg = TriggerConfig(calendar_frequency=CalendarFrequency.WEEKLY)
        trigger = RebalanceTrigger(cfg)
        last = date(2024, 6, 7)  # 8 days ago
        result = trigger.evaluate(TODAY, last_rebalance_date=last)
        calendar = [t for t in result.triggers if t.reason == TriggerReason.CALENDAR]
        assert calendar[0].fired

    def test_weekly_holds_after_3_days(self):
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.WEEKLY,
            max_drift=1.0,  # Disable drift trigger
            max_total_drift=10.0,
        )
        trigger = RebalanceTrigger(cfg)
        last = date(2024, 6, 12)  # 3 days ago
        result = trigger.evaluate(TODAY, last_rebalance_date=last)
        calendar = [t for t in result.triggers if t.reason == TriggerReason.CALENDAR]
        assert not calendar[0].fired

    def test_daily_fires_after_1_day(self):
        cfg = TriggerConfig(calendar_frequency=CalendarFrequency.DAILY)
        trigger = RebalanceTrigger(cfg)
        last = date(2024, 6, 14)  # Yesterday
        result = trigger.evaluate(TODAY, last_rebalance_date=last)
        calendar = [t for t in result.triggers if t.reason == TriggerReason.CALENDAR]
        assert calendar[0].fired

    def test_monthly_holds_after_15_days(self):
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.MONTHLY,
            max_drift=1.0,
            max_total_drift=10.0,
        )
        trigger = RebalanceTrigger(cfg)
        last = date(2024, 6, 1)  # 14 days ago
        result = trigger.evaluate(TODAY, last_rebalance_date=last)
        calendar = [t for t in result.triggers if t.reason == TriggerReason.CALENDAR]
        assert not calendar[0].fired

    def test_none_frequency_never_fires(self):
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.NONE,
            max_drift=1.0,
            max_total_drift=10.0,
        )
        trigger = RebalanceTrigger(cfg)
        result = trigger.evaluate(TODAY, last_rebalance_date=date(2020, 1, 1))
        calendar = [t for t in result.triggers if t.reason == TriggerReason.CALENDAR]
        assert not calendar[0].fired

    def test_quarterly_fires(self):
        cfg = TriggerConfig(calendar_frequency=CalendarFrequency.QUARTERLY)
        trigger = RebalanceTrigger(cfg)
        last = date(2024, 3, 1)  # ~106 days ago
        result = trigger.evaluate(TODAY, last_rebalance_date=last)
        calendar = [t for t in result.triggers if t.reason == TriggerReason.CALENDAR]
        assert calendar[0].fired


# ---------------------------------------------------------------------------
# Drift triggers
# ---------------------------------------------------------------------------


class TestDriftTrigger:
    def test_small_drift_no_trigger(self):
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.NONE,
            max_drift=0.05,
            max_total_drift=0.20,
        )
        trigger = RebalanceTrigger(cfg)
        result = trigger.evaluate(
            TODAY, last_rebalance_date=TODAY,
            current_weights=_current_close(), target_weights=_target(),
        )
        assert not result.should_rebalance

    def test_position_drift_fires(self):
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.NONE,
            max_drift=0.05,
        )
        trigger = RebalanceTrigger(cfg)
        result = trigger.evaluate(
            TODAY, last_rebalance_date=TODAY,
            current_weights=_current_drifted(), target_weights=_target(),
        )
        drift_triggers = [
            t for t in result.fired_triggers
            if t.reason == TriggerReason.POSITION_DRIFT
        ]
        assert len(drift_triggers) == 1

    def test_total_drift_fires(self):
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.NONE,
            max_drift=1.0,  # Disable position drift
            max_total_drift=0.10,
        )
        trigger = RebalanceTrigger(cfg)
        result = trigger.evaluate(
            TODAY, last_rebalance_date=TODAY,
            current_weights=_current_drifted(), target_weights=_target(),
        )
        total_triggers = [
            t for t in result.fired_triggers
            if t.reason == TriggerReason.TOTAL_DRIFT
        ]
        assert len(total_triggers) == 1

    def test_max_drift_value(self):
        cfg = TriggerConfig(calendar_frequency=CalendarFrequency.NONE)
        trigger = RebalanceTrigger(cfg)
        result = trigger.evaluate(
            TODAY, last_rebalance_date=TODAY,
            current_weights=_current_drifted(), target_weights=_target(),
        )
        # AAPL drifted 0.10 (0.20 vs 0.30), GOOG drifted 0.10
        assert result.max_position_drift >= 0.10 - 1e-9

    def test_total_drift_value(self):
        trigger = RebalanceTrigger()
        result = trigger.evaluate(
            TODAY, last_rebalance_date=TODAY,
            current_weights=_current_drifted(), target_weights=_target(),
        )
        # AAPL: |0.20-0.30|=0.10, GOOG: |0.40-0.30|=0.10, MSFT: 0
        assert result.total_drift >= 0.20 - 1e-9

    def test_no_weights_no_drift(self):
        trigger = RebalanceTrigger()
        result = trigger.evaluate(TODAY, last_rebalance_date=TODAY)
        assert result.max_position_drift == 0.0
        assert result.total_drift == 0.0


# ---------------------------------------------------------------------------
# Risk trigger
# ---------------------------------------------------------------------------


class TestRiskTrigger:
    def test_risk_trigger_fires(self):
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.NONE,
            max_drift=1.0,
            max_total_drift=10.0,
            max_volatility=0.15,
        )
        trigger = RebalanceTrigger(cfg)
        result = trigger.evaluate(
            TODAY, last_rebalance_date=TODAY,
            portfolio_volatility=0.20,
        )
        risk_triggers = [
            t for t in result.fired_triggers
            if t.reason == TriggerReason.RISK_BREACH
        ]
        assert len(risk_triggers) == 1

    def test_risk_trigger_holds(self):
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.NONE,
            max_drift=1.0,
            max_total_drift=10.0,
            max_volatility=0.25,
        )
        trigger = RebalanceTrigger(cfg)
        result = trigger.evaluate(
            TODAY, last_rebalance_date=TODAY,
            portfolio_volatility=0.20,
        )
        assert not result.should_rebalance

    def test_no_risk_trigger_by_default(self):
        trigger = RebalanceTrigger()
        result = trigger.evaluate(TODAY, last_rebalance_date=TODAY)
        risk_triggers = [
            t for t in result.triggers if t.reason == TriggerReason.RISK_BREACH
        ]
        assert len(risk_triggers) == 0


# ---------------------------------------------------------------------------
# Signal change trigger
# ---------------------------------------------------------------------------


class TestSignalTrigger:
    def test_signal_change_fires(self):
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.NONE,
            max_drift=1.0,
            max_total_drift=10.0,
            signal_change_threshold=0.20,
        )
        trigger = RebalanceTrigger(cfg)
        result = trigger.evaluate(
            TODAY, last_rebalance_date=TODAY,
            signal_rank_correlation=0.70,  # Change = 0.30 > 0.20
        )
        signal_triggers = [
            t for t in result.fired_triggers
            if t.reason == TriggerReason.SIGNAL_CHANGE
        ]
        assert len(signal_triggers) == 1

    def test_signal_stable_holds(self):
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.NONE,
            max_drift=1.0,
            max_total_drift=10.0,
            signal_change_threshold=0.20,
        )
        trigger = RebalanceTrigger(cfg)
        result = trigger.evaluate(
            TODAY, last_rebalance_date=TODAY,
            signal_rank_correlation=0.95,  # Change = 0.05 < 0.20
        )
        assert not result.should_rebalance

    def test_no_signal_trigger_by_default(self):
        trigger = RebalanceTrigger()
        result = trigger.evaluate(TODAY, last_rebalance_date=TODAY)
        signal_triggers = [
            t for t in result.triggers if t.reason == TriggerReason.SIGNAL_CHANGE
        ]
        assert len(signal_triggers) == 0


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_cooldown_prevents_rebalance(self):
        """Even if triggers fire, cooldown should prevent rebalance."""
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.DAILY,
            min_days_between=3,
        )
        trigger = RebalanceTrigger(cfg)
        last = date(2024, 6, 14)  # 1 day ago
        result = trigger.evaluate(TODAY, last_rebalance_date=last)
        # Calendar fires (1 day ≥ daily threshold) but cooldown blocks
        assert len(result.fired_triggers) > 0
        assert not result.should_rebalance

    def test_cooldown_allows_after_minimum(self):
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.DAILY,
            min_days_between=3,
        )
        trigger = RebalanceTrigger(cfg)
        last = date(2024, 6, 12)  # 3 days ago
        result = trigger.evaluate(TODAY, last_rebalance_date=last)
        assert result.should_rebalance


# ---------------------------------------------------------------------------
# Multiple triggers
# ---------------------------------------------------------------------------


class TestMultipleTriggers:
    def test_any_trigger_sufficient(self):
        """A single trigger firing should be enough."""
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.NONE,
            max_drift=0.05,
            max_total_drift=10.0,  # Won't fire
            min_days_between=0,
        )
        trigger = RebalanceTrigger(cfg)
        result = trigger.evaluate(
            TODAY, last_rebalance_date=TODAY,
            current_weights=_current_drifted(), target_weights=_target(),
        )
        assert result.should_rebalance

    def test_multiple_can_fire_simultaneously(self):
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.WEEKLY,
            max_drift=0.05,
        )
        trigger = RebalanceTrigger(cfg)
        last = date(2024, 6, 1)  # 14 days ago
        result = trigger.evaluate(
            TODAY, last_rebalance_date=last,
            current_weights=_current_drifted(), target_weights=_target(),
        )
        assert len(result.fired_triggers) >= 2  # Calendar + drift


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_same_day_rebalance(self):
        """Rebalancing same day should hold if no drift."""
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.NONE,
            max_drift=0.05,
            max_total_drift=0.20,
        )
        trigger = RebalanceTrigger(cfg)
        result = trigger.evaluate(
            TODAY, last_rebalance_date=TODAY,
            current_weights=_target(), target_weights=_target(),
        )
        assert not result.should_rebalance

    def test_new_position_in_target(self):
        """New position in target that's not in current should count as drift."""
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.NONE,
            max_drift=0.05,
            min_days_between=0,
        )
        trigger = RebalanceTrigger(cfg)
        result = trigger.evaluate(
            TODAY, last_rebalance_date=TODAY,
            current_weights={"AAPL": 1.0},
            target_weights={"AAPL": 0.50, "NEW": 0.50},
        )
        assert result.max_position_drift >= 0.50 - 1e-9
        assert result.should_rebalance

    def test_exited_position(self):
        """Position in current but not target should count as drift."""
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.NONE,
            max_drift=0.05,
            min_days_between=0,
        )
        trigger = RebalanceTrigger(cfg)
        result = trigger.evaluate(
            TODAY, last_rebalance_date=TODAY,
            current_weights={"AAPL": 0.50, "OLD": 0.50},
            target_weights={"AAPL": 1.0},
        )
        assert result.max_position_drift >= 0.50 - 1e-9


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        trigger = RebalanceTrigger()
        result = trigger.evaluate(
            TODAY, last_rebalance_date=date(2024, 6, 1),
            current_weights=_current_drifted(), target_weights=_target(),
        )
        summary = result.summary()
        assert "Rebalance Decision" in summary
        assert "REBALANCE" in summary or "HOLD" in summary
        assert "drift" in summary.lower()

    def test_hold_summary(self):
        cfg = TriggerConfig(
            calendar_frequency=CalendarFrequency.NONE,
            max_drift=1.0,
            max_total_drift=10.0,
        )
        trigger = RebalanceTrigger(cfg)
        result = trigger.evaluate(TODAY, last_rebalance_date=TODAY)
        summary = result.summary()
        assert "HOLD" in summary
