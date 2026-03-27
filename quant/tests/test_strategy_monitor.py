"""Tests for strategy performance monitor and drawdown management (QUA-42)."""
from __future__ import annotations

from quant.risk.strategy_monitor import (
    HealthLevel,
    MonitorConfig,
    StrategyMonitor,
    StrategyStatus,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_monitor(**kwargs) -> StrategyMonitor:
    return StrategyMonitor(MonitorConfig(**kwargs))


def _simulate_returns(
    monitor: StrategyMonitor,
    name: str,
    initial_value: float,
    returns: list[float],
) -> float:
    """Apply a sequence of returns to a strategy and return final value."""
    value = initial_value
    for r in returns:
        value *= 1 + r
        monitor.update(name, value)
    return value


# ── Tests: Basic health classification ────────────────────────────────────


class TestHealthClassification:
    def test_healthy_at_start(self):
        m = _make_monitor()
        health = m.update("strat_a", 1_000_000)
        assert health == HealthLevel.HEALTHY

    def test_healthy_with_gains(self):
        m = _make_monitor()
        _simulate_returns(m, "strat_a", 1_000_000, [0.01] * 10)
        assert m.status("strat_a").health == HealthLevel.HEALTHY

    def test_warning_on_mild_drawdown(self):
        m = _make_monitor(warn_drawdown=0.05)
        # Start at 1M, go up to 1.1M, then drop 6%
        _simulate_returns(m, "s", 1_000_000, [0.10])  # peak at 1.1M
        _simulate_returns(m, "s", m.status("s").current_value, [-0.06])
        assert m.status("s").health == HealthLevel.WARNING

    def test_reduced_on_significant_drawdown(self):
        m = _make_monitor(warn_drawdown=0.05, reduce_drawdown=0.10)
        _simulate_returns(m, "s", 1_000_000, [0.10])
        _simulate_returns(m, "s", m.status("s").current_value, [-0.12])
        assert m.status("s").health == HealthLevel.REDUCED

    def test_paused_on_severe_drawdown(self):
        m = _make_monitor(pause_drawdown=0.20)
        _simulate_returns(m, "s", 1_000_000, [0.10])
        _simulate_returns(m, "s", m.status("s").current_value, [-0.25])
        assert m.status("s").health == HealthLevel.PAUSED

    def test_losing_streak_triggers_reduced(self):
        m = _make_monitor(max_losing_streak=5)
        # Small losses that don't breach drawdown thresholds
        _simulate_returns(m, "s", 1_000_000, [-0.005] * 6)
        assert m.status("s").health == HealthLevel.REDUCED


# ── Tests: Capital scaling ────────────────────────────────────────────────


class TestCapitalScaling:
    def test_healthy_returns_full_scale(self):
        m = _make_monitor()
        m.update("s", 1_000_000)
        assert m.capital_scale("s") == 1.0

    def test_warning_returns_warn_scale(self):
        m = _make_monitor(warn_drawdown=0.03, warn_scale=0.70)
        _simulate_returns(m, "s", 1_000_000, [0.10])
        _simulate_returns(m, "s", m.status("s").current_value, [-0.04])
        assert m.capital_scale("s") == 0.70

    def test_reduced_returns_reduce_scale(self):
        m = _make_monitor(reduce_drawdown=0.08, reduce_scale=0.40)
        _simulate_returns(m, "s", 1_000_000, [0.10])
        _simulate_returns(m, "s", m.status("s").current_value, [-0.10])
        assert m.capital_scale("s") == 0.40

    def test_paused_returns_zero(self):
        m = _make_monitor(pause_drawdown=0.15)
        _simulate_returns(m, "s", 1_000_000, [0.10])
        _simulate_returns(m, "s", m.status("s").current_value, [-0.20])
        assert m.capital_scale("s") == 0.0

    def test_unknown_strategy_returns_full_scale(self):
        m = _make_monitor()
        assert m.capital_scale("never_seen") == 1.0


# ── Tests: Recovery ───────────────────────────────────────────────────────


class TestRecovery:
    def test_auto_reinstate_on_recovery(self):
        m = _make_monitor(pause_drawdown=0.15, recovery_threshold=0.95,
                          auto_reinstate=True)
        # Peak at 1.1M, drop to below pause threshold
        _simulate_returns(m, "s", 1_000_000, [0.10])
        peak = m.status("s").peak_value
        _simulate_returns(m, "s", m.status("s").current_value, [-0.20])
        assert m.status("s").health == HealthLevel.PAUSED

        # Recover above recovery_threshold (0.95 * peak)
        recovery_target = peak * 0.96
        m.update("s", recovery_target)
        assert m.status("s").health == HealthLevel.HEALTHY

    def test_no_auto_reinstate_when_disabled(self):
        m = _make_monitor(pause_drawdown=0.15, recovery_threshold=0.95,
                          auto_reinstate=False)
        _simulate_returns(m, "s", 1_000_000, [0.10])
        peak = m.status("s").peak_value
        _simulate_returns(m, "s", m.status("s").current_value, [-0.20])
        assert m.status("s").health == HealthLevel.PAUSED

        # Recover — should stay paused
        m.update("s", peak * 0.96)
        assert m.status("s").health == HealthLevel.PAUSED

    def test_manual_reinstate(self):
        m = _make_monitor(pause_drawdown=0.15, auto_reinstate=False)
        _simulate_returns(m, "s", 1_000_000, [0.10])
        _simulate_returns(m, "s", m.status("s").current_value, [-0.20])
        assert m.status("s").health == HealthLevel.PAUSED

        assert m.reinstate("s") is True
        assert m.status("s").health == HealthLevel.HEALTHY

    def test_reinstate_non_paused_returns_false(self):
        m = _make_monitor()
        m.update("s", 1_000_000)
        assert m.reinstate("s") is False

    def test_reinstate_unknown_returns_false(self):
        m = _make_monitor()
        assert m.reinstate("never_seen") is False


# ── Tests: Drawdown tracking ─────────────────────────────────────────────


class TestDrawdownTracking:
    def test_drawdown_at_peak_is_zero(self):
        m = _make_monitor()
        m.update("s", 1_000_000)
        assert m.status("s").drawdown == 0.0

    def test_drawdown_increases_on_loss(self):
        m = _make_monitor()
        m.update("s", 1_000_000)
        m.update("s", 950_000)
        assert abs(m.status("s").drawdown - 0.05) < 1e-6

    def test_max_drawdown_tracked(self):
        m = _make_monitor()
        m.update("s", 1_000_000)
        m.update("s", 900_000)  # 10% DD
        m.update("s", 950_000)  # 5% DD from 1M peak
        assert abs(m.status("s").max_drawdown - 0.10) < 1e-6

    def test_new_peak_resets_drawdown(self):
        m = _make_monitor()
        m.update("s", 1_000_000)
        m.update("s", 900_000)
        m.update("s", 1_100_000)  # New peak
        assert m.status("s").drawdown == 0.0
        assert m.status("s").peak_value == 1_100_000


# ── Tests: Rolling Sharpe ─────────────────────────────────────────────────


class TestRollingSharpe:
    def test_sharpe_positive_for_gains(self):
        import random
        rng = random.Random(42)
        m = _make_monitor(rolling_window=20)
        # Positive drift with noise so std > 0
        returns = [0.005 + rng.gauss(0, 0.002) for _ in range(25)]
        _simulate_returns(m, "s", 1_000_000, returns)
        assert m.status("s").rolling_sharpe > 0

    def test_sharpe_negative_for_losses(self):
        import random
        rng = random.Random(42)
        m = _make_monitor(rolling_window=20)
        returns = [-0.005 + rng.gauss(0, 0.002) for _ in range(25)]
        _simulate_returns(m, "s", 1_000_000, returns)
        assert m.status("s").rolling_sharpe < 0

    def test_sharpe_floor_triggers_warning(self):
        m = _make_monitor(rolling_window=10, sharpe_floor=-0.50)
        # Strong consistent losses → negative Sharpe
        _simulate_returns(m, "s", 1_000_000, [-0.02] * 15)
        # Should be at least WARNING due to Sharpe floor
        assert m.status("s").health in (
            HealthLevel.WARNING, HealthLevel.REDUCED, HealthLevel.PAUSED
        )

    def test_insufficient_data_returns_zero_sharpe(self):
        m = _make_monitor()
        m.update("s", 1_000_000)
        m.update("s", 1_010_000)
        assert m.status("s").rolling_sharpe == 0.0


# ── Tests: Strategy status ────────────────────────────────────────────────


class TestStrategyStatus:
    def test_status_fields(self):
        m = _make_monitor()
        m.update("s", 1_000_000)
        status = m.status("s")
        assert isinstance(status, StrategyStatus)
        assert status.name == "s"
        assert status.n_updates == 1
        assert status.current_value == 1_000_000
        assert status.last_updated is not None

    def test_all_statuses(self):
        m = _make_monitor()
        m.update("a", 1_000_000)
        m.update("b", 2_000_000)
        statuses = m.all_statuses()
        assert len(statuses) == 2
        names = {s.name for s in statuses}
        assert names == {"a", "b"}

    def test_strategy_names(self):
        m = _make_monitor()
        m.update("z", 100)
        m.update("a", 200)
        assert m.strategy_names == ["a", "z"]

    def test_unknown_strategy_raises(self):
        import pytest

        m = _make_monitor()
        with pytest.raises(KeyError):
            m.status("unknown")


# ── Tests: Reset ──────────────────────────────────────────────────────────


class TestReset:
    def test_reset_single(self):
        import pytest

        m = _make_monitor()
        m.update("s", 1_000_000)
        m.reset("s")
        with pytest.raises(KeyError):
            m.status("s")
        # Capital scale returns 1.0 for unknown
        assert m.capital_scale("s") == 1.0

    def test_reset_all(self):
        m = _make_monitor()
        m.update("a", 100)
        m.update("b", 200)
        m.reset_all()
        assert m.strategy_names == []


# ── Tests: Multi-strategy independence ────────────────────────────────────


class TestMultiStrategy:
    def test_independent_tracking(self):
        m = _make_monitor(pause_drawdown=0.15)
        # Strategy A is healthy
        _simulate_returns(m, "a", 1_000_000, [0.01] * 5)
        # Strategy B is in drawdown
        _simulate_returns(m, "b", 1_000_000, [0.05])
        _simulate_returns(m, "b", m.status("b").current_value, [-0.20])

        assert m.status("a").health == HealthLevel.HEALTHY
        assert m.status("b").health == HealthLevel.PAUSED
        assert m.capital_scale("a") == 1.0
        assert m.capital_scale("b") == 0.0

    def test_pausing_one_doesnt_affect_others(self):
        m = _make_monitor(pause_drawdown=0.15)
        m.update("a", 1_000_000)
        m.update("b", 1_000_000)
        m.update("c", 1_000_000)

        # Crash strategy b
        m.update("b", 800_000)

        assert m.capital_scale("a") == 1.0
        assert m.capital_scale("b") == 0.0
        assert m.capital_scale("c") == 1.0
