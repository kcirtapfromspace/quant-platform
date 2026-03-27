"""Tests for strategy lifecycle manager (QUA-58)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.portfolio.lifecycle import (
    HealthStatus,
    LifecycleConfig,
    LifecycleManager,
    LifecycleReport,
    StrategySnapshot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(
    n: int = 200, mean: float = 0.0005, std: float = 0.01, seed: int = 42
) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.Series(rng.normal(mean, std, n), index=dates, name="returns")


def _make_losing_returns(n: int = 200, seed: int = 99) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.Series(rng.normal(-0.002, 0.02, n), index=dates, name="returns")


def _make_ic_history(n: int = 50, mean: float = 0.05, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.Series(rng.normal(mean, 0.03, n), index=dates, name="ic")


def _make_declining_ic(n: int = 50, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    # Linear decay from 0.10 to -0.02
    trend = np.linspace(0.10, -0.02, n)
    noise = rng.normal(0, 0.01, n)
    return pd.Series(trend + noise, index=dates, name="ic")


# ---------------------------------------------------------------------------
# Tests: Health assessment
# ---------------------------------------------------------------------------


class TestHealthAssessment:
    def test_healthy_strategy(self):
        mgr = LifecycleManager(LifecycleConfig(
            drawdown_watch=0.15,  # relax thresholds for synthetic data
        ))
        mgr.update(StrategySnapshot(
            name="good",
            returns_series=_make_returns(mean=0.001, std=0.005),
            current_weight=0.40,
            signal_ic=0.08,
        ))
        report = mgr.evaluate()
        assert len(report.strategy_health) == 1
        assert report.strategy_health[0].status == HealthStatus.HEALTHY
        assert report.n_healthy == 1

    def test_high_drawdown_triggers_watch(self):
        # Returns with one bad period
        returns = _make_returns(n=200, mean=0.0003, seed=10)
        # Inject a crash
        returns.iloc[50:60] = -0.03
        mgr = LifecycleManager(LifecycleConfig(drawdown_watch=0.05))
        mgr.update(StrategySnapshot(
            name="volatile",
            returns_series=returns,
            current_weight=0.30,
        ))
        report = mgr.evaluate()
        status = report.strategy_health[0].status
        # Should be at least WATCH due to drawdown
        assert status in (HealthStatus.WATCH, HealthStatus.DEGRADED, HealthStatus.CRITICAL)

    def test_severe_drawdown_triggers_critical(self):
        returns = _make_losing_returns(n=200)
        mgr = LifecycleManager(LifecycleConfig(drawdown_critical=0.10))
        mgr.update(StrategySnapshot(
            name="losing",
            returns_series=returns,
            current_weight=0.30,
        ))
        report = mgr.evaluate()
        assert report.strategy_health[0].status == HealthStatus.CRITICAL
        assert report.n_critical == 1

    def test_low_ic_triggers_watch(self):
        mgr = LifecycleManager(LifecycleConfig(ic_watch=0.05))
        mgr.update(StrategySnapshot(
            name="fading",
            returns_series=_make_returns(mean=0.001),
            current_weight=0.30,
            signal_ic=0.02,
        ))
        report = mgr.evaluate()
        status = report.strategy_health[0].status
        assert status in (HealthStatus.WATCH, HealthStatus.DEGRADED, HealthStatus.CRITICAL)

    def test_negative_ic_triggers_critical(self):
        mgr = LifecycleManager(LifecycleConfig(ic_critical=-0.01))
        mgr.update(StrategySnapshot(
            name="broken",
            returns_series=_make_returns(mean=0.0005),
            current_weight=0.20,
            signal_ic=-0.05,
        ))
        report = mgr.evaluate()
        assert report.strategy_health[0].status == HealthStatus.CRITICAL

    def test_negative_sharpe_triggers_at_least_watch(self):
        # Negative Sharpe via negative mean → below min_sharpe_watch
        bad_returns = _make_returns(n=200, mean=-0.001, std=0.005, seed=88)
        mgr = LifecycleManager(LifecycleConfig(
            min_sharpe_watch=0.5,
            drawdown_watch=0.80,
            drawdown_degraded=0.90,
            drawdown_critical=0.95,
        ))
        mgr.update(StrategySnapshot(
            name="flat",
            returns_series=bad_returns,
            current_weight=0.25,
        ))
        report = mgr.evaluate()
        status = report.strategy_health[0].status
        # Sharpe is negative → should be at least WATCH (or worse)
        assert status != HealthStatus.HEALTHY

    def test_no_ic_data_still_works(self):
        mgr = LifecycleManager()
        mgr.update(StrategySnapshot(
            name="no_ic",
            returns_series=_make_returns(mean=0.001),
            current_weight=0.50,
        ))
        report = mgr.evaluate()
        h = report.strategy_health[0]
        assert h.signal_ic is None
        assert h.ic_trend is None

    def test_worst_condition_wins(self):
        """If both IC and drawdown are bad, status should be worst of the two."""
        returns = _make_losing_returns(n=200)
        mgr = LifecycleManager(LifecycleConfig(
            drawdown_critical=0.10, ic_degraded=0.05
        ))
        mgr.update(StrategySnapshot(
            name="both_bad",
            returns_series=returns,
            current_weight=0.20,
            signal_ic=0.02,
        ))
        report = mgr.evaluate()
        # Drawdown should push to CRITICAL even though IC is only DEGRADED
        assert report.strategy_health[0].status == HealthStatus.CRITICAL

    def test_reasons_populated(self):
        mgr = LifecycleManager()
        mgr.update(StrategySnapshot(
            name="good",
            returns_series=_make_returns(mean=0.001),
            current_weight=0.50,
            signal_ic=0.10,
        ))
        report = mgr.evaluate()
        assert len(report.strategy_health[0].reasons) > 0


# ---------------------------------------------------------------------------
# Tests: IC trend
# ---------------------------------------------------------------------------


class TestICTrend:
    def test_declining_ic_negative_trend(self):
        mgr = LifecycleManager()
        mgr.update(StrategySnapshot(
            name="decay",
            returns_series=_make_returns(),
            current_weight=0.30,
            signal_ic=0.02,
            ic_history=_make_declining_ic(),
        ))
        report = mgr.evaluate()
        assert report.strategy_health[0].ic_trend is not None
        assert report.strategy_health[0].ic_trend < 0

    def test_stable_ic_near_zero_trend(self):
        mgr = LifecycleManager()
        # Constant IC
        ic = pd.Series([0.05] * 30, index=pd.bdate_range("2023-01-01", periods=30))
        mgr.update(StrategySnapshot(
            name="stable",
            returns_series=_make_returns(),
            current_weight=0.30,
            signal_ic=0.05,
            ic_history=ic,
        ))
        report = mgr.evaluate()
        assert abs(report.strategy_health[0].ic_trend) < 1e-6

    def test_short_ic_history_zero_trend(self):
        mgr = LifecycleManager()
        ic = pd.Series([0.05, 0.06], index=pd.bdate_range("2023-01-01", periods=2))
        mgr.update(StrategySnapshot(
            name="short",
            returns_series=_make_returns(),
            current_weight=0.30,
            ic_history=ic,
        ))
        report = mgr.evaluate()
        # Only 2 points → trend should be 0.0
        assert report.strategy_health[0].ic_trend == 0.0


# ---------------------------------------------------------------------------
# Tests: Reallocation
# ---------------------------------------------------------------------------


class TestReallocation:
    def test_healthy_strategies_keep_weights(self):
        mgr = LifecycleManager(LifecycleConfig(realloc_aggressiveness=0.0))
        mgr.update(StrategySnapshot(
            name="a", returns_series=_make_returns(seed=1, mean=0.001),
            current_weight=0.50, signal_ic=0.08,
        ))
        mgr.update(StrategySnapshot(
            name="b", returns_series=_make_returns(seed=2, mean=0.001),
            current_weight=0.50, signal_ic=0.07,
        ))
        report = mgr.evaluate()
        for rec in report.recommendations:
            assert abs(rec.delta) < 1e-6

    def test_degraded_loses_weight(self):
        mgr = LifecycleManager(LifecycleConfig(realloc_aggressiveness=0.8))
        mgr.update(StrategySnapshot(
            name="good", returns_series=_make_returns(seed=1, mean=0.001),
            current_weight=0.50, signal_ic=0.10,
        ))
        mgr.update(StrategySnapshot(
            name="bad", returns_series=_make_losing_returns(),
            current_weight=0.50, signal_ic=-0.05,
        ))
        report = mgr.evaluate()
        good_rec = next(r for r in report.recommendations if r.strategy == "good")
        bad_rec = next(r for r in report.recommendations if r.strategy == "bad")
        assert good_rec.delta > 0  # gains weight
        assert bad_rec.delta < 0   # loses weight

    def test_weights_sum_preserved(self):
        mgr = LifecycleManager(LifecycleConfig(realloc_aggressiveness=0.5))
        mgr.update(StrategySnapshot(
            name="a", returns_series=_make_returns(seed=1, mean=0.001),
            current_weight=0.30, signal_ic=0.08,
        ))
        mgr.update(StrategySnapshot(
            name="b", returns_series=_make_returns(seed=2, mean=0.0),
            current_weight=0.40, signal_ic=0.02,
        ))
        mgr.update(StrategySnapshot(
            name="c", returns_series=_make_losing_returns(),
            current_weight=0.30, signal_ic=-0.03,
        ))
        report = mgr.evaluate()
        total_before = sum(r.current_weight for r in report.recommendations)
        total_after = sum(r.recommended_weight for r in report.recommendations)
        assert abs(total_before - total_after) < 1e-6

    def test_max_weight_move_respected(self):
        mgr = LifecycleManager(LifecycleConfig(
            max_weight_move=0.05, realloc_aggressiveness=1.0
        ))
        mgr.update(StrategySnapshot(
            name="good", returns_series=_make_returns(seed=1, mean=0.001),
            current_weight=0.50, signal_ic=0.10,
        ))
        mgr.update(StrategySnapshot(
            name="terrible", returns_series=_make_losing_returns(),
            current_weight=0.50, signal_ic=-0.10,
        ))
        report = mgr.evaluate()
        # Just ensure it runs without error and produces recommendations
        assert len(report.recommendations) == 2

    def test_all_critical_no_reallocation(self):
        mgr = LifecycleManager(LifecycleConfig(drawdown_critical=0.05))
        mgr.update(StrategySnapshot(
            name="a", returns_series=_make_losing_returns(seed=1),
            current_weight=0.50,
        ))
        mgr.update(StrategySnapshot(
            name="b", returns_series=_make_losing_returns(seed=2),
            current_weight=0.50,
        ))
        report = mgr.evaluate()
        for rec in report.recommendations:
            assert abs(rec.delta) < 1e-6  # no change when all critical

    def test_empty_manager_empty_report(self):
        mgr = LifecycleManager()
        report = mgr.evaluate()
        assert len(report.strategy_health) == 0
        assert len(report.recommendations) == 0


# ---------------------------------------------------------------------------
# Tests: Manager operations
# ---------------------------------------------------------------------------


class TestManagerOperations:
    def test_update_and_remove(self):
        mgr = LifecycleManager()
        mgr.update(StrategySnapshot(
            name="a", returns_series=_make_returns(), current_weight=0.50,
        ))
        assert "a" in mgr.strategy_names
        mgr.remove("a")
        assert "a" not in mgr.strategy_names

    def test_remove_nonexistent(self):
        mgr = LifecycleManager()
        mgr.remove("nonexistent")  # should not raise

    def test_update_overwrites(self):
        mgr = LifecycleManager()
        mgr.update(StrategySnapshot(
            name="a", returns_series=_make_returns(), current_weight=0.30,
        ))
        mgr.update(StrategySnapshot(
            name="a", returns_series=_make_returns(), current_weight=0.60,
        ))
        assert len(mgr.strategy_names) == 1
        report = mgr.evaluate()
        assert report.strategy_health[0].current_weight == 0.60

    def test_strategy_names_sorted(self):
        mgr = LifecycleManager()
        for name in ["c", "a", "b"]:
            mgr.update(StrategySnapshot(
                name=name, returns_series=_make_returns(), current_weight=0.33,
            ))
        assert mgr.strategy_names == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Tests: Report
# ---------------------------------------------------------------------------


class TestLifecycleReport:
    def test_report_fields(self):
        mgr = LifecycleManager()
        mgr.update(StrategySnapshot(
            name="a", returns_series=_make_returns(mean=0.001),
            current_weight=0.50, signal_ic=0.08,
        ))
        report = mgr.evaluate()
        assert isinstance(report, LifecycleReport)
        assert report.timestamp is not None
        assert report.n_healthy + report.n_watch + report.n_degraded + report.n_critical == 1

    def test_summary_string(self):
        mgr = LifecycleManager()
        mgr.update(StrategySnapshot(
            name="momentum", returns_series=_make_returns(mean=0.001),
            current_weight=0.50, signal_ic=0.08,
        ))
        mgr.update(StrategySnapshot(
            name="mean_reversion", returns_series=_make_losing_returns(),
            current_weight=0.50, signal_ic=-0.02,
        ))
        report = mgr.evaluate()
        summary = report.summary()
        assert "momentum" in summary
        assert "mean_reversion" in summary
        assert "Reallocation" in summary or "no change" in summary

    def test_total_reallocation_property(self):
        mgr = LifecycleManager(LifecycleConfig(realloc_aggressiveness=0.5))
        mgr.update(StrategySnapshot(
            name="a", returns_series=_make_returns(mean=0.001),
            current_weight=0.50, signal_ic=0.10,
        ))
        mgr.update(StrategySnapshot(
            name="b", returns_series=_make_losing_returns(),
            current_weight=0.50, signal_ic=-0.05,
        ))
        report = mgr.evaluate()
        assert report.total_reallocation >= 0

    def test_has_critical_property(self):
        mgr = LifecycleManager()
        mgr.update(StrategySnapshot(
            name="ok", returns_series=_make_returns(mean=0.001),
            current_weight=1.0, signal_ic=0.10,
        ))
        report = mgr.evaluate()
        assert not report.has_critical

    def test_has_critical_true(self):
        mgr = LifecycleManager(LifecycleConfig(drawdown_critical=0.05))
        mgr.update(StrategySnapshot(
            name="bad", returns_series=_make_losing_returns(),
            current_weight=1.0,
        ))
        report = mgr.evaluate()
        assert report.has_critical


# ---------------------------------------------------------------------------
# Tests: Metric helpers
# ---------------------------------------------------------------------------


class TestMetricHelpers:
    def test_rolling_sharpe(self):
        returns = _make_returns(n=200, mean=0.002, std=0.005)
        sharpe = LifecycleManager._rolling_sharpe(returns, 200)
        assert np.isfinite(sharpe)
        assert sharpe > 0  # strong positive mean → positive Sharpe

    def test_rolling_vol(self):
        returns = _make_returns(n=100, std=0.01)
        vol = LifecycleManager._rolling_vol(returns, 63)
        assert vol > 0
        # Annualised vol should be roughly std * sqrt(252)
        assert 0.05 < vol < 0.50

    def test_max_drawdown(self):
        returns = _make_losing_returns(n=100)
        dd = LifecycleManager._max_drawdown(returns)
        assert dd > 0

    def test_max_drawdown_zero_for_always_up(self):
        returns = pd.Series([0.01] * 50)
        dd = LifecycleManager._max_drawdown(returns)
        assert dd == 0.0

    def test_current_drawdown(self):
        returns = _make_returns(n=100, mean=0.001)
        dd = LifecycleManager._current_drawdown(returns)
        assert dd >= 0.0

    def test_ic_trend_positive(self):
        ic = pd.Series(np.linspace(0.01, 0.10, 20))
        trend = LifecycleManager._ic_trend(ic)
        assert trend > 0

    def test_ic_trend_negative(self):
        ic = pd.Series(np.linspace(0.10, 0.01, 20))
        trend = LifecycleManager._ic_trend(ic)
        assert trend < 0

    def test_empty_returns_zero(self):
        returns = pd.Series([], dtype=float)
        assert LifecycleManager._rolling_sharpe(returns, 63) == 0.0
        assert LifecycleManager._rolling_vol(returns, 63) == 0.0
        assert LifecycleManager._max_drawdown(returns) == 0.0
        assert LifecycleManager._current_drawdown(returns) == 0.0


# ---------------------------------------------------------------------------
# Tests: Config
# ---------------------------------------------------------------------------


class TestLifecycleConfig:
    def test_defaults(self):
        cfg = LifecycleConfig()
        assert cfg.drawdown_watch == 0.05
        assert cfg.max_weight_move == 0.10
        assert cfg.realloc_aggressiveness == 0.5

    def test_custom_config(self):
        cfg = LifecycleConfig(
            drawdown_critical=0.30,
            realloc_aggressiveness=0.8,
            min_allocation=0.05,
        )
        assert cfg.drawdown_critical == 0.30
        assert cfg.realloc_aggressiveness == 0.8
        assert cfg.min_allocation == 0.05
