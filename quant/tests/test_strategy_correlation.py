"""Tests for the cross-strategy correlation monitor (QUA-65)."""
from __future__ import annotations

import math
import random

from quant.portfolio.strategy_correlation import (
    StrategyCorrelationConfig,
    StrategyCorrelationMonitor,
)

# ── Helpers ───────────────────────────────────────────────────────────────


def _make_correlated_returns(
    n: int, rho: float, *, seed: int = 42
) -> tuple[list[float], list[float]]:
    """Generate two correlated return series of length *n*.

    Uses a simple Cholesky-like decomposition:
        Y = rho * X + sqrt(1 - rho^2) * Z
    where X and Z are independent standard normals.
    """
    rng = random.Random(seed)
    x = [rng.gauss(0, 0.01) for _ in range(n)]
    z = [rng.gauss(0, 0.01) for _ in range(n)]
    coeff = math.sqrt(max(1.0 - rho**2, 0.0))
    y = [rho * x[i] + coeff * z[i] for i in range(n)]
    return x, y


def _make_independent_returns(
    n: int, *, n_strategies: int = 3, seed: int = 42
) -> dict[str, list[float]]:
    """Generate independent return series for multiple strategies."""
    rng = random.Random(seed)
    names = [f"strat_{i}" for i in range(n_strategies)]
    return {name: [rng.gauss(0, 0.01) for _ in range(n)] for name in names}


# ── Tests: configuration ─────────────────────────────────────────────────


class TestStrategyCorrelationConfig:
    def test_defaults(self):
        cfg = StrategyCorrelationConfig()
        assert cfg.window == 63
        assert cfg.min_observations == 21
        assert cfg.avg_corr_warn == 0.50
        assert cfg.avg_corr_critical == 0.70
        assert cfg.crowding_threshold == 0.80
        assert cfg.min_effective_strategies == 1.5

    def test_custom_config(self):
        cfg = StrategyCorrelationConfig(window=126, avg_corr_warn=0.40)
        assert cfg.window == 126
        assert cfg.avg_corr_warn == 0.40


# ── Tests: basic evaluation ──────────────────────────────────────────────


class TestStrategyCorrelationMonitorBasic:
    def test_single_strategy_returns_normal(self):
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate({"strat_a": [0.01, 0.02, -0.01] * 10})
        assert report.n_strategies == 1
        assert report.level == "normal"
        assert report.avg_pairwise_corr == 0.0

    def test_two_identical_strategies_perfect_correlation(self):
        """Two strategies with identical returns should have corr ~1.0."""
        n = 63
        returns = [0.01 * (i % 5 - 2) for i in range(n)]
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate({"a": returns, "b": list(returns)})
        assert report.n_strategies == 2
        assert report.avg_pairwise_corr > 0.99
        assert report.max_pairwise_corr > 0.99

    def test_two_independent_strategies_low_correlation(self):
        """Independent strategies should have low correlation."""
        rets = _make_independent_returns(100, n_strategies=2, seed=99)
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate(rets)
        assert report.n_strategies == 2
        assert abs(report.avg_pairwise_corr) < 0.30

    def test_correlated_strategies(self):
        """Strategies with known positive correlation."""
        x, y = _make_correlated_returns(100, rho=0.80, seed=123)
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate({"momentum": x, "trend": y})
        assert report.avg_pairwise_corr > 0.50
        assert report.n_observations == 63  # default window caps it

    def test_custom_window_respected(self):
        cfg = StrategyCorrelationConfig(window=30, min_observations=10)
        monitor = StrategyCorrelationMonitor(cfg)
        x, y = _make_correlated_returns(100, rho=0.50)
        report = monitor.evaluate({"a": x, "b": y})
        assert report.n_observations == 30

    def test_insufficient_observations_returns_normal(self):
        cfg = StrategyCorrelationConfig(min_observations=50)
        monitor = StrategyCorrelationMonitor(cfg)
        report = monitor.evaluate({"a": [0.01] * 30, "b": [0.02] * 30})
        assert report.level == "normal"
        assert report.n_observations == 30

    def test_empty_strategies(self):
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate({})
        assert report.n_strategies == 0
        assert report.level == "normal"


# ── Tests: effective strategies ──────────────────────────────────────────


class TestEffectiveStrategies:
    def test_perfectly_correlated_effective_n_is_one(self):
        """Two identical strategies = effectively 1 independent strategy."""
        n = 63
        returns = [0.01 * (i % 7 - 3) for i in range(n)]
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate(
            {"a": returns, "b": list(returns)},
            capital_weights={"a": 0.5, "b": 0.5},
        )
        assert report.effective_strategies < 1.2

    def test_independent_strategies_higher_effective_n(self):
        rets = _make_independent_returns(100, n_strategies=3, seed=77)
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate(
            rets,
            capital_weights=dict.fromkeys(rets, 1.0 / 3),
        )
        # 3 independent strategies → effective N should be close to 3
        assert report.effective_strategies > 2.0

    def test_capital_weights_affect_effective_n(self):
        """Concentrated capital should reduce effective N."""
        rets = _make_independent_returns(100, n_strategies=3, seed=88)
        monitor = StrategyCorrelationMonitor()

        # Equal weight
        eq_report = monitor.evaluate(
            rets, capital_weights=dict.fromkeys(rets, 1.0 / 3)
        )
        # Concentrated weight (90% in one strategy)
        names = sorted(rets.keys())
        conc_weights = {names[0]: 0.90, names[1]: 0.05, names[2]: 0.05}
        conc_report = monitor.evaluate(rets, capital_weights=conc_weights)

        assert eq_report.effective_strategies > conc_report.effective_strategies

    def test_default_equal_weight_when_none(self):
        rets = _make_independent_returns(100, n_strategies=2, seed=55)
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate(rets, capital_weights=None)
        assert report.effective_strategies > 1.0


# ── Tests: risk levels ───────────────────────────────────────────────────


class TestRiskLevels:
    def test_normal_level_for_independent(self):
        rets = _make_independent_returns(100, n_strategies=3)
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate(rets)
        assert report.level == "normal"

    def test_elevated_for_moderate_correlation(self):
        cfg = StrategyCorrelationConfig(avg_corr_warn=0.30, avg_corr_critical=0.70)
        x, y = _make_correlated_returns(100, rho=0.60, seed=10)
        monitor = StrategyCorrelationMonitor(cfg)
        report = monitor.evaluate({"a": x, "b": y})
        assert report.level in ("elevated", "critical")

    def test_critical_for_high_correlation(self):
        n = 63
        returns = [0.01 * (i % 5 - 2) for i in range(n)]
        cfg = StrategyCorrelationConfig(avg_corr_critical=0.70)
        monitor = StrategyCorrelationMonitor(cfg)
        report = monitor.evaluate({"a": returns, "b": list(returns)})
        assert report.level == "critical"

    def test_elevated_when_effective_n_too_low(self):
        """Even moderate avg_corr → elevated if effective_strategies < min."""
        cfg = StrategyCorrelationConfig(
            avg_corr_warn=0.90,  # very high warn so avg corr won't trigger
            min_effective_strategies=5.0,  # but min eff strategies is high
        )
        rets = _make_independent_returns(100, n_strategies=2, seed=66)
        monitor = StrategyCorrelationMonitor(cfg)
        report = monitor.evaluate(rets)
        # 2 strategies can never reach effective N of 5
        assert report.level == "elevated"


# ── Tests: crowding alerts ───────────────────────────────────────────────


class TestCrowdingAlerts:
    def test_crowding_alert_for_identical_strategies(self):
        n = 63
        returns = [0.01 * (i % 5 - 2) for i in range(n)]
        cfg = StrategyCorrelationConfig(crowding_threshold=0.80)
        monitor = StrategyCorrelationMonitor(cfg)
        report = monitor.evaluate({"a": returns, "b": list(returns)})
        assert len(report.crowding_alerts) == 1
        alert = report.crowding_alerts[0]
        assert alert.correlation > 0.80
        assert "a" in alert.strategy_a or "a" in alert.strategy_b

    def test_no_crowding_for_independent(self):
        rets = _make_independent_returns(100, n_strategies=3)
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate(rets)
        assert len(report.crowding_alerts) == 0

    def test_crowding_alert_message_contains_names(self):
        n = 63
        returns = [0.01 * (i % 5 - 2) for i in range(n)]
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate({"momentum": returns, "trend": list(returns)})
        if report.crowding_alerts:
            alert = report.crowding_alerts[0]
            assert "momentum" in alert.message or "trend" in alert.message


# ── Tests: correlation matrix ────────────────────────────────────────────


class TestCorrelationMatrix:
    def test_matrix_is_symmetric(self):
        rets = _make_independent_returns(100, n_strategies=3)
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate(rets)
        for si in report.correlation_matrix:
            for sj in report.correlation_matrix[si]:
                assert abs(
                    report.correlation_matrix[si][sj]
                    - report.correlation_matrix[sj][si]
                ) < 1e-10

    def test_diagonal_is_one(self):
        rets = _make_independent_returns(100, n_strategies=3)
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate(rets)
        for s in report.correlation_matrix:
            assert abs(report.correlation_matrix[s][s] - 1.0) < 1e-10

    def test_max_corr_pair_matches_matrix(self):
        x, y = _make_correlated_returns(100, rho=0.80, seed=200)
        rng = random.Random(200)
        z = [rng.gauss(0, 0.01) for _ in range(100)]
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate({"a": x, "b": y, "c": z})
        sa, sb = report.max_corr_pair
        assert abs(
            report.correlation_matrix[sa][sb] - report.max_pairwise_corr
        ) < 1e-10


# ── Tests: incremental API ───────────────────────────────────────────────


class TestIncrementalAPI:
    def test_update_and_evaluate_incremental(self):
        cfg = StrategyCorrelationConfig(window=30, min_observations=10)
        monitor = StrategyCorrelationMonitor(cfg)
        rng = random.Random(42)

        # Feed 30 observations one at a time
        for _ in range(30):
            monitor.update({"a": rng.gauss(0, 0.01), "b": rng.gauss(0, 0.01)})

        report = monitor.evaluate_incremental()
        assert report.n_strategies == 2
        assert report.n_observations == 30

    def test_buffer_depth(self):
        monitor = StrategyCorrelationMonitor()
        assert monitor.buffer_depth == 0
        monitor.update({"a": 0.01, "b": -0.01})
        assert monitor.buffer_depth == 1
        monitor.update({"a": 0.02, "b": 0.01})
        assert monitor.buffer_depth == 2

    def test_buffer_window_cap(self):
        cfg = StrategyCorrelationConfig(window=5)
        monitor = StrategyCorrelationMonitor(cfg)
        for i in range(20):
            monitor.update({"a": 0.01 * i, "b": -0.01 * i})
        assert monitor.buffer_depth == 5

    def test_reset_clears_buffer(self):
        monitor = StrategyCorrelationMonitor()
        monitor.update({"a": 0.01, "b": 0.02})
        monitor.reset()
        assert monitor.buffer_depth == 0

    def test_incremental_insufficient_returns_normal(self):
        cfg = StrategyCorrelationConfig(min_observations=50)
        monitor = StrategyCorrelationMonitor(cfg)
        for _ in range(10):
            monitor.update({"a": 0.01, "b": 0.02})
        report = monitor.evaluate_incremental()
        assert report.level == "normal"


# ── Tests: report summary ───────────────────────────────────────────────


class TestReportSummary:
    def test_summary_contains_level(self):
        rets = _make_independent_returns(100, n_strategies=3)
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate(rets)
        summary = report.summary()
        assert "NORMAL" in summary

    def test_summary_contains_crowding_alerts(self):
        n = 63
        returns = [0.01 * (i % 5 - 2) for i in range(n)]
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate({"a": returns, "b": list(returns)})
        summary = report.summary()
        if report.crowding_alerts:
            assert "CROWDING" in summary

    def test_summary_contains_strategy_count(self):
        rets = _make_independent_returns(100, n_strategies=2)
        monitor = StrategyCorrelationMonitor()
        report = monitor.evaluate(rets)
        assert "2 strategies" in report.summary()
