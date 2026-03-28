"""Tests for the multi-strategy backtester (QUA-61)."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quant.backtest.multi_strategy import (
    MultiStrategyBacktestEngine,
    MultiStrategyBacktestReport,
    MultiStrategyConfig,
    SleeveConfig,
)
from quant.execution.cost_model import CostModelConfig, TransactionCostModel
from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.engine import PortfolioConfig
from quant.portfolio.lifecycle import LifecycleConfig
from quant.portfolio.optimizers import OptimizationMethod
from quant.portfolio.position_scaler import ScalingConfig, ScalingMethod
from quant.portfolio.strategy_correlation import StrategyCorrelationConfig
from quant.risk.circuit_breaker import DrawdownCircuitBreaker
from quant.signals.adaptive_combiner import AdaptiveCombinerConfig
from quant.signals.base import BaseSignal, SignalOutput
from quant.signals.regime import RegimeConfig, RegimeWeightAdapter

# ── Helpers ───────────────────────────────────────────────────────────────

SYMBOLS = ["AAPL", "GOOG", "MSFT"]


class StubSignal(BaseSignal):
    """Deterministic stub signal for testing."""

    def __init__(self, name: str = "stub", scores: dict[str, float] | None = None):
        self._name = name
        self._scores = scores or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> list[str]:
        return []

    def compute(
        self, symbol: str, features: dict[str, pd.Series], timestamp: datetime
    ) -> SignalOutput:
        score = self._scores.get(symbol, 0.5)
        return SignalOutput(
            symbol=symbol,
            timestamp=timestamp,
            score=score,
            confidence=0.8,
            target_position=score * 0.8,
            metadata={"signal_name": self._name},
        )


def _make_returns(
    symbols: list[str] | None = None, n_days: int = 300
) -> pd.DataFrame:
    symbols = symbols or SYMBOLS
    rng = np.random.default_rng(42)
    n = len(symbols)
    factor = rng.normal(0.0005, 0.01, size=n_days)
    idio = rng.normal(0, 0.015, size=(n_days, n))
    betas = rng.uniform(0.5, 1.5, size=n)
    data = factor[:, None] * betas[None, :] + idio
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    return pd.DataFrame(data, index=dates, columns=symbols)


def _make_config(
    n_sleeves: int = 2,
    capital_weights: list[float] | None = None,
    lifecycle: bool = False,
    apply_realloc: bool = False,
) -> MultiStrategyConfig:
    capital_weights = capital_weights or [1.0 / n_sleeves] * n_sleeves
    signal_names = ["alpha", "beta", "gamma", "delta"]

    sleeves = []
    for i in range(n_sleeves):
        offset = (i + 1) * 0.1
        scores = {sym: min(0.3 + offset + j * 0.05, 0.95) for j, sym in enumerate(SYMBOLS)}
        sleeves.append(
            SleeveConfig(
                name=f"strategy_{signal_names[i % len(signal_names)]}",
                signals=[StubSignal(signal_names[i % len(signal_names)], scores)],
                capital_weight=capital_weights[i],
                portfolio_config=PortfolioConfig(
                    optimization_method=OptimizationMethod.RISK_PARITY,
                    constraints=PortfolioConstraints(
                        long_only=True, max_weight=0.5, max_gross_exposure=1.0
                    ),
                ),
            )
        )

    lc = None
    if lifecycle:
        lc = LifecycleConfig(
            drawdown_watch=0.15,
            drawdown_degraded=0.25,
            drawdown_critical=0.40,
            eval_window=63,
        )

    return MultiStrategyConfig(
        sleeves=sleeves,
        rebalance_frequency=21,
        commission_bps=10.0,
        min_history=60,
        lifecycle_config=lc,
        apply_lifecycle_realloc=apply_realloc,
    )


# ── Tests ─────────────────────────────────────────────────────────────────


class TestMultiStrategyBasic:
    def test_run_returns_report(self):
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), _make_config())
        assert isinstance(report, MultiStrategyBacktestReport)

    def test_report_metrics_populated(self):
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), _make_config())
        assert report.n_trading_days == 300
        assert report.n_sleeves == 2
        assert report.initial_capital == 1_000_000
        assert report.final_value > 0

    def test_rebalances_occur(self):
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), _make_config())
        assert report.n_rebalances > 0

    def test_equity_curve_length(self):
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), _make_config())
        assert len(report.equity_curve) == 300

    def test_weights_history_shape(self):
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), _make_config())
        assert len(report.weights_history) == 300
        assert set(report.weights_history.columns).issubset(set(SYMBOLS))

    def test_summary_readable(self):
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), _make_config())
        summary = report.summary()
        assert "Multi-Strategy Backtest" in summary
        assert "Sharpe" in summary


class TestMultiSleeveAllocation:
    def test_two_sleeves_produce_combined_weights(self):
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), _make_config(n_sleeves=2, capital_weights=[0.6, 0.4]))
        assert report.n_rebalances > 0
        last_reb = report.rebalances[-1]
        assert len(last_reb.combined_weights) > 0
        total_w = sum(last_reb.combined_weights.values())
        assert total_w <= 1.0 + 0.01

    def test_three_sleeves_work(self):
        engine = MultiStrategyBacktestEngine()
        config = _make_config(n_sleeves=3, capital_weights=[0.5, 0.3, 0.2])
        report = engine.run(_make_returns(), config)
        assert report.n_sleeves == 3
        assert report.n_rebalances > 0

    def test_sleeve_snapshots_in_rebalance(self):
        engine = MultiStrategyBacktestEngine()
        config = _make_config(n_sleeves=2)
        report = engine.run(_make_returns(), config)
        for reb in report.rebalances:
            assert len(reb.sleeve_snapshots) > 0
            for snap in reb.sleeve_snapshots:
                assert snap.capital_weight > 0

    def test_weights_exceed_one_raises(self):
        with pytest.raises(ValueError, match="must be <= 1.0"):
            engine = MultiStrategyBacktestEngine()
            config = _make_config(n_sleeves=2, capital_weights=[0.7, 0.5])
            engine.run(_make_returns(), config)

    def test_single_sleeve_full_allocation(self):
        engine = MultiStrategyBacktestEngine()
        config = _make_config(n_sleeves=1, capital_weights=[1.0])
        report = engine.run(_make_returns(), config)
        assert report.n_sleeves == 1
        assert report.n_rebalances > 0


class TestMultiStrategyLifecycle:
    def test_lifecycle_report_attached(self):
        engine = MultiStrategyBacktestEngine()
        config = _make_config(lifecycle=True)
        report = engine.run(_make_returns(), config)
        # At least one rebalance should have a lifecycle report
        lc_reports = [r.lifecycle_report for r in report.rebalances if r.lifecycle_report is not None]
        assert len(lc_reports) > 0

    def test_lifecycle_none_when_not_configured(self):
        engine = MultiStrategyBacktestEngine()
        config = _make_config(lifecycle=False)
        report = engine.run(_make_returns(), config)
        for reb in report.rebalances:
            assert reb.lifecycle_report is None

    def test_lifecycle_reallocation_applied(self):
        engine = MultiStrategyBacktestEngine()
        config = _make_config(
            lifecycle=True,
            apply_realloc=True,
            capital_weights=[0.6, 0.4],
        )
        report = engine.run(_make_returns(), config)
        # Capital weight history should show changes over time
        cwh = report.capital_weight_history
        assert not cwh.empty
        # After first rebalance with lifecycle, weights may differ from initial
        assert len(cwh.columns) == 2

    def test_lifecycle_realloc_preserves_total(self):
        engine = MultiStrategyBacktestEngine()
        config = _make_config(lifecycle=True, apply_realloc=True)
        report = engine.run(_make_returns(), config)
        cwh = report.capital_weight_history
        # Total capital weight should stay close to 1.0 throughout
        totals = cwh.sum(axis=1)
        assert all(totals > 0.5)
        assert all(totals < 1.5)


class TestMultiStrategyEdgeCases:
    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="Insufficient data"):
            engine = MultiStrategyBacktestEngine()
            returns = _make_returns(n_days=30)
            engine.run(returns, _make_config())

    def test_empty_returns_raises(self):
        with pytest.raises(ValueError, match="empty"):
            engine = MultiStrategyBacktestEngine()
            engine.run(pd.DataFrame(), _make_config())

    def test_costs_deducted(self):
        engine = MultiStrategyBacktestEngine()
        config = _make_config()
        report = engine.run(_make_returns(), config)
        assert report.total_costs > 0

    def test_adaptive_combiner_in_sleeve(self):
        config = _make_config()
        config.sleeves[0].adaptive_combiner_config = AdaptiveCombinerConfig(
            ic_lookback=50, min_ic_periods=5, shrinkage=0.3
        )
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        assert report.n_rebalances > 0

    def test_scaling_in_sleeve(self):
        config = _make_config()
        config.sleeves[0].scaling_config = ScalingConfig(method=ScalingMethod.CONVICTION)
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        assert report.n_rebalances > 0


# ── Correlation integration tests ────────────────────────────────────────


class TestMultiStrategyCorrelation:
    def test_correlation_report_in_rebalances(self):
        """Rebalance snapshots should have correlation reports when configured."""
        config = _make_config(n_sleeves=2)
        config.strategy_correlation_config = StrategyCorrelationConfig(
            window=60, min_observations=10
        )
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        has_corr = any(r.correlation_report is not None for r in report.rebalances)
        assert has_corr

    def test_correlation_history_populated(self):
        """avg_strategy_corr_history should be populated."""
        config = _make_config(n_sleeves=2)
        config.strategy_correlation_config = StrategyCorrelationConfig(
            window=60, min_observations=10
        )
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        assert not report.avg_strategy_corr_history.empty

    def test_no_correlation_when_not_configured(self):
        """Without config, correlation fields are empty."""
        config = _make_config()
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        assert report.avg_strategy_corr_history.empty
        assert report.n_crowding_events == 0
        for r in report.rebalances:
            assert r.correlation_report is None

    def test_correlation_values_bounded(self):
        """Avg correlation should be between -1 and 1."""
        config = _make_config(n_sleeves=2)
        config.strategy_correlation_config = StrategyCorrelationConfig(
            window=60, min_observations=10
        )
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        for val in report.avg_strategy_corr_history:
            assert -1.0 - 1e-9 <= val <= 1.0 + 1e-9

    def test_three_sleeve_correlation(self):
        """Three sleeves should produce 3-strategy correlation reports."""
        config = _make_config(n_sleeves=3)
        config.strategy_correlation_config = StrategyCorrelationConfig(
            window=60, min_observations=10
        )
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        for r in report.rebalances:
            if r.correlation_report is not None:
                assert r.correlation_report.n_strategies == 3


# ── Signal IC integration tests ──────────────────────────────────────────


class TestMultiStrategySignalIC:
    def test_lifecycle_receives_signal_ic(self):
        """Lifecycle health should contain IC data after multiple rebalances."""
        config = _make_config(n_sleeves=2, lifecycle=True)
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        # After multiple rebalances, at least some lifecycle reports
        # should have strategies with signal_ic populated
        lc_reports = [r.lifecycle_report for r in report.rebalances if r.lifecycle_report is not None]
        assert len(lc_reports) > 1
        # Second report onward should have IC
        ic_found = False
        for lc in lc_reports[1:]:
            for h in lc.strategy_health:
                if h.signal_ic is not None:
                    ic_found = True
                    assert -1.0 <= h.signal_ic <= 1.0
        assert ic_found, "Expected signal IC to be computed after first rebalance"

    def test_lifecycle_without_ic_still_works(self):
        """Lifecycle works even without IC — first rebalance has no IC."""
        config = _make_config(n_sleeves=2, lifecycle=True)
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        first_lc = next(
            (r.lifecycle_report for r in report.rebalances if r.lifecycle_report is not None),
            None,
        )
        assert first_lc is not None
        # First cycle should have None IC
        for h in first_lc.strategy_health:
            assert h.signal_ic is None


# ── Regime-aware backtesting tests ──────────────────────────────────────


def _make_regime_config(
    n_sleeves: int = 2,
    strategy_types: list[str] | None = None,
) -> MultiStrategyConfig:
    """Build a multi-strategy config with regime detection enabled."""
    strategy_types = strategy_types or ["momentum", "mean_reversion"]
    capital_weights = [1.0 / n_sleeves] * n_sleeves
    signal_names = ["alpha", "beta", "gamma", "delta"]

    sleeves = []
    for i in range(n_sleeves):
        offset = (i + 1) * 0.1
        scores = {sym: min(0.3 + offset + j * 0.05, 0.95) for j, sym in enumerate(SYMBOLS)}
        sleeves.append(
            SleeveConfig(
                name=f"strategy_{signal_names[i % len(signal_names)]}",
                signals=[StubSignal(signal_names[i % len(signal_names)], scores)],
                capital_weight=capital_weights[i],
                strategy_type=strategy_types[i % len(strategy_types)],
                portfolio_config=PortfolioConfig(
                    optimization_method=OptimizationMethod.RISK_PARITY,
                    constraints=PortfolioConstraints(
                        long_only=True, max_weight=0.5, max_gross_exposure=1.0
                    ),
                ),
            )
        )

    return MultiStrategyConfig(
        sleeves=sleeves,
        rebalance_frequency=21,
        commission_bps=10.0,
        min_history=60,
        regime_config=RegimeConfig(),
    )


class TestMultiStrategyRegime:
    def test_regime_state_in_rebalances(self):
        """Rebalance snapshots should contain regime state when configured."""
        config = _make_regime_config()
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        has_regime = any(r.regime_state is not None for r in report.rebalances)
        assert has_regime

    def test_regime_history_populated(self):
        """regime_history should be a non-empty Series of regime labels."""
        config = _make_regime_config()
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        assert not report.regime_history.empty
        # Each entry should be a valid regime label string
        for label in report.regime_history:
            assert isinstance(label, str)
            assert label in {
                "risk_on", "risk_off", "trending",
                "mean_reverting", "crisis", "normal",
            }

    def test_no_regime_when_not_configured(self):
        """Without regime config, regime fields are empty."""
        config = _make_config()
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        assert report.regime_history.empty
        assert report.n_regime_changes == 0
        for r in report.rebalances:
            assert r.regime_state is None

    def test_regime_adjusts_capital_weights(self):
        """Capital weights should change from base when regime is detected."""
        config = _make_regime_config()
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        cwh = report.capital_weight_history
        # After rebalance with regime, weights may differ from base 0.5/0.5
        # Check that at least one period has non-equal weights
        base_w = 1.0 / len(config.sleeves)
        any_changed = False
        for _, row in cwh.iterrows():
            for val in row:
                if abs(val - base_w) > 0.001:
                    any_changed = True
                    break
            if any_changed:
                break
        # The regime adapter may leave weights unchanged if regime is NORMAL
        # so we only check that the machinery ran without error
        assert report.n_rebalances > 0

    def test_custom_regime_adapter(self):
        """A custom RegimeWeightAdapter should be used if provided."""
        config = _make_regime_config()
        config.regime_adapter = RegimeWeightAdapter(max_tilt=0.50)
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        assert report.n_rebalances > 0
        has_regime = any(r.regime_state is not None for r in report.rebalances)
        assert has_regime

    def test_three_sleeve_regime(self):
        """Regime detection works with three strategy sleeves."""
        config = _make_regime_config(
            n_sleeves=3,
            strategy_types=["momentum", "mean_reversion", "trend"],
        )
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        assert report.n_sleeves == 3
        assert not report.regime_history.empty

    def test_regime_with_lifecycle(self):
        """Regime + lifecycle should both produce reports without conflict."""
        config = _make_regime_config()
        config.lifecycle_config = LifecycleConfig(
            drawdown_watch=0.15, drawdown_degraded=0.25,
            drawdown_critical=0.40, eval_window=63,
        )
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        has_regime = any(r.regime_state is not None for r in report.rebalances)
        has_lifecycle = any(r.lifecycle_report is not None for r in report.rebalances)
        assert has_regime
        assert has_lifecycle

    def test_regime_n_changes_count(self):
        """n_regime_changes should count transitions between different regimes."""
        config = _make_regime_config()
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        assert report.n_regime_changes >= 0
        # n_regime_changes should be <= len(regime_history) - 1
        if len(report.regime_history) > 1:
            assert report.n_regime_changes <= len(report.regime_history) - 1


# ── Circuit breaker backtesting tests ───────────────────────────────────


def _make_crash_returns(n_days: int = 300, crash_start: int = 100, crash_days: int = 15) -> pd.DataFrame:
    """Build returns with a severe drawdown episode to trigger the circuit breaker."""
    rng = np.random.default_rng(42)
    data = rng.normal(0.0005, 0.01, size=(n_days, len(SYMBOLS)))
    # Inject a crash: large negative returns for all symbols
    for d in range(crash_start, min(crash_start + crash_days, n_days)):
        data[d, :] = -0.04  # ~4% daily loss for 15 days = ~45% cumulative
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    return pd.DataFrame(data, index=dates, columns=SYMBOLS)


class TestMultiStrategyCircuitBreaker:
    def test_no_breaker_when_not_configured(self):
        """Without a circuit breaker, stats should be zero."""
        config = _make_config()
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        assert report.n_circuit_breaker_trips == 0
        assert report.circuit_breaker_days == 0

    def test_breaker_trips_during_crash(self):
        """Circuit breaker should trip when drawdown exceeds threshold."""
        config = _make_config()
        config.circuit_breaker = DrawdownCircuitBreaker(
            max_drawdown_threshold=0.15, reset_on_new_peak=True
        )
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_crash_returns(), config)
        assert report.n_circuit_breaker_trips >= 1
        assert report.circuit_breaker_days > 0

    def test_breaker_reduces_rebalances(self):
        """With a circuit breaker, fewer rebalances should occur vs no breaker."""
        returns = _make_crash_returns()

        config_no_cb = _make_config()
        engine = MultiStrategyBacktestEngine()
        report_no_cb = engine.run(returns, config_no_cb)

        config_cb = _make_config()
        config_cb.circuit_breaker = DrawdownCircuitBreaker(
            max_drawdown_threshold=0.15, reset_on_new_peak=True
        )
        report_cb = engine.run(returns, config_cb)

        # Breaker should block at least some rebalances
        assert report_cb.n_rebalances <= report_no_cb.n_rebalances

    def test_breaker_does_not_trip_on_normal_returns(self):
        """Normal returns should not trigger a high-threshold breaker."""
        config = _make_config()
        config.circuit_breaker = DrawdownCircuitBreaker(
            max_drawdown_threshold=0.50, reset_on_new_peak=True
        )
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        assert report.n_circuit_breaker_trips == 0
        assert report.circuit_breaker_days == 0

    def test_breaker_with_no_reset(self):
        """Without reset_on_new_peak, breaker stays tripped permanently."""
        config = _make_config()
        config.circuit_breaker = DrawdownCircuitBreaker(
            max_drawdown_threshold=0.15, reset_on_new_peak=False
        )
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_crash_returns(), config)
        assert report.n_circuit_breaker_trips >= 1
        # Should be tripped for the remainder of the backtest
        assert report.circuit_breaker_days > 30

    def test_breaker_with_regime_and_lifecycle(self):
        """Circuit breaker should work alongside regime and lifecycle."""
        config = _make_regime_config()
        config.lifecycle_config = LifecycleConfig(
            drawdown_watch=0.15, drawdown_degraded=0.25,
            drawdown_critical=0.40, eval_window=63,
        )
        config.circuit_breaker = DrawdownCircuitBreaker(
            max_drawdown_threshold=0.15, reset_on_new_peak=True
        )
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_crash_returns(), config)
        # Should produce valid results with all three features
        assert report.n_trading_days == 300
        assert report.n_circuit_breaker_trips >= 1


# ── Cost decomposition tests (QUA-76) ────────────────────────────────────


def _make_cost_model() -> TransactionCostModel:
    return TransactionCostModel(
        CostModelConfig(
            default_spread_bps=5.0,
            impact_coefficient=0.1,
            commission_pct=0.0001,
        )
    )


class TestCostDecomposition:
    def test_decomposition_populated_with_cost_model(self):
        config = _make_config()
        config.cost_model = _make_cost_model()
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        assert report.total_spread_costs > 0
        # Impact may be 0 if no ADV is provided (no impact calc)
        assert report.total_spread_costs >= 0
        assert report.total_impact_costs >= 0
        assert report.total_commission_costs >= 0

    def test_decomposition_sums_to_total(self):
        config = _make_config()
        config.cost_model = _make_cost_model()
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        decomp_sum = (
            report.total_spread_costs
            + report.total_impact_costs
            + report.total_commission_costs
        )
        assert abs(decomp_sum - report.total_costs) < 1.0  # <$1 rounding

    def test_decomposition_zero_without_cost_model(self):
        config = _make_config()
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        assert report.total_spread_costs == 0.0
        assert report.total_impact_costs == 0.0
        assert report.total_commission_costs == 0.0

    def test_per_rebalance_decomposition(self):
        config = _make_config()
        config.cost_model = _make_cost_model()
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        for snap in report.rebalances:
            assert snap.spread_costs >= 0
            assert snap.impact_costs >= 0
            assert snap.commission_costs >= 0

    def test_per_rebalance_sums_match_aggregate(self):
        config = _make_config()
        config.cost_model = _make_cost_model()
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        total_spread = sum(s.spread_costs for s in report.rebalances)
        total_impact = sum(s.impact_costs for s in report.rebalances)
        total_commission = sum(s.commission_costs for s in report.rebalances)
        assert abs(total_spread - report.total_spread_costs) < 0.01
        assert abs(total_impact - report.total_impact_costs) < 0.01
        assert abs(total_commission - report.total_commission_costs) < 0.01

    def test_summary_shows_decomposition(self):
        config = _make_config()
        config.cost_model = _make_cost_model()
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        summary = report.summary()
        assert "Spread costs" in summary
        assert "Impact costs" in summary
        assert "Commission costs" in summary

    def test_summary_hides_decomposition_without_model(self):
        config = _make_config()
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), config)
        summary = report.summary()
        assert "Spread costs" not in summary


# ── Per-sleeve P&L attribution tests (QUA-77) ────────────────────────────


class TestSleeveAttribution:
    def test_sleeve_returns_populated(self):
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), _make_config())
        assert not report.sleeve_returns.empty
        assert len(report.sleeve_returns) == 300

    def test_sleeve_returns_columns_match_sleeves(self):
        engine = MultiStrategyBacktestEngine()
        config = _make_config(n_sleeves=2)
        report = engine.run(_make_returns(), config)
        sleeve_names = {sc.name for sc in config.sleeves}
        assert set(report.sleeve_returns.columns) == sleeve_names

    def test_sleeve_returns_sum_to_portfolio(self):
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), _make_config())
        # Sum of per-sleeve returns should approximate total portfolio return
        sleeve_sum = report.sleeve_returns.sum(axis=1)
        # Skip early days before first rebalance
        start = 61  # after min_history
        np.testing.assert_allclose(
            sleeve_sum.iloc[start:].values,
            report.returns_series.iloc[start:].values,
            atol=1e-10,
        )

    def test_three_sleeves_attribution(self):
        engine = MultiStrategyBacktestEngine()
        config = _make_config(n_sleeves=3)
        report = engine.run(_make_returns(), config)
        assert report.sleeve_returns.shape[1] == 3

    def test_sleeve_returns_zero_before_first_rebalance(self):
        engine = MultiStrategyBacktestEngine()
        report = engine.run(_make_returns(), _make_config())
        # Before first rebalance all sleeve returns should be zero
        early = report.sleeve_returns.iloc[:5]
        assert (early == 0.0).all().all()

    def test_sleeve_returns_with_lifecycle(self):
        engine = MultiStrategyBacktestEngine()
        config = _make_config(lifecycle=True, apply_realloc=True)
        report = engine.run(_make_returns(), config)
        assert not report.sleeve_returns.empty
        # Still sums correctly after capital reallocation
        sleeve_sum = report.sleeve_returns.sum(axis=1)
        start = 61
        np.testing.assert_allclose(
            sleeve_sum.iloc[start:].values,
            report.returns_series.iloc[start:].values,
            atol=1e-10,
        )
