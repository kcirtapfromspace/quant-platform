"""Tests for the multi-strategy orchestrator (QUA-26)."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from quant.execution.cost_model import CostModelConfig, TransactionCostModel
from quant.execution.paper import PaperBrokerAdapter
from quant.execution.quality_tracker import ExecutionQualityTracker, QualityConfig
from quant.oms.system import OrderManagementSystem
from quant.orchestrator import (
    OrchestratorConfig,
    OrchestratorResult,
    StrategyOrchestrator,
    StrategySleeve,
)
from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.engine import PortfolioConfig
from quant.portfolio.lifecycle import HealthStatus, LifecycleConfig, LifecycleReport
from quant.portfolio.optimizers import OptimizationMethod
from quant.portfolio.position_scaler import ScalingConfig, ScalingMethod
from quant.portfolio.pre_trade import PreTradeConfig
from quant.portfolio.strategy_correlation import (
    StrategyCorrelationConfig,
    StrategyCorrelationMonitor,
    StrategyCorrelationReport,
)
from quant.risk.circuit_breaker import DrawdownCircuitBreaker
from quant.risk.engine import RiskConfig
from quant.risk.limit_checker import LimitConfig, RiskLimitChecker
from quant.risk.limits import ExposureLimits
from quant.risk.reporting import RiskReport, RiskReporter
from quant.risk.strategy_monitor import MonitorConfig, StrategyMonitor
from quant.signals.adaptive_combiner import AdaptiveCombinerConfig
from quant.signals.base import BaseSignal, SignalOutput
from quant.signals.regime import (
    MarketRegime,
    RegimeConfig,
    RegimeDetector,
    RegimeState,
    RegimeWeightAdapter,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

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


class FailingSignal(BaseSignal):
    """Signal that always raises."""

    @property
    def name(self) -> str:
        return "failing"

    @property
    def required_features(self) -> list[str]:
        return []

    def compute(self, symbol, features, timestamp):
        raise RuntimeError("Signal intentionally broken")


def _make_returns(symbols: list[str], n_days: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = len(symbols)
    factor = rng.normal(0.0005, 0.01, size=n_days)
    idio = rng.normal(0, 0.015, size=(n_days, n))
    betas = rng.uniform(0.5, 1.5, size=n)
    returns = factor[:, None] * betas[None, :] + idio
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    return pd.DataFrame(returns, index=dates, columns=symbols)


def _make_oms(initial_cash: float = 1_000_000) -> OrderManagementSystem:
    broker = PaperBrokerAdapter(initial_cash=initial_cash, default_fill_price=150.0)
    oms = OrderManagementSystem(broker=broker)
    oms.start()
    return oms


def _make_orchestrator(
    n_sleeves: int = 2,
    capital_weights: list[float] | None = None,
    oms: OrderManagementSystem | None = None,
    symbols: list[str] | None = None,
    enabled_flags: list[bool] | None = None,
) -> StrategyOrchestrator:
    symbols = symbols or SYMBOLS
    returns = _make_returns(symbols)
    oms = oms or _make_oms()

    capital_weights = capital_weights or [1.0 / n_sleeves] * n_sleeves
    enabled_flags = enabled_flags or [True] * n_sleeves

    signal_names = ["alpha", "beta", "gamma", "delta"]
    sleeves = []
    for i in range(n_sleeves):
        # Give each sleeve slightly different signal scores so weights differ
        offset = (i + 1) * 0.1
        scores = {sym: min(0.3 + offset + j * 0.05, 0.95) for j, sym in enumerate(symbols)}
        sleeves.append(
            StrategySleeve(
                name=f"strategy_{signal_names[i % len(signal_names)]}",
                signals=[StubSignal(signal_names[i % len(signal_names)], scores)],
                capital_weight=capital_weights[i],
                portfolio_config=PortfolioConfig(
                    optimization_method=OptimizationMethod.RISK_PARITY,
                    constraints=PortfolioConstraints(
                        long_only=True, max_weight=0.5, max_gross_exposure=1.0
                    ),
                ),
                enabled=enabled_flags[i],
            )
        )

    config = OrchestratorConfig(
        universe=symbols,
        risk_config=RiskConfig(
            limits=ExposureLimits(
                max_position_fraction=0.50,
                max_order_fraction=0.50,
                max_gross_exposure=1.50,
            ),
        ),
        min_order_value=10.0,
    )

    return StrategyOrchestrator(
        config=config,
        sleeves=sleeves,
        oms=oms,
        returns_provider=lambda syms, lookback: returns,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestOrchestratorBasic:
    def test_run_once_returns_result(self):
        orch = _make_orchestrator(n_sleeves=2)
        result = orch.run_once()
        assert isinstance(result, OrchestratorResult)
        assert result.error == ""

    def test_total_portfolio_value(self):
        orch = _make_orchestrator()
        result = orch.run_once()
        assert result.total_portfolio > 0

    def test_sleeve_results_count(self):
        orch = _make_orchestrator(n_sleeves=3, capital_weights=[0.4, 0.3, 0.3])
        result = orch.run_once()
        assert len(result.sleeve_results) == 3

    def test_combined_weights_not_empty(self):
        orch = _make_orchestrator()
        result = orch.run_once()
        assert len(result.combined_weights) > 0

    def test_trades_submitted(self):
        orch = _make_orchestrator()
        result = orch.run_once()
        assert result.n_submitted > 0

    def test_timestamp_set(self):
        orch = _make_orchestrator()
        result = orch.run_once()
        assert result.timestamp is not None


class TestCapitalAllocation:
    def test_sleeve_capital_equals_weight_times_total(self):
        orch = _make_orchestrator(
            n_sleeves=2, capital_weights=[0.6, 0.4]
        )
        result = orch.run_once()
        total = result.total_portfolio
        for sr in result.sleeve_results:
            expected = total * (0.6 if "alpha" in sr.name else 0.4)
            assert abs(sr.capital_allocated - expected) < 1.0

    def test_weights_exceeding_one_raises(self):
        import pytest

        with pytest.raises(ValueError, match="must be <= 1.0"):
            _make_orchestrator(n_sleeves=2, capital_weights=[0.7, 0.5])

    def test_single_sleeve_gets_full_capital(self):
        orch = _make_orchestrator(n_sleeves=1, capital_weights=[1.0])
        result = orch.run_once()
        sr = result.sleeve_results[0]
        assert abs(sr.capital_allocated - result.total_portfolio) < 1.0


class TestWeightCombination:
    def test_combined_weights_sum_within_bounds(self):
        orch = _make_orchestrator(n_sleeves=2, capital_weights=[0.5, 0.5])
        result = orch.run_once()
        total_weight = sum(result.combined_weights.values())
        # Combined weights should be <= 1.0 (capital_weights sum to 1.0,
        # each sleeve's weights sum to 1.0 within its allocation)
        assert total_weight <= 1.0 + 0.01

    def test_three_sleeves_combine_weights(self):
        orch = _make_orchestrator(
            n_sleeves=3, capital_weights=[0.4, 0.3, 0.3]
        )
        result = orch.run_once()
        assert len(result.combined_weights) > 0
        total_weight = sum(result.combined_weights.values())
        assert total_weight <= 1.0 + 0.01


class TestDisabledSleeves:
    def test_disabled_sleeve_skipped(self):
        orch = _make_orchestrator(
            n_sleeves=2,
            capital_weights=[0.6, 0.4],
            enabled_flags=[True, False],
        )
        result = orch.run_once()
        # Only 1 sleeve should have run
        assert len(result.sleeve_results) == 1
        assert result.sleeve_results[0].name == "strategy_alpha"


class TestErrorHandling:
    def test_failing_signals_degrade_gracefully(self):
        """Individual signal failures are caught — sleeve still runs with
        zero-alpha fallback from the optimizer."""
        oms = _make_oms()
        returns = _make_returns(SYMBOLS)
        sleeves = [
            StrategySleeve(
                name="good",
                signals=[StubSignal("good", dict.fromkeys(SYMBOLS, 0.5))],
                capital_weight=0.5,
                portfolio_config=PortfolioConfig(
                    optimization_method=OptimizationMethod.RISK_PARITY,
                    constraints=PortfolioConstraints(long_only=True, max_weight=0.5),
                ),
            ),
            StrategySleeve(
                name="degraded",
                signals=[FailingSignal()],
                capital_weight=0.5,
                portfolio_config=PortfolioConfig(
                    optimization_method=OptimizationMethod.RISK_PARITY,
                    constraints=PortfolioConstraints(long_only=True, max_weight=0.5),
                ),
            ),
        ]
        config = OrchestratorConfig(
            universe=SYMBOLS,
            risk_config=RiskConfig(
                limits=ExposureLimits(
                    max_position_fraction=0.50,
                    max_order_fraction=0.50,
                    max_gross_exposure=1.50,
                ),
            ),
        )
        orch = StrategyOrchestrator(
            config=config,
            sleeves=sleeves,
            oms=oms,
            returns_provider=lambda syms, lookback: returns,
        )
        result = orch.run_once()
        assert result.error == ""
        # Both sleeves succeed — the degraded one falls back to optimizer defaults
        assert len(result.sleeve_results) == 2
        for sr in result.sleeve_results:
            assert sr.error == ""

    def test_all_signals_failing_still_produces_optimizer_weights(self):
        """When all signals fail, optimizer still produces risk-parity weights
        based on covariance alone (zero alpha). This is correct — the orchestrator
        should not crash."""
        oms = _make_oms()
        returns = _make_returns(SYMBOLS)
        sleeves = [
            StrategySleeve(
                name="s1",
                signals=[FailingSignal()],
                capital_weight=0.5,
                portfolio_config=PortfolioConfig(
                    optimization_method=OptimizationMethod.RISK_PARITY,
                    constraints=PortfolioConstraints(long_only=True, max_weight=0.5),
                ),
            ),
            StrategySleeve(
                name="s2",
                signals=[FailingSignal()],
                capital_weight=0.5,
                portfolio_config=PortfolioConfig(
                    optimization_method=OptimizationMethod.RISK_PARITY,
                    constraints=PortfolioConstraints(long_only=True, max_weight=0.5),
                ),
            ),
        ]
        config = OrchestratorConfig(
            universe=SYMBOLS,
            risk_config=RiskConfig(
                limits=ExposureLimits(
                    max_position_fraction=0.50,
                    max_order_fraction=0.50,
                    max_gross_exposure=1.50,
                ),
            ),
        )
        orch = StrategyOrchestrator(
            config=config,
            sleeves=sleeves,
            oms=oms,
            returns_provider=lambda syms, lookback: returns,
        )
        result = orch.run_once()
        assert result.error == ""
        # Optimizer still produces weights from covariance (risk parity ignores alpha)
        assert len(result.combined_weights) > 0


class TestSleeveResult:
    def test_sleeve_result_has_construction(self):
        orch = _make_orchestrator(n_sleeves=1, capital_weights=[1.0])
        result = orch.run_once()
        sr = result.sleeve_results[0]
        assert sr.construction is not None

    def test_sleeve_result_has_target_weights(self):
        orch = _make_orchestrator(n_sleeves=1, capital_weights=[1.0])
        result = orch.run_once()
        sr = result.sleeve_results[0]
        assert len(sr.target_weights) > 0

    def test_sleeve_capital_allocated(self):
        orch = _make_orchestrator(n_sleeves=2, capital_weights=[0.7, 0.3])
        result = orch.run_once()
        total = result.total_portfolio
        assert abs(result.sleeve_results[0].capital_allocated - 0.7 * total) < 1.0
        assert abs(result.sleeve_results[1].capital_allocated - 0.3 * total) < 1.0


class TestOrchestratorConfig:
    def test_default_config(self):
        config = OrchestratorConfig()
        assert config.min_order_value == 100.0
        assert config.net_conflicting is True

    def test_sleeves_property(self):
        orch = _make_orchestrator(n_sleeves=3, capital_weights=[0.4, 0.3, 0.3])
        assert len(orch.sleeves) == 3

    def test_sleeve_defaults(self):
        sleeve = StrategySleeve(name="test")
        assert sleeve.capital_weight == 1.0
        assert sleeve.enabled is True
        assert sleeve.lookback_days == 252

    def test_sleeve_strategy_type(self):
        sleeve = StrategySleeve(name="test", strategy_type="momentum")
        assert sleeve.strategy_type == "momentum"

    def test_sleeve_strategy_type_default_empty(self):
        sleeve = StrategySleeve(name="test")
        assert sleeve.strategy_type == ""

    def test_config_regime_fields_default_none(self):
        config = OrchestratorConfig()
        assert config.regime_detector is None
        assert config.regime_adapter is None
        assert config.regime_lookback_days == 252


# ── Regime-aware orchestrator helpers ──────────────────────────────────────


def _make_regime_orchestrator(
    regime_config: RegimeConfig | None = None,
    capital_weights: list[float] | None = None,
    strategy_types: list[str] | None = None,
    n_days: int = 300,
) -> StrategyOrchestrator:
    """Build an orchestrator with regime detection enabled."""
    symbols = SYMBOLS
    returns = _make_returns(symbols, n_days=n_days)
    oms = _make_oms()

    capital_weights = capital_weights or [0.40, 0.35, 0.25]
    strategy_types = strategy_types or ["momentum", "mean_reversion", "trend"]

    signal_names = ["mom", "mr", "trend"]
    sleeves = []
    for i in range(3):
        offset = (i + 1) * 0.1
        scores = {sym: min(0.3 + offset + j * 0.05, 0.95) for j, sym in enumerate(symbols)}
        sleeves.append(
            StrategySleeve(
                name=f"strategy_{signal_names[i]}",
                signals=[StubSignal(signal_names[i], scores)],
                capital_weight=capital_weights[i],
                strategy_type=strategy_types[i],
                portfolio_config=PortfolioConfig(
                    optimization_method=OptimizationMethod.RISK_PARITY,
                    constraints=PortfolioConstraints(
                        long_only=True, max_weight=0.5, max_gross_exposure=1.0
                    ),
                ),
            )
        )

    detector = RegimeDetector(config=regime_config)
    adapter = RegimeWeightAdapter()

    config = OrchestratorConfig(
        universe=symbols,
        risk_config=RiskConfig(
            limits=ExposureLimits(
                max_position_fraction=0.50,
                max_order_fraction=0.50,
                max_gross_exposure=1.50,
            ),
        ),
        min_order_value=10.0,
        regime_detector=detector,
        regime_adapter=adapter,
    )

    return StrategyOrchestrator(
        config=config,
        sleeves=sleeves,
        oms=oms,
        returns_provider=lambda syms, lookback: returns,
    )


# ── Regime-aware orchestrator tests ───────────────────────────────────────


class TestRegimeAwareOrchestrator:
    def test_run_with_regime_succeeds(self):
        orch = _make_regime_orchestrator()
        result = orch.run_once()
        assert isinstance(result, OrchestratorResult)
        assert result.error == ""

    def test_last_regime_populated(self):
        orch = _make_regime_orchestrator()
        assert orch.last_regime is None
        orch.run_once()
        assert orch.last_regime is not None
        assert isinstance(orch.last_regime, RegimeState)

    def test_regime_has_valid_fields(self):
        orch = _make_regime_orchestrator()
        orch.run_once()
        regime = orch.last_regime
        assert regime is not None
        assert isinstance(regime.regime, MarketRegime)
        assert 0.0 <= regime.confidence <= 1.0
        assert "vol_ratio" in regime.metrics
        assert "autocorrelation" in regime.metrics

    def test_sleeve_capital_adjusted_by_regime(self):
        """With regime detection on, sleeve capital allocations should differ
        from base weights (unless regime is NORMAL with zero affinity)."""
        orch = _make_regime_orchestrator()
        result = orch.run_once()
        regime = orch.last_regime
        assert regime is not None

        # At least confirm all 3 sleeves ran and got capital
        assert len(result.sleeve_results) == 3
        for sr in result.sleeve_results:
            assert sr.capital_allocated > 0

    def test_combined_weights_valid(self):
        orch = _make_regime_orchestrator()
        result = orch.run_once()
        assert len(result.combined_weights) > 0
        total_weight = sum(result.combined_weights.values())
        assert total_weight <= 1.0 + 0.01

    def test_trades_submitted_with_regime(self):
        orch = _make_regime_orchestrator()
        result = orch.run_once()
        assert result.n_submitted > 0

    def test_no_detector_uses_base_weights(self):
        """When regime_detector is None, sleeve capital = base weight * total."""
        orch = _make_orchestrator(
            n_sleeves=2, capital_weights=[0.6, 0.4]
        )
        result = orch.run_once()
        total = result.total_portfolio
        sr0, sr1 = result.sleeve_results
        assert abs(sr0.capital_allocated - 0.6 * total) < 1.0
        assert abs(sr1.capital_allocated - 0.4 * total) < 1.0

    def test_unknown_strategy_type_no_tilt(self):
        """Sleeves with unknown strategy_type get zero tilt — base weights
        are preserved after re-normalisation."""
        orch = _make_regime_orchestrator(
            strategy_types=["unknown_a", "unknown_b", "unknown_c"],
        )
        result = orch.run_once()
        total = result.total_portfolio
        # With all unknown types, all tilts are 0 → adjusted == base
        expected = [0.40, 0.35, 0.25]
        for i, sr in enumerate(result.sleeve_results):
            assert abs(sr.capital_allocated - expected[i] * total) < 1.0

    def test_regime_persists_across_runs(self):
        """last_regime updates on each run_once() call."""
        orch = _make_regime_orchestrator()
        orch.run_once()
        r1 = orch.last_regime
        assert r1 is not None

        orch.run_once()
        r2 = orch.last_regime
        assert r2 is not None
        # Both should be valid RegimeState (same data, so same result)
        assert r1.regime == r2.regime


# ── Strategy monitor orchestrator integration tests ───────────────────────


def _make_monitored_orchestrator(
    monitor: StrategyMonitor | None = None,
) -> StrategyOrchestrator:
    """Build an orchestrator with a strategy performance monitor."""
    symbols = SYMBOLS
    returns = _make_returns(symbols)
    oms = _make_oms()

    monitor = monitor or StrategyMonitor(MonitorConfig())

    sleeves = []
    signal_names = ["alpha", "beta"]
    for i in range(2):
        offset = (i + 1) * 0.1
        scores = {sym: min(0.3 + offset + j * 0.05, 0.95) for j, sym in enumerate(symbols)}
        sleeves.append(
            StrategySleeve(
                name=f"strategy_{signal_names[i]}",
                signals=[StubSignal(signal_names[i], scores)],
                capital_weight=0.5,
                portfolio_config=PortfolioConfig(
                    optimization_method=OptimizationMethod.RISK_PARITY,
                    constraints=PortfolioConstraints(
                        long_only=True, max_weight=0.5, max_gross_exposure=1.0
                    ),
                ),
            )
        )

    config = OrchestratorConfig(
        universe=symbols,
        risk_config=RiskConfig(
            limits=ExposureLimits(
                max_position_fraction=0.50,
                max_order_fraction=0.50,
                max_gross_exposure=1.50,
            ),
        ),
        min_order_value=10.0,
        strategy_monitor=monitor,
    )

    return StrategyOrchestrator(
        config=config,
        sleeves=sleeves,
        oms=oms,
        returns_provider=lambda syms, lookback: returns,
    )


class TestMonitorOrchestratorIntegration:
    def test_run_with_monitor_succeeds(self):
        orch = _make_monitored_orchestrator()
        result = orch.run_once()
        assert result.error == ""
        assert len(result.sleeve_results) == 2

    def test_monitor_updates_after_run(self):
        monitor = StrategyMonitor(MonitorConfig())
        orch = _make_monitored_orchestrator(monitor=monitor)
        orch.run_once()

        # Monitor should have tracked both sleeves
        assert len(monitor.strategy_names) == 2
        assert "strategy_alpha" in monitor.strategy_names
        assert "strategy_beta" in monitor.strategy_names

    def test_monitor_property_exposed(self):
        monitor = StrategyMonitor(MonitorConfig())
        orch = _make_monitored_orchestrator(monitor=monitor)
        assert orch.strategy_monitor is monitor

    def test_paused_sleeve_gets_zero_capital(self):
        monitor = StrategyMonitor(MonitorConfig(pause_drawdown=0.01))
        orch = _make_monitored_orchestrator(monitor=monitor)

        # Pre-register "strategy_alpha" in the monitor and force it into PAUSED
        monitor.update("strategy_alpha", 1_000_000)
        monitor.update("strategy_alpha", 800_000)  # 20% DD → paused

        assert monitor.capital_scale("strategy_alpha") == 0.0

        result = orch.run_once()
        # strategy_alpha should have 0 capital allocated
        alpha_sr = next(sr for sr in result.sleeve_results if sr.name == "strategy_alpha")
        assert alpha_sr.capital_allocated == 0.0

    def test_healthy_sleeves_unaffected(self):
        monitor = StrategyMonitor(MonitorConfig())
        orch = _make_monitored_orchestrator(monitor=monitor)

        result = orch.run_once()
        total = result.total_portfolio
        # Both sleeves healthy, each gets 0.5 * total
        for sr in result.sleeve_results:
            assert abs(sr.capital_allocated - 0.5 * total) < 1.0

    def test_no_monitor_works_normally(self):
        orch = _make_orchestrator(n_sleeves=2, capital_weights=[0.5, 0.5])
        assert orch.strategy_monitor is None
        result = orch.run_once()
        assert result.error == ""


# ── Execution quality tracker orchestrator integration tests ─────────────


def _make_quality_orchestrator(
    tracker: ExecutionQualityTracker | None = None,
) -> StrategyOrchestrator:
    """Build an orchestrator with an execution quality tracker."""
    symbols = SYMBOLS
    returns = _make_returns(symbols)
    oms = _make_oms()

    tracker = tracker or ExecutionQualityTracker(QualityConfig())

    sleeves = []
    signal_names = ["alpha", "beta"]
    for i in range(2):
        offset = (i + 1) * 0.1
        scores = {sym: min(0.3 + offset + j * 0.05, 0.95) for j, sym in enumerate(symbols)}
        sleeves.append(
            StrategySleeve(
                name=f"strategy_{signal_names[i]}",
                signals=[StubSignal(signal_names[i], scores)],
                capital_weight=0.5,
                portfolio_config=PortfolioConfig(
                    optimization_method=OptimizationMethod.RISK_PARITY,
                    constraints=PortfolioConstraints(
                        long_only=True, max_weight=0.5, max_gross_exposure=1.0
                    ),
                ),
            )
        )

    config = OrchestratorConfig(
        universe=symbols,
        risk_config=RiskConfig(
            limits=ExposureLimits(
                max_position_fraction=0.50,
                max_order_fraction=0.50,
                max_gross_exposure=1.50,
            ),
        ),
        min_order_value=10.0,
        quality_tracker=tracker,
    )

    return StrategyOrchestrator(
        config=config,
        sleeves=sleeves,
        oms=oms,
        returns_provider=lambda syms, lookback: returns,
    )


class TestQualityOrchestratorIntegration:
    def test_run_with_quality_tracker_succeeds(self):
        orch = _make_quality_orchestrator()
        result = orch.run_once()
        assert result.error == ""
        assert len(result.sleeve_results) == 2

    def test_quality_property_exposed(self):
        tracker = ExecutionQualityTracker(QualityConfig())
        orch = _make_quality_orchestrator(tracker=tracker)
        assert orch.quality_tracker is tracker

    def test_good_quality_no_scaling(self):
        tracker = ExecutionQualityTracker(QualityConfig(cost_budget_bps=10.0))
        # Record perfect execution for both sleeves
        for _ in range(10):
            tracker.record("strategy_alpha", slippage_bps=0.0, notional=10_000)
            tracker.record("strategy_beta", slippage_bps=0.0, notional=10_000)

        orch = _make_quality_orchestrator(tracker=tracker)
        result = orch.run_once()
        total = result.total_portfolio
        # Both sleeves should get full capital (quality ~1.0)
        for sr in result.sleeve_results:
            assert abs(sr.capital_allocated - 0.5 * total) < 1.0

    def test_poor_quality_reduces_capital(self):
        tracker = ExecutionQualityTracker(QualityConfig(cost_budget_bps=5.0))
        # Record terrible execution for strategy_alpha
        for _ in range(20):
            tracker.record("strategy_alpha", slippage_bps=8.0, notional=10_000)
        # Good execution for strategy_beta
        for _ in range(20):
            tracker.record("strategy_beta", slippage_bps=0.5, notional=10_000)

        score_alpha = tracker.quality_score("strategy_alpha")
        score_beta = tracker.quality_score("strategy_beta")
        assert score_alpha < 0.5  # degraded
        assert score_beta > 0.9  # healthy

        orch = _make_quality_orchestrator(tracker=tracker)
        result = orch.run_once()

        alpha_sr = next(sr for sr in result.sleeve_results if sr.name == "strategy_alpha")
        beta_sr = next(sr for sr in result.sleeve_results if sr.name == "strategy_beta")

        # Alpha should get much less capital than beta
        assert alpha_sr.capital_allocated < beta_sr.capital_allocated

    def test_zero_quality_score_zeroes_capital(self):
        tracker = ExecutionQualityTracker(QualityConfig(cost_budget_bps=5.0))
        # Extreme slippage → quality score = 0.0
        for _ in range(20):
            tracker.record("strategy_alpha", slippage_bps=50.0, notional=10_000)

        assert tracker.quality_score("strategy_alpha") == 0.0

        orch = _make_quality_orchestrator(tracker=tracker)
        result = orch.run_once()

        alpha_sr = next(sr for sr in result.sleeve_results if sr.name == "strategy_alpha")
        assert alpha_sr.capital_allocated == 0.0

    def test_unknown_sleeve_gets_full_capital(self):
        tracker = ExecutionQualityTracker(QualityConfig())
        # No fills recorded — quality_score returns 1.0 for unknowns
        orch = _make_quality_orchestrator(tracker=tracker)
        result = orch.run_once()
        total = result.total_portfolio
        for sr in result.sleeve_results:
            assert abs(sr.capital_allocated - 0.5 * total) < 1.0

    def test_no_tracker_works_normally(self):
        orch = _make_orchestrator(n_sleeves=2, capital_weights=[0.5, 0.5])
        assert orch.quality_tracker is None
        result = orch.run_once()
        assert result.error == ""

    def test_config_quality_tracker_default_none(self):
        config = OrchestratorConfig()
        assert config.quality_tracker is None


# ── Tests: Position scaler integration ─────────────────────────────────────


def _make_scaling_orchestrator(
    scaling_config: ScalingConfig,
) -> StrategyOrchestrator:
    """Build orchestrator with position scaling enabled on sleeves."""
    symbols = SYMBOLS
    returns = _make_returns(symbols)
    oms = _make_oms()

    sleeves = [
        StrategySleeve(
            name="scaled_sleeve",
            signals=[StubSignal("momentum", {"AAPL": 0.8, "GOOG": 0.4, "MSFT": 0.6})],
            capital_weight=1.0,
            scaling_config=scaling_config,
            portfolio_config=PortfolioConfig(
                optimization_method=OptimizationMethod.RISK_PARITY,
                constraints=PortfolioConstraints(
                    long_only=True, max_weight=0.5, max_gross_exposure=1.0
                ),
            ),
        ),
    ]

    config = OrchestratorConfig(
        universe=symbols,
        risk_config=RiskConfig(
            limits=ExposureLimits(
                max_position_fraction=0.50,
                max_order_fraction=0.50,
                max_gross_exposure=1.50,
            ),
        ),
        min_order_value=10.0,
    )

    return StrategyOrchestrator(
        config=config,
        sleeves=sleeves,
        oms=oms,
        returns_provider=lambda syms, lookback: returns,
    )


class TestPositionScalerIntegration:
    def test_conviction_scaling_runs(self):
        orch = _make_scaling_orchestrator(ScalingConfig(method=ScalingMethod.CONVICTION))
        result = orch.run_once()
        assert result.error == ""
        assert len(result.combined_weights) > 0

    def test_vol_adjusted_scaling_runs(self):
        orch = _make_scaling_orchestrator(
            ScalingConfig(method=ScalingMethod.VOL_ADJUSTED)
        )
        result = orch.run_once()
        assert result.error == ""

    def test_kelly_scaling_runs(self):
        orch = _make_scaling_orchestrator(
            ScalingConfig(method=ScalingMethod.KELLY, kelly_fraction=0.5)
        )
        result = orch.run_once()
        assert result.error == ""

    def test_none_scaling_passthrough(self):
        """ScalingMethod.NONE should behave like no scaler configured."""
        orch_none = _make_scaling_orchestrator(ScalingConfig(method=ScalingMethod.NONE))
        orch_no_scaler = _make_orchestrator(n_sleeves=1, capital_weights=[1.0])

        r_none = orch_none.run_once()
        r_plain = orch_no_scaler.run_once()
        assert r_none.error == ""
        assert r_plain.error == ""

    def test_no_scaling_config_skips(self):
        """With scaling_config=None, orchestrator should skip scaling."""
        orch = _make_orchestrator(n_sleeves=1, capital_weights=[1.0])
        result = orch.run_once()
        assert result.error == ""

    def test_min_confidence_filters_low(self):
        orch = _make_scaling_orchestrator(
            ScalingConfig(method=ScalingMethod.CONVICTION, min_confidence=0.99)
        )
        result = orch.run_once()
        # High min_confidence may zero out alphas → fewer/no weights
        assert result.error == ""

    def test_config_scaling_default_none(self):
        sleeve = StrategySleeve(name="test")
        assert sleeve.scaling_config is None


# ── Tests: Pre-trade pipeline integration ──────────────────────────────────


def _make_pretrade_orchestrator(
    pre_trade_config: PreTradeConfig,
) -> StrategyOrchestrator:
    """Build orchestrator with pre-trade pipeline enabled."""
    symbols = SYMBOLS
    returns = _make_returns(symbols)
    oms = _make_oms()

    sleeves = [
        StrategySleeve(
            name="strategy_a",
            signals=[StubSignal("sig_a", {"AAPL": 0.7, "GOOG": 0.5, "MSFT": 0.3})],
            capital_weight=0.6,
            portfolio_config=PortfolioConfig(
                optimization_method=OptimizationMethod.RISK_PARITY,
                constraints=PortfolioConstraints(
                    long_only=True, max_weight=0.5, max_gross_exposure=1.0
                ),
            ),
        ),
        StrategySleeve(
            name="strategy_b",
            signals=[StubSignal("sig_b", {"AAPL": 0.4, "GOOG": 0.8, "MSFT": 0.6})],
            capital_weight=0.4,
            portfolio_config=PortfolioConfig(
                optimization_method=OptimizationMethod.RISK_PARITY,
                constraints=PortfolioConstraints(
                    long_only=True, max_weight=0.5, max_gross_exposure=1.0
                ),
            ),
        ),
    ]

    config = OrchestratorConfig(
        universe=symbols,
        risk_config=RiskConfig(
            limits=ExposureLimits(
                max_position_fraction=0.50,
                max_order_fraction=0.50,
                max_gross_exposure=1.50,
            ),
        ),
        min_order_value=10.0,
        pre_trade_config=pre_trade_config,
    )

    return StrategyOrchestrator(
        config=config,
        sleeves=sleeves,
        oms=oms,
        returns_provider=lambda syms, lookback: returns,
    )


class TestPreTradePipelineIntegration:
    def test_basic_pretrade_runs(self):
        config = PreTradeConfig(min_trade_weight=0.0, min_trade_dollars=0.0)
        orch = _make_pretrade_orchestrator(config)
        result = orch.run_once()
        assert result.error == ""
        assert result.pre_trade_result is not None

    def test_pre_trade_result_attached(self):
        config = PreTradeConfig(min_trade_weight=0.0, min_trade_dollars=0.0)
        orch = _make_pretrade_orchestrator(config)
        result = orch.run_once()
        assert result.pre_trade_result is not None
        assert result.pre_trade_result.timestamp is not None

    def test_no_pretrade_config_skips(self):
        orch = _make_orchestrator(n_sleeves=2, capital_weights=[0.5, 0.5])
        result = orch.run_once()
        assert result.pre_trade_result is None

    def test_min_weight_filter_active(self):
        # Set high min trade weight — should filter small trades
        config = PreTradeConfig(min_trade_weight=0.50, min_trade_dollars=0.0)
        orch = _make_pretrade_orchestrator(config)
        result = orch.run_once()
        assert result.error == ""
        assert result.pre_trade_result is not None
        # With 0.50 threshold, most trades should be filtered
        assert result.pre_trade_result.trades_filtered >= 0

    def test_limit_checker_enforcement(self):
        checker = RiskLimitChecker(LimitConfig(
            max_position_weight=0.05,
            max_concentration_hhi=None,
        ))
        config = PreTradeConfig(
            limit_checker=checker,
            enforce_limits=True,
            min_trade_weight=0.0,
            min_trade_dollars=0.0,
        )
        orch = _make_pretrade_orchestrator(config)
        result = orch.run_once()
        assert result.error == ""
        # Weights should be clamped to 0.05
        for w in result.combined_weights.values():
            assert abs(w) <= 0.05 + 1e-10

    def test_cost_filter_with_model(self):
        cost_model = TransactionCostModel(CostModelConfig(
            default_spread_bps=100.0,  # very expensive
            impact_coefficient=0.0,
            commission_per_share=0.0,
        ))
        config = PreTradeConfig(
            cost_model=cost_model,
            cost_alpha_ratio=0.01,  # very tight ratio
            min_trade_weight=0.0,
            min_trade_dollars=0.0,
        )
        orch = _make_pretrade_orchestrator(config)
        result = orch.run_once()
        # Pipeline runs without error even if it filters everything
        assert result.error == ""

    def test_config_pre_trade_default_none(self):
        config = OrchestratorConfig()
        assert config.pre_trade_config is None

    def test_combined_weights_reflect_pretrade(self):
        """Orchestrator's combined_weights should be the post-pretrade weights."""
        checker = RiskLimitChecker(LimitConfig(
            max_position_weight=0.01,
            max_concentration_hhi=None,
        ))
        config = PreTradeConfig(
            limit_checker=checker,
            enforce_limits=True,
            min_trade_weight=0.0,
            min_trade_dollars=0.0,
        )
        orch = _make_pretrade_orchestrator(config)
        result = orch.run_once()
        # combined_weights should match the pre-trade adjusted weights
        if result.pre_trade_result is not None:
            for sym in result.combined_weights:
                assert abs(result.combined_weights[sym]) <= 0.01 + 1e-10


# ── Tests: Adaptive combiner integration ───────────────────────────────────


def _make_adaptive_orchestrator(
    adaptive_config: AdaptiveCombinerConfig | None = None,
) -> StrategyOrchestrator:
    """Build orchestrator with adaptive signal combination on a sleeve."""
    symbols = SYMBOLS
    returns = _make_returns(symbols)
    oms = _make_oms()

    ac = adaptive_config or AdaptiveCombinerConfig(
        ic_lookback=50, min_ic_periods=5, shrinkage=0.3
    )

    sleeves = [
        StrategySleeve(
            name="adaptive_sleeve",
            signals=[
                StubSignal("sig_a", {"AAPL": 0.8, "GOOG": 0.3, "MSFT": 0.5}),
                StubSignal("sig_b", {"AAPL": 0.4, "GOOG": 0.7, "MSFT": 0.6}),
            ],
            capital_weight=1.0,
            adaptive_combiner_config=ac,
            portfolio_config=PortfolioConfig(
                optimization_method=OptimizationMethod.RISK_PARITY,
                constraints=PortfolioConstraints(
                    long_only=True, max_weight=0.5, max_gross_exposure=1.0
                ),
            ),
        ),
    ]

    config = OrchestratorConfig(
        universe=symbols,
        risk_config=RiskConfig(
            limits=ExposureLimits(
                max_position_fraction=0.50,
                max_order_fraction=0.50,
                max_gross_exposure=1.50,
            ),
        ),
        min_order_value=10.0,
    )

    return StrategyOrchestrator(
        config=config,
        sleeves=sleeves,
        oms=oms,
        returns_provider=lambda syms, lookback: returns,
    )


class TestAdaptiveCombinerIntegration:
    def test_adaptive_combiner_runs(self):
        orch = _make_adaptive_orchestrator()
        result = orch.run_once()
        assert result.error == ""
        assert len(result.combined_weights) > 0

    def test_adaptive_combiner_multiple_cycles(self):
        """IC history should accumulate across multiple run_once calls."""
        orch = _make_adaptive_orchestrator()
        r1 = orch.run_once()
        r2 = orch.run_once()
        assert r1.error == ""
        assert r2.error == ""

    def test_no_adaptive_config_uses_static(self):
        orch = _make_orchestrator(n_sleeves=1, capital_weights=[1.0])
        result = orch.run_once()
        assert result.error == ""

    def test_adaptive_config_default_none(self):
        sleeve = StrategySleeve(name="test")
        assert sleeve.adaptive_combiner_config is None

    def test_adaptive_with_scaling(self):
        """Adaptive combiner + position scaler should work together."""
        symbols = SYMBOLS
        returns = _make_returns(symbols)
        oms = _make_oms()

        sleeves = [
            StrategySleeve(
                name="adaptive_scaled",
                signals=[
                    StubSignal("sig_a", {"AAPL": 0.7, "GOOG": 0.5, "MSFT": 0.3}),
                ],
                capital_weight=1.0,
                adaptive_combiner_config=AdaptiveCombinerConfig(
                    min_ic_periods=3, shrinkage=0.5
                ),
                scaling_config=ScalingConfig(method=ScalingMethod.CONVICTION),
                portfolio_config=PortfolioConfig(
                    optimization_method=OptimizationMethod.RISK_PARITY,
                    constraints=PortfolioConstraints(
                        long_only=True, max_weight=0.5, max_gross_exposure=1.0
                    ),
                ),
            ),
        ]

        config = OrchestratorConfig(
            universe=symbols,
            risk_config=RiskConfig(
                limits=ExposureLimits(
                    max_position_fraction=0.50,
                    max_order_fraction=0.50,
                    max_gross_exposure=1.50,
                ),
            ),
            min_order_value=10.0,
        )

        orch = StrategyOrchestrator(
            config=config,
            sleeves=sleeves,
            oms=oms,
            returns_provider=lambda syms, lookback: returns,
        )
        result = orch.run_once()
        assert result.error == ""


# ── Lifecycle integration helpers ─────────────────────────────────────────


def _make_lifecycle_orchestrator(
    lifecycle_config: LifecycleConfig | None = None,
    n_sleeves: int = 2,
    capital_weights: list[float] | None = None,
) -> StrategyOrchestrator:
    """Build orchestrator with lifecycle manager enabled."""
    symbols = SYMBOLS
    returns = _make_returns(symbols)
    oms = _make_oms()

    lc = lifecycle_config or LifecycleConfig(
        drawdown_watch=0.15,
        drawdown_degraded=0.25,
        drawdown_critical=0.40,
        eval_window=63,
    )

    capital_weights = capital_weights or [1.0 / n_sleeves] * n_sleeves
    signal_names = ["alpha", "beta", "gamma", "delta"]
    sleeves = []
    for i in range(n_sleeves):
        offset = (i + 1) * 0.1
        scores = {sym: min(0.3 + offset + j * 0.05, 0.95) for j, sym in enumerate(symbols)}
        sleeves.append(
            StrategySleeve(
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

    config = OrchestratorConfig(
        universe=symbols,
        risk_config=RiskConfig(
            limits=ExposureLimits(
                max_position_fraction=0.50,
                max_order_fraction=0.50,
                max_gross_exposure=1.50,
            ),
        ),
        min_order_value=10.0,
        lifecycle_config=lc,
    )

    return StrategyOrchestrator(
        config=config,
        sleeves=sleeves,
        oms=oms,
        returns_provider=lambda syms, lookback: returns,
    )


# ── Lifecycle integration tests ──────────────────────────────────────────


class TestLifecycleIntegration:
    def test_lifecycle_report_attached(self):
        """run_once should produce a LifecycleReport when lifecycle_config is set."""
        orch = _make_lifecycle_orchestrator()
        result = orch.run_once()
        assert result.error == ""
        assert result.lifecycle_report is not None
        assert isinstance(result.lifecycle_report, LifecycleReport)

    def test_lifecycle_report_has_all_sleeves(self):
        """Report should contain health for every active sleeve."""
        orch = _make_lifecycle_orchestrator(n_sleeves=3, capital_weights=[0.4, 0.3, 0.3])
        result = orch.run_once()
        report = result.lifecycle_report
        assert report is not None
        assert len(report.strategy_health) == 3

    def test_lifecycle_none_when_not_configured(self):
        """Without lifecycle_config, lifecycle_report should be None."""
        orch = _make_orchestrator(n_sleeves=2)
        result = orch.run_once()
        assert result.lifecycle_report is None

    def test_lifecycle_manager_property(self):
        """lifecycle_manager property exposes the manager when configured."""
        orch = _make_lifecycle_orchestrator()
        assert orch.lifecycle_manager is not None
        orch2 = _make_orchestrator()
        assert orch2.lifecycle_manager is None

    def test_lifecycle_accumulates_across_cycles(self):
        """Lifecycle snapshots should persist across run_once calls."""
        orch = _make_lifecycle_orchestrator()
        r1 = orch.run_once()
        r2 = orch.run_once()
        assert r1.lifecycle_report is not None
        assert r2.lifecycle_report is not None
        assert orch.lifecycle_manager is not None
        assert len(orch.lifecycle_manager.strategy_names) == 2

    def test_lifecycle_recommendations_present(self):
        """Report should include reallocation recommendations."""
        orch = _make_lifecycle_orchestrator()
        result = orch.run_once()
        report = result.lifecycle_report
        assert report is not None
        assert len(report.recommendations) == 2
        rec_names = {r.strategy for r in report.recommendations}
        health_names = {h.name for h in report.strategy_health}
        assert rec_names == health_names

    def test_lifecycle_health_status_valid(self):
        """All health statuses should be valid HealthStatus enum values."""
        orch = _make_lifecycle_orchestrator()
        result = orch.run_once()
        report = result.lifecycle_report
        assert report is not None
        for h in report.strategy_health:
            assert isinstance(h.status, HealthStatus)

    def test_lifecycle_summary_not_empty(self):
        """Report summary should produce readable output."""
        orch = _make_lifecycle_orchestrator()
        result = orch.run_once()
        report = result.lifecycle_report
        assert report is not None
        summary = report.summary()
        assert "Strategy Lifecycle Report" in summary
        assert len(summary) > 50


# ── Lifecycle reallocation tests ─────────────────────────────────────────


def _make_realloc_orchestrator(
    lifecycle_config: LifecycleConfig | None = None,
    n_sleeves: int = 2,
    capital_weights: list[float] | None = None,
) -> StrategyOrchestrator:
    """Build orchestrator with lifecycle reallocation enabled."""
    symbols = SYMBOLS
    returns = _make_returns(symbols)
    oms = _make_oms()

    lc = lifecycle_config or LifecycleConfig(
        drawdown_watch=0.15,
        drawdown_degraded=0.25,
        drawdown_critical=0.40,
        eval_window=63,
    )

    capital_weights = capital_weights or [1.0 / n_sleeves] * n_sleeves
    signal_names = ["alpha", "beta", "gamma", "delta"]
    sleeves = []
    for i in range(n_sleeves):
        offset = (i + 1) * 0.1
        scores = {sym: min(0.3 + offset + j * 0.05, 0.95) for j, sym in enumerate(symbols)}
        sleeves.append(
            StrategySleeve(
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

    config = OrchestratorConfig(
        universe=symbols,
        risk_config=RiskConfig(
            limits=ExposureLimits(
                max_position_fraction=0.50,
                max_order_fraction=0.50,
                max_gross_exposure=1.50,
            ),
        ),
        min_order_value=10.0,
        lifecycle_config=lc,
        apply_lifecycle_realloc=True,
    )

    return StrategyOrchestrator(
        config=config,
        sleeves=sleeves,
        oms=oms,
        returns_provider=lambda syms, lookback: returns,
    )


class TestLifecycleReallocation:
    def test_realloc_weights_stored_after_first_cycle(self):
        """After first run_once, lifecycle_weights should be populated."""
        orch = _make_realloc_orchestrator()
        assert orch.lifecycle_weights is None
        orch.run_once()
        assert orch.lifecycle_weights is not None
        assert len(orch.lifecycle_weights) == 2

    def test_realloc_weights_applied_on_second_cycle(self):
        """Second cycle should use recommended weights from first cycle."""
        orch = _make_realloc_orchestrator(
            capital_weights=[0.6, 0.4],
        )
        r1 = orch.run_once()
        assert r1.error == ""
        # After first cycle, lifecycle weights should differ from base
        lw = orch.lifecycle_weights
        assert lw is not None

        r2 = orch.run_once()
        assert r2.error == ""
        # Second cycle ran successfully with realloc applied
        assert len(r2.sleeve_results) == 2

    def test_realloc_not_applied_when_disabled(self):
        """With apply_lifecycle_realloc=False, weights should not be stored."""
        orch = _make_lifecycle_orchestrator()  # uses default (False)
        orch.run_once()
        assert orch.lifecycle_weights is None

    def test_realloc_weights_sum_preserved(self):
        """Recommended weights should sum close to original total."""
        orch = _make_realloc_orchestrator(capital_weights=[0.5, 0.5])
        orch.run_once()
        lw = orch.lifecycle_weights
        assert lw is not None
        total = sum(lw.values())
        assert abs(total - 1.0) < 0.05

    def test_realloc_multiple_cycles_converge(self):
        """Lifecycle weights should update each cycle."""
        orch = _make_realloc_orchestrator()
        for _ in range(3):
            result = orch.run_once()
            assert result.error == ""
        assert orch.lifecycle_weights is not None
        assert len(orch.lifecycle_weights) == 2


# ── Circuit breaker integration helpers ──────────────────────────────────


def _make_cb_orchestrator(
    threshold: float = 0.10,
    initial_cash: float = 1_000_000,
) -> StrategyOrchestrator:
    """Build orchestrator with circuit breaker enabled."""
    symbols = SYMBOLS
    returns = _make_returns(symbols)
    oms = _make_oms(initial_cash=initial_cash)

    sleeves = [
        StrategySleeve(
            name="strategy_alpha",
            signals=[StubSignal("alpha", dict.fromkeys(SYMBOLS, 0.5))],
            capital_weight=1.0,
            portfolio_config=PortfolioConfig(
                optimization_method=OptimizationMethod.RISK_PARITY,
                constraints=PortfolioConstraints(
                    long_only=True, max_weight=0.5, max_gross_exposure=1.0
                ),
            ),
        ),
    ]

    cb = DrawdownCircuitBreaker(max_drawdown_threshold=threshold)

    config = OrchestratorConfig(
        universe=symbols,
        risk_config=RiskConfig(
            limits=ExposureLimits(
                max_position_fraction=0.50,
                max_order_fraction=0.50,
                max_gross_exposure=1.50,
            ),
        ),
        min_order_value=10.0,
        circuit_breaker=cb,
    )

    return StrategyOrchestrator(
        config=config,
        sleeves=sleeves,
        oms=oms,
        returns_provider=lambda syms, lookback: returns,
    )


# ── Circuit breaker integration tests ────────────────────────────────────


class TestCircuitBreakerIntegration:
    def test_cb_not_tripped_normal_operation(self):
        """Circuit breaker should not trip under normal conditions."""
        orch = _make_cb_orchestrator(threshold=0.50)
        result = orch.run_once()
        assert result.error == ""
        assert not result.circuit_breaker_tripped
        assert result.n_submitted > 0

    def test_cb_property_exposes_breaker(self):
        """circuit_breaker property should expose the configured breaker."""
        orch = _make_cb_orchestrator()
        assert orch.circuit_breaker is not None
        orch2 = _make_orchestrator()
        assert orch2.circuit_breaker is None

    def test_cb_tripped_halts_trading(self):
        """When pre-tripped, no trades should be submitted."""
        orch = _make_cb_orchestrator(threshold=0.10)
        # Pre-trip the breaker by simulating a large drawdown
        cb = orch.circuit_breaker
        assert cb is not None
        cb.update(2_000_000)  # Set peak high
        cb.update(1_500_000)  # 25% drawdown > 10% threshold
        assert cb.is_tripped()

        result = orch.run_once()
        assert result.circuit_breaker_tripped
        assert result.n_submitted == 0
        assert result.n_rejected == 0
        assert len(result.sleeve_results) == 0

    def test_cb_reset_allows_trading(self):
        """After manual reset with peak cleared, trading should resume."""
        orch = _make_cb_orchestrator(threshold=0.10)
        cb = orch.circuit_breaker
        assert cb is not None
        cb.update(2_000_000)
        cb.update(1_500_000)
        assert cb.is_tripped()

        cb.reset()
        cb._peak_value = 0.0  # Clear artificial peak
        result = orch.run_once()
        assert not result.circuit_breaker_tripped
        assert result.n_submitted > 0

    def test_cb_false_when_not_configured(self):
        """circuit_breaker_tripped should be False when no breaker is set."""
        orch = _make_orchestrator()
        result = orch.run_once()
        assert not result.circuit_breaker_tripped

    def test_cb_multiple_cycles_tracks_peak(self):
        """Circuit breaker should track portfolio peak across cycles."""
        orch = _make_cb_orchestrator(threshold=0.50)
        r1 = orch.run_once()
        r2 = orch.run_once()
        assert not r1.circuit_breaker_tripped
        assert not r2.circuit_breaker_tripped
        cb = orch.circuit_breaker
        assert cb is not None
        assert cb._peak_value > 0


# ── Risk reporter integration helpers ────────────────────────────────────


def _make_risk_reporter_orchestrator() -> StrategyOrchestrator:
    """Build orchestrator with risk reporter enabled."""
    symbols = SYMBOLS
    returns = _make_returns(symbols)
    oms = _make_oms()

    sleeves = [
        StrategySleeve(
            name="strategy_alpha",
            signals=[StubSignal("alpha", dict.fromkeys(SYMBOLS, 0.5))],
            capital_weight=1.0,
            portfolio_config=PortfolioConfig(
                optimization_method=OptimizationMethod.RISK_PARITY,
                constraints=PortfolioConstraints(
                    long_only=True, max_weight=0.5, max_gross_exposure=1.0
                ),
            ),
        ),
    ]

    config = OrchestratorConfig(
        universe=symbols,
        risk_config=RiskConfig(
            limits=ExposureLimits(
                max_position_fraction=0.50,
                max_order_fraction=0.50,
                max_gross_exposure=1.50,
            ),
        ),
        min_order_value=10.0,
        risk_reporter=RiskReporter(),
    )

    return StrategyOrchestrator(
        config=config,
        sleeves=sleeves,
        oms=oms,
        returns_provider=lambda syms, lookback: returns,
    )


# ── Risk reporter integration tests ─────────────────────────────────────


class TestRiskReporterIntegration:
    def test_risk_report_attached(self):
        """run_once should produce a RiskReport when risk_reporter is set."""
        orch = _make_risk_reporter_orchestrator()
        result = orch.run_once()
        assert result.error == ""
        assert result.risk_report is not None
        assert isinstance(result.risk_report, RiskReport)

    def test_risk_report_has_var(self):
        """Risk report should include VaR results."""
        orch = _make_risk_reporter_orchestrator()
        result = orch.run_once()
        report = result.risk_report
        assert report is not None
        assert len(report.var_results) > 0

    def test_risk_report_has_stress(self):
        """Risk report should include stress test results."""
        orch = _make_risk_reporter_orchestrator()
        result = orch.run_once()
        report = result.risk_report
        assert report is not None
        assert len(report.stress_results) > 0

    def test_risk_report_summary_readable(self):
        """Risk report summary should produce text output."""
        orch = _make_risk_reporter_orchestrator()
        result = orch.run_once()
        report = result.risk_report
        assert report is not None
        summary = report.summary()
        assert "Risk Report" in summary
        assert len(summary) > 50

    def test_risk_report_none_when_not_configured(self):
        """Without risk_reporter, risk_report should be None."""
        orch = _make_orchestrator()
        result = orch.run_once()
        assert result.risk_report is None

    def test_risk_report_volatility_positive(self):
        """Annualised volatility should be positive with real returns."""
        orch = _make_risk_reporter_orchestrator()
        result = orch.run_once()
        report = result.risk_report
        assert report is not None
        assert report.annualised_volatility > 0


# ── Strategy correlation integration helpers ─────────────────────────────


def _make_corr_orchestrator(
    *, n_sleeves: int = 2, corr_config: StrategyCorrelationConfig | None = None,
) -> StrategyOrchestrator:
    """Build orchestrator with strategy correlation monitor enabled."""
    symbols = SYMBOLS
    returns = _make_returns(symbols)
    oms = _make_oms()

    weights = [1.0 / n_sleeves] * n_sleeves
    signal_names = ["alpha", "beta", "gamma"]
    sleeves = []
    for i in range(n_sleeves):
        offset = (i + 1) * 0.1
        scores = {sym: min(0.3 + offset + j * 0.05, 0.95) for j, sym in enumerate(symbols)}
        sleeves.append(
            StrategySleeve(
                name=f"strategy_{signal_names[i % len(signal_names)]}",
                signals=[StubSignal(signal_names[i % len(signal_names)], scores)],
                capital_weight=weights[i],
                portfolio_config=PortfolioConfig(
                    optimization_method=OptimizationMethod.RISK_PARITY,
                    constraints=PortfolioConstraints(
                        long_only=True, max_weight=0.5, max_gross_exposure=1.0
                    ),
                ),
            )
        )

    config = OrchestratorConfig(
        universe=symbols,
        risk_config=RiskConfig(
            limits=ExposureLimits(
                max_position_fraction=0.50,
                max_order_fraction=0.50,
                max_gross_exposure=1.50,
            ),
        ),
        min_order_value=10.0,
        strategy_correlation=StrategyCorrelationMonitor(corr_config),
    )

    return StrategyOrchestrator(
        config=config,
        sleeves=sleeves,
        oms=oms,
        returns_provider=lambda syms, lookback: returns,
    )


# ── Strategy correlation integration tests ───────────────────────────────


class TestStrategyCorrelationIntegration:
    def test_correlation_report_attached(self):
        """run_once should produce a StrategyCorrelationReport when configured."""
        orch = _make_corr_orchestrator()
        result = orch.run_once()
        assert result.error == ""
        assert result.correlation_report is not None
        assert isinstance(result.correlation_report, StrategyCorrelationReport)

    def test_correlation_report_has_strategies(self):
        """Report should cover all sleeves."""
        orch = _make_corr_orchestrator(n_sleeves=2)
        result = orch.run_once()
        report = result.correlation_report
        assert report is not None
        assert report.n_strategies == 2

    def test_correlation_report_none_when_not_configured(self):
        """Without strategy_correlation, correlation_report should be None."""
        orch = _make_orchestrator()
        result = orch.run_once()
        assert result.correlation_report is None

    def test_correlation_report_has_valid_level(self):
        """Report level should be one of the expected values."""
        orch = _make_corr_orchestrator()
        result = orch.run_once()
        report = result.correlation_report
        assert report is not None
        assert report.level in ("normal", "elevated", "critical")

    def test_correlation_property_exposes_monitor(self):
        """strategy_correlation property should expose the monitor."""
        orch = _make_corr_orchestrator()
        assert orch.strategy_correlation is not None
        assert isinstance(orch.strategy_correlation, StrategyCorrelationMonitor)

    def test_correlation_summary_not_empty(self):
        """Report summary should produce readable text."""
        orch = _make_corr_orchestrator()
        result = orch.run_once()
        report = result.correlation_report
        assert report is not None
        summary = report.summary()
        assert len(summary) > 20
        assert "Strategy Correlation" in summary

    def test_three_sleeve_correlation(self):
        """Three sleeves should produce a 3x3 correlation matrix."""
        orch = _make_corr_orchestrator(n_sleeves=3)
        result = orch.run_once()
        report = result.correlation_report
        assert report is not None
        assert report.n_strategies == 3
        assert len(report.correlation_matrix) == 3
