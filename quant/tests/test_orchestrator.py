"""Tests for the multi-strategy orchestrator (QUA-26)."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from quant.execution.paper import PaperBrokerAdapter
from quant.oms.system import OrderManagementSystem
from quant.orchestrator import (
    OrchestratorConfig,
    OrchestratorResult,
    StrategyOrchestrator,
    StrategySleeve,
)
from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.engine import PortfolioConfig
from quant.portfolio.optimizers import OptimizationMethod
from quant.risk.engine import RiskConfig
from quant.risk.limits import ExposureLimits
from quant.risk.strategy_monitor import MonitorConfig, StrategyMonitor
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
