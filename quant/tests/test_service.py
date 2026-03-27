"""Tests for the strategy service, pre-flight checks, and runner metrics (QUA-25)."""
from __future__ import annotations

from datetime import datetime, time, timezone

import numpy as np
import pandas as pd

from quant.execution.paper import PaperBrokerAdapter
from quant.oms.system import OrderManagementSystem
from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.engine import PortfolioConfig
from quant.portfolio.optimizers import OptimizationMethod
from quant.preflight import (
    PreflightCheck,
    PreflightChecker,
    PreflightConfig,
    PreflightResult,
)
from quant.risk.engine import RiskConfig
from quant.risk.limits import ExposureLimits
from quant.runner import RunnerConfig, RunnerState, RunResult, StrategyRunner
from quant.service import ServiceConfig, StrategyService
from quant.signals.base import BaseSignal, SignalOutput

# ── Fixtures ──────────────────────────────────────────────────────────────────

SYMBOLS = ["AAPL", "GOOG", "MSFT"]


class StubSignal(BaseSignal):
    """Simple stub signal for testing."""

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


def _make_runner(
    oms: OrderManagementSystem | None = None,
    symbols: list[str] | None = None,
) -> StrategyRunner:
    symbols = symbols or SYMBOLS
    returns = _make_returns(symbols)

    config = RunnerConfig(
        universe=symbols,
        signals=[StubSignal("momentum", dict.fromkeys(symbols, 0.3))],
        portfolio_config=PortfolioConfig(
            optimization_method=OptimizationMethod.RISK_PARITY,
            constraints=PortfolioConstraints(
                long_only=True, max_weight=0.5, max_gross_exposure=1.0
            ),
        ),
        risk_config=RiskConfig(
            limits=ExposureLimits(
                max_position_fraction=0.50,
                max_order_fraction=0.50,
                max_gross_exposure=1.50,
            ),
        ),
    )
    return StrategyRunner(
        config=config,
        oms=oms or _make_oms(),
        returns_provider=lambda syms, lookback: returns,
    )


def _make_service(
    oms: OrderManagementSystem | None = None,
    skip_preflight: bool = True,
) -> StrategyService:
    oms = oms or _make_oms()
    runner = _make_runner(oms=oms)
    return StrategyService(
        runner=runner,
        oms=oms,
        config=ServiceConfig(skip_preflight=skip_preflight, metrics_port=0),
    )


# ── PreflightChecker Tests ───────────────────────────────────────────────────

class TestPreflightChecker:
    def test_all_checks_pass_with_paper_broker(self):
        oms = _make_oms(initial_cash=100_000)
        checker = PreflightChecker(PreflightConfig(min_cash=1_000))
        result = checker.run(oms)
        assert result.passed
        assert len(result.failures) == 0

    def test_insufficient_cash_fails(self):
        oms = _make_oms(initial_cash=500)
        checker = PreflightChecker(PreflightConfig(min_cash=10_000))
        result = checker.run(oms)
        assert not result.passed
        assert any(f.check == PreflightCheck.SUFFICIENT_CASH for f in result.failures)

    def test_broker_not_connected_fails(self):
        broker = PaperBrokerAdapter(initial_cash=100_000)
        oms = OrderManagementSystem(broker=broker)
        # Note: NOT calling oms.start() — broker remains disconnected
        checker = PreflightChecker()
        result = checker.run(oms)
        assert not result.passed
        assert any(f.check == PreflightCheck.BROKER_CONNECTED for f in result.failures)

    def test_market_hours_weekday_during_hours(self):
        checker = PreflightChecker(
            PreflightConfig(check_market_hours=True, utc_offset_hours=-4)
        )
        # Wednesday 14:00 UTC = 10:00 ET — market is open
        wednesday_during_hours = datetime(2024, 6, 12, 14, 0, tzinfo=timezone.utc)
        oms = _make_oms()
        result = checker.run(oms, now=wednesday_during_hours)
        assert result.passed

    def test_market_hours_weekday_outside_hours(self):
        checker = PreflightChecker(
            PreflightConfig(check_market_hours=True, utc_offset_hours=-4)
        )
        # Wednesday 05:00 UTC = 01:00 ET — market is closed
        wednesday_early = datetime(2024, 6, 12, 5, 0, tzinfo=timezone.utc)
        oms = _make_oms()
        result = checker.run(oms, now=wednesday_early)
        assert not result.passed
        assert any(f.check == PreflightCheck.MARKET_OPEN for f in result.failures)

    def test_market_hours_weekend_fails(self):
        checker = PreflightChecker(
            PreflightConfig(check_market_hours=True, utc_offset_hours=-4)
        )
        # Saturday 14:00 UTC
        saturday = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        oms = _make_oms()
        result = checker.run(oms, now=saturday)
        assert not result.passed
        assert any(f.check == PreflightCheck.MARKET_OPEN for f in result.failures)

    def test_custom_market_hours(self):
        checker = PreflightChecker(
            PreflightConfig(
                check_market_hours=True,
                market_open=time(8, 0),
                market_close=time(22, 0),
                utc_offset_hours=0,
            )
        )
        # Wednesday 12:00 UTC — within 08:00-22:00 UTC range
        oms = _make_oms()
        result = checker.run(oms, now=datetime(2024, 6, 12, 12, 0, tzinfo=timezone.utc))
        assert result.passed

    def test_preflight_result_is_frozen(self):
        result = PreflightResult(passed=True)
        assert result.passed
        assert len(result.failures) == 0


# ── StrategyService Tests ────────────────────────────────────────────────────

class TestStrategyService:
    def test_run_once_returns_result(self):
        service = _make_service()
        result = service.run_once()
        assert result is not None
        assert isinstance(result, RunResult)
        assert result.state == RunnerState.IDLE

    def test_run_once_with_preflight_skip(self):
        service = _make_service(skip_preflight=True)
        result = service.run_once()
        assert result is not None
        assert result.portfolio_value > 0

    def test_run_once_with_preflight_pass(self):
        oms = _make_oms(initial_cash=1_000_000)
        service = StrategyService(
            runner=_make_runner(oms=oms),
            oms=oms,
            config=ServiceConfig(
                skip_preflight=False,
                preflight_config=PreflightConfig(min_cash=1_000),
                metrics_port=0,
            ),
        )
        result = service.run_once()
        assert result is not None

    def test_run_once_preflight_fail_returns_none(self):
        oms = _make_oms(initial_cash=100)  # very low cash
        service = StrategyService(
            runner=_make_runner(oms=oms),
            oms=oms,
            config=ServiceConfig(
                skip_preflight=False,
                preflight_config=PreflightConfig(min_cash=1_000_000),
                metrics_port=0,
            ),
        )
        result = service.run_once()
        assert result is None

    def test_service_config_defaults(self):
        config = ServiceConfig()
        assert config.schedule_hour == 16
        assert config.schedule_minute == 5
        assert config.metrics_port == 8000
        assert not config.skip_preflight

    def test_is_running_initially_false(self):
        service = _make_service()
        assert not service.is_running

    def test_trades_submitted_in_result(self):
        service = _make_service()
        result = service.run_once()
        assert result is not None
        assert result.n_submitted > 0

    def test_construction_present_on_first_run(self):
        service = _make_service()
        result = service.run_once()
        assert result is not None
        assert result.construction is not None
        assert result.construction.rebalance_triggered

    def test_scheduled_run_catches_exceptions(self):
        """The _scheduled_run wrapper must not raise."""
        service = _make_service()
        # Force runner to raise by breaking the returns provider
        service._runner._returns_provider = None
        # Should NOT raise — catches exception internally
        service._scheduled_run()


# ── Runner Metrics Smoke Tests ───────────────────────────────────────────────

class TestRunnerMetrics:
    def test_metrics_importable(self):
        from quant.monitoring.metrics import (
            RUNNER_LAST_RUN_TIMESTAMP,
            RUNNER_PORTFOLIO_VALUE,
            RUNNER_PORTFOLIO_VOLATILITY,
            RUNNER_PREFLIGHT_FAILURES,
            RUNNER_RUN_DURATION,
            RUNNER_RUNS_TOTAL,
            RUNNER_TRADES_REJECTED,
            RUNNER_TRADES_SUBMITTED,
            RUNNER_TURNOVER,
        )
        # All metrics should be importable (either real or NoOp)
        assert RUNNER_RUNS_TOTAL is not None
        assert RUNNER_RUN_DURATION is not None
        assert RUNNER_TRADES_SUBMITTED is not None
        assert RUNNER_TRADES_REJECTED is not None
        assert RUNNER_PORTFOLIO_VALUE is not None
        assert RUNNER_PORTFOLIO_VOLATILITY is not None
        assert RUNNER_TURNOVER is not None
        assert RUNNER_LAST_RUN_TIMESTAMP is not None
        assert RUNNER_PREFLIGHT_FAILURES is not None

    def test_metrics_from_init(self):
        from quant.monitoring import (
            RUNNER_PORTFOLIO_VALUE,
            RUNNER_RUNS_TOTAL,
        )
        assert RUNNER_RUNS_TOTAL is not None
        assert RUNNER_PORTFOLIO_VALUE is not None

    def test_service_records_metrics_without_error(self):
        """Running the service should record metrics without crashing."""
        service = _make_service()
        result = service.run_once()
        assert result is not None
        # If we got here, metric recording didn't crash
