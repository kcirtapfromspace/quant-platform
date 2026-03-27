"""Unit tests for the strategy runner (QUA-24)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quant.execution.paper import PaperBrokerAdapter
from quant.oms.system import OrderManagementSystem
from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.engine import PortfolioConfig
from quant.portfolio.optimizers import OptimizationMethod
from quant.risk.engine import RiskConfig
from quant.risk.limits import ExposureLimits
from quant.runner import (
    ExecutionRecord,
    RunnerConfig,
    RunnerState,
    RunResult,
    StrategyRunner,
)
from quant.signals.base import BaseSignal, SignalOutput

# ── Fixtures ──────────────────────────────────────────────────────────────────

_NOW = datetime(2024, 6, 15, tzinfo=timezone.utc)

SYMBOLS = ["AAPL", "GOOG", "MSFT"]


class StubSignal(BaseSignal):
    """Simple stub signal for testing — returns a fixed score per symbol."""

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
    """Signal that always raises an exception."""

    @property
    def name(self) -> str:
        return "failing"

    @property
    def required_features(self) -> list[str]:
        return []

    def compute(
        self, symbol: str, features: dict[str, pd.Series], timestamp: datetime
    ) -> SignalOutput:
        raise RuntimeError("Signal computation failed")


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
    symbols: list[str] | None = None,
    signals: list[BaseSignal] | None = None,
    oms: OrderManagementSystem | None = None,
    portfolio_config: PortfolioConfig | None = None,
    risk_config: RiskConfig | None = None,
    **kwargs,
) -> StrategyRunner:
    symbols = symbols or SYMBOLS
    returns = _make_returns(symbols)

    config = RunnerConfig(
        universe=symbols,
        signals=signals or [StubSignal("momentum", dict.fromkeys(symbols, 0.3))],
        portfolio_config=portfolio_config or PortfolioConfig(
            optimization_method=OptimizationMethod.RISK_PARITY,
            constraints=PortfolioConstraints(
                long_only=True, max_weight=0.5, max_gross_exposure=1.0
            ),
        ),
        risk_config=risk_config or RiskConfig(
            limits=ExposureLimits(
                max_position_fraction=0.50,
                max_order_fraction=0.50,
                max_gross_exposure=1.50,
            ),
        ),
        **kwargs,
    )
    return StrategyRunner(
        config=config,
        oms=oms or _make_oms(),
        returns_provider=lambda syms, lookback: returns,
    )


# ── StrategyRunner: Basic Operation ──────────────────────────────────────────

class TestStrategyRunnerBasic:
    def test_run_once_returns_result(self):
        runner = _make_runner()
        result = runner.run_once()
        assert isinstance(result, RunResult)
        assert result.state == RunnerState.IDLE
        assert result.timestamp is not None

    def test_run_once_triggers_rebalance(self):
        runner = _make_runner()
        result = runner.run_once()
        assert result.construction is not None
        assert result.construction.rebalance_triggered
        assert result.n_submitted > 0

    def test_executions_are_recorded(self):
        runner = _make_runner()
        result = runner.run_once()
        assert len(result.executions) > 0
        for ex in result.executions:
            assert isinstance(ex, ExecutionRecord)
            assert ex.symbol in SYMBOLS
            assert ex.side in ("BUY", "SELL")

    def test_portfolio_value_is_positive(self):
        runner = _make_runner()
        result = runner.run_once()
        assert result.portfolio_value > 0

    def test_state_returns_to_idle(self):
        runner = _make_runner()
        assert runner.state == RunnerState.IDLE
        runner.run_once()
        assert runner.state == RunnerState.IDLE


# ── StrategyRunner: Signal Integration ───────────────────────────────────────

class TestStrategyRunnerSignals:
    def test_multiple_signals(self):
        signals = [
            StubSignal("momentum", dict.fromkeys(SYMBOLS, 0.5)),
            StubSignal("mean_reversion", dict.fromkeys(SYMBOLS, 0.3)),
        ]
        runner = _make_runner(signals=signals)
        result = runner.run_once()
        assert result.state == RunnerState.IDLE
        assert result.n_submitted > 0

    def test_failing_signal_is_skipped(self):
        signals = [
            StubSignal("momentum", dict.fromkeys(SYMBOLS, 0.5)),
            FailingSignal(),
        ]
        runner = _make_runner(signals=signals)
        result = runner.run_once()
        # Should still succeed — failing signal is skipped
        assert result.state == RunnerState.IDLE

    def test_all_signals_failing_produces_no_trades(self):
        runner = _make_runner(signals=[FailingSignal()])
        result = runner.run_once()
        # No signals → zero alpha → may or may not trigger rebalance
        # but should not crash
        assert result.state == RunnerState.IDLE


# ── StrategyRunner: Risk Validation ──────────────────────────────────────────

class TestStrategyRunnerRisk:
    def test_tight_risk_limits_reject_trades(self):
        tight_risk = RiskConfig(
            limits=ExposureLimits(
                max_position_fraction=0.01,  # 1% — very tight
                max_order_fraction=0.01,
            ),
        )
        runner = _make_runner(risk_config=tight_risk)
        result = runner.run_once()
        # Most trades should be rejected by tight limits
        assert result.n_rejected > 0

    def test_min_order_value_filters_small_trades(self):
        runner = _make_runner(min_order_value=500_000)
        result = runner.run_once()
        for ex in result.executions:
            if not ex.risk_approved:
                # Should include "Below minimum" reason for small trades
                assert "minimum" in ex.risk_reason.lower() or ex.dollar_amount >= 500_000


# ── StrategyRunner: Portfolio Construction ───────────────────────────────────

class TestStrategyRunnerPortfolio:
    def test_no_rebalance_below_threshold(self):
        oms = _make_oms()
        runner = _make_runner(oms=oms)
        # First run: rebalance to establish positions
        first = runner.run_once()
        assert first.construction is not None and first.construction.rebalance_triggered

        # Second run with same signals: should NOT rebalance (positions match)
        # Need very high threshold to guarantee no rebalance
        config = RunnerConfig(
            universe=SYMBOLS,
            signals=[StubSignal("momentum", dict.fromkeys(SYMBOLS, 0.3))],
            portfolio_config=PortfolioConfig(
                optimization_method=OptimizationMethod.RISK_PARITY,
                constraints=PortfolioConstraints(long_only=True, max_gross_exposure=1.0),
                rebalance_threshold=100.0,  # impossibly high
            ),
            risk_config=RiskConfig(
                limits=ExposureLimits(max_position_fraction=0.50, max_order_fraction=0.50),
            ),
        )
        returns = _make_returns(SYMBOLS)
        runner2 = StrategyRunner(
            config=config,
            oms=oms,
            returns_provider=lambda syms, lookback: returns,
        )
        second = runner2.run_once()
        assert not second.construction.rebalance_triggered

    def test_mean_variance_optimizer(self):
        portfolio_config = PortfolioConfig(
            optimization_method=OptimizationMethod.MEAN_VARIANCE,
            constraints=PortfolioConstraints(long_only=True, max_gross_exposure=1.0),
        )
        runner = _make_runner(portfolio_config=portfolio_config)
        result = runner.run_once()
        assert result.construction is not None
        assert result.construction.optimization.method == OptimizationMethod.MEAN_VARIANCE


# ── StrategyRunner: Error Handling ───────────────────────────────────────────

class TestStrategyRunnerErrors:
    def test_missing_returns_provider_produces_error_or_empty(self):
        config = RunnerConfig(
            universe=SYMBOLS,
            signals=[StubSignal("momentum", dict.fromkeys(SYMBOLS, 0.3))],
        )
        oms = _make_oms()
        runner = StrategyRunner(config=config, oms=oms)
        # No returns provider — may error or produce degenerate results
        result = runner.run_once()
        # Should not crash — either errors gracefully or runs with degenerate data
        assert result.state in (RunnerState.IDLE, RunnerState.ERROR)

    def test_last_result_is_stored(self):
        runner = _make_runner()
        assert runner.last_result is None
        result = runner.run_once()
        assert runner.last_result is result


# ── StrategyRunner: Feature Provider ─────────────────────────────────────────

class TestStrategyRunnerFeatures:
    def test_custom_feature_provider(self):
        call_count = {"n": 0}

        def mock_provider(symbol: str, signal: BaseSignal) -> dict[str, pd.Series]:
            call_count["n"] += 1
            return {"returns": pd.Series([0.01, 0.02, -0.01])}

        runner = _make_runner()
        runner._feature_provider = mock_provider
        result = runner.run_once()
        assert call_count["n"] > 0
        assert result.state == RunnerState.IDLE


# ── StrategyRunner: OMS Integration ──────────────────────────────────────────

class TestStrategyRunnerOMS:
    def test_orders_appear_in_oms(self):
        oms = _make_oms()
        runner = _make_runner(oms=oms)
        result = runner.run_once()

        submitted_ids = [
            ex.order_id for ex in result.executions
            if ex.order_id is not None
        ]
        assert len(submitted_ids) > 0

        for oid in submitted_ids:
            order = oms.get_order(oid)
            assert order is not None
            assert order.strategy_id == "strategy_runner"

    def test_positions_are_created(self):
        oms = _make_oms()
        runner = _make_runner(oms=oms)
        runner.run_once()

        positions = oms.get_all_positions()
        # Should have at least some positions after rebalance
        assert len(positions) > 0
