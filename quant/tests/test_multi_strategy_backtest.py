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
from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.engine import PortfolioConfig
from quant.portfolio.lifecycle import LifecycleConfig
from quant.portfolio.optimizers import OptimizationMethod
from quant.portfolio.position_scaler import ScalingConfig, ScalingMethod
from quant.signals.adaptive_combiner import AdaptiveCombinerConfig
from quant.signals.base import BaseSignal, SignalOutput

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
