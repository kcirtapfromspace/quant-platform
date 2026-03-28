"""Tests for backtest parameter sensitivity analysis (QUA-81)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quant.backtest.multi_strategy import (
    MultiStrategyConfig,
    SleeveConfig,
)
from quant.backtest.sensitivity import (
    ParameterPoint,
    ParameterSweep,
    SensitivityAnalyzer,
    SensitivityConfig,
    SensitivityResult,
)
from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.engine import PortfolioConfig
from quant.portfolio.optimizers import OptimizationMethod
from quant.signals.base import BaseSignal, SignalOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOLS = ["AAPL", "GOOG", "MSFT"]


class _StubSignal(BaseSignal):
    def __init__(self, name: str = "stub", offset: float = 0.0) -> None:
        self._name = name
        self._offset = offset

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> list[str]:
        return []

    def compute(self, symbol, features, timestamp):
        score = 0.5 + self._offset
        return SignalOutput(
            symbol=symbol,
            timestamp=timestamp,
            score=score,
            confidence=0.8,
            target_position=score * 0.8,
            metadata={"signal_name": self._name},
        )


def _make_returns(n_days: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    factor = rng.normal(0.0005, 0.01, size=n_days)
    idio = rng.normal(0, 0.015, size=(n_days, len(SYMBOLS)))
    betas = rng.uniform(0.5, 1.5, size=len(SYMBOLS))
    data = factor[:, None] * betas[None, :] + idio
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    return pd.DataFrame(data, index=dates, columns=SYMBOLS)


def _make_base_config() -> MultiStrategyConfig:
    sleeves = [
        SleeveConfig(
            name="momentum",
            signals=[_StubSignal("mom", 0.1)],
            capital_weight=0.6,
            strategy_type="momentum",
            portfolio_config=PortfolioConfig(
                optimization_method=OptimizationMethod.RISK_PARITY,
                constraints=PortfolioConstraints(
                    long_only=True, max_weight=0.5, max_gross_exposure=1.0
                ),
            ),
        ),
        SleeveConfig(
            name="mean_rev",
            signals=[_StubSignal("mr", 0.2)],
            capital_weight=0.4,
            strategy_type="mean_reversion",
            portfolio_config=PortfolioConfig(
                optimization_method=OptimizationMethod.RISK_PARITY,
                constraints=PortfolioConstraints(
                    long_only=True, max_weight=0.5, max_gross_exposure=1.0
                ),
            ),
        ),
    ]
    return MultiStrategyConfig(
        sleeves=sleeves,
        rebalance_frequency=21,
        commission_bps=10.0,
        min_history=60,
    )


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[ParameterSweep("rebalance_frequency", [10, 21])],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert isinstance(result, SensitivityResult)

    def test_n_runs_tracked(self):
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[ParameterSweep("rebalance_frequency", [10, 21, 42])],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert result.n_runs == 3

    def test_base_metrics_populated(self):
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[ParameterSweep("rebalance_frequency", [21])],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert math.isfinite(result.base_sharpe)
        assert math.isfinite(result.base_cagr)
        assert math.isfinite(result.base_max_drawdown)

    def test_multiple_sweeps(self):
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[
                ParameterSweep("rebalance_frequency", [10, 21]),
                ParameterSweep("commission_bps", [5.0, 10.0, 20.0]),
            ],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert result.n_runs == 5  # 2 + 3
        assert len(result.sensitivities) == 2


# ---------------------------------------------------------------------------
# Sensitivity metrics
# ---------------------------------------------------------------------------


class TestSensitivityMetrics:
    def test_sharpe_range_non_negative(self):
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[ParameterSweep("rebalance_frequency", [5, 10, 21, 42])],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        for sens in result.sensitivities:
            assert sens.sharpe_range >= 0

    def test_best_worst_different_with_range(self):
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[ParameterSweep("rebalance_frequency", [5, 63])],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        sens = result.sensitivities[0]
        if sens.sharpe_range > 0.001:
            assert sens.best_sharpe_value != sens.worst_sharpe_value

    def test_points_match_sweep_values(self):
        values = [10, 21, 42]
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[ParameterSweep("rebalance_frequency", values)],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        pts = result.sensitivities[0].points
        assert [p.value for p in pts] == values

    def test_per_point_metrics_finite(self):
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[ParameterSweep("rebalance_frequency", [10, 21])],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        for pt in result.all_points:
            assert isinstance(pt, ParameterPoint)
            assert math.isfinite(pt.sharpe)
            assert math.isfinite(pt.cagr)
            assert math.isfinite(pt.max_drawdown)
            assert math.isfinite(pt.volatility)


# ---------------------------------------------------------------------------
# Parameter importance
# ---------------------------------------------------------------------------


class TestParameterImportance:
    def test_importance_sorted_descending(self):
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[
                ParameterSweep("rebalance_frequency", [5, 21, 63]),
                ParameterSweep("commission_bps", [5.0, 10.0, 20.0]),
            ],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        impacts = [v for _, v in result.parameter_importance]
        assert impacts == sorted(impacts, reverse=True)

    def test_importance_includes_all_params(self):
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[
                ParameterSweep("rebalance_frequency", [10, 21]),
                ParameterSweep("commission_bps", [5.0, 10.0]),
            ],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        param_names = {p for p, _ in result.parameter_importance}
        assert "rebalance_frequency" in param_names
        assert "commission_bps" in param_names


# ---------------------------------------------------------------------------
# Stability score
# ---------------------------------------------------------------------------


class TestStabilityScore:
    def test_stability_between_zero_and_one(self):
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[ParameterSweep("rebalance_frequency", [10, 21, 42])],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert 0.0 <= result.stability_score <= 1.0

    def test_high_threshold_lowers_stability(self):
        returns = _make_returns()
        analyzer = SensitivityAnalyzer()
        low = analyzer.run(
            returns,
            SensitivityConfig(
                base_config=_make_base_config(),
                sweeps=[ParameterSweep("rebalance_frequency", [5, 10, 21, 42])],
                sharpe_threshold=-10.0,
            ),
        )
        high = analyzer.run(
            returns,
            SensitivityConfig(
                base_config=_make_base_config(),
                sweeps=[ParameterSweep("rebalance_frequency", [5, 10, 21, 42])],
                sharpe_threshold=10.0,
            ),
        )
        assert low.stability_score >= high.stability_score


# ---------------------------------------------------------------------------
# Nested parameter paths
# ---------------------------------------------------------------------------


class TestNestedParameters:
    def test_nested_sleeve_parameter(self):
        """Should be able to vary a nested sleeve config field."""
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[
                ParameterSweep(
                    "sleeves.0.capital_weight",
                    [0.3, 0.5, 0.7],
                    label="momentum_weight",
                ),
            ],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert result.n_runs == 3
        sens = result.sensitivities[0]
        assert sens.label == "momentum_weight"

    def test_custom_label(self):
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[
                ParameterSweep(
                    "rebalance_frequency",
                    [10, 21],
                    label="Rebalance Period",
                ),
            ],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert result.sensitivities[0].label == "Rebalance Period"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_value_sweep(self):
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[ParameterSweep("rebalance_frequency", [21])],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert result.n_runs == 1
        assert result.sensitivities[0].sharpe_range == 0.0

    def test_empty_sweeps(self):
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        assert result.n_runs == 0
        assert result.stability_score == 0.0

    def test_commission_sensitivity(self):
        """Higher costs should generally reduce Sharpe."""
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[ParameterSweep("commission_bps", [0.0, 50.0, 100.0])],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        pts = result.sensitivities[0].points
        # Higher commission should not improve Sharpe
        assert pts[0].sharpe >= pts[-1].sharpe - 0.1


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[
                ParameterSweep("rebalance_frequency", [10, 21, 42]),
                ParameterSweep("commission_bps", [5.0, 10.0]),
            ],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        summary = result.summary()
        assert "Sensitivity Analysis" in summary
        assert "Sharpe" in summary
        assert "Stability" in summary
        assert "rebalance_frequency" in summary
        assert "commission_bps" in summary

    def test_summary_includes_base(self):
        config = SensitivityConfig(
            base_config=_make_base_config(),
            sweeps=[ParameterSweep("rebalance_frequency", [21])],
        )
        analyzer = SensitivityAnalyzer()
        result = analyzer.run(_make_returns(), config)
        summary = result.summary()
        assert "Base:" in summary
