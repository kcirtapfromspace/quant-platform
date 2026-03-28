"""Backtest parameter sensitivity analysis.

Systematically varies key backtest parameters and measures the stability
of results across the parameter grid.  Helps PMs understand which parameters
the strategy is robust to, and which have outsized influence on reported
performance — a critical diligence step before live deployment.

Key outputs:

  * **Per-parameter sensitivity**: how Sharpe / CAGR / max-DD change
    when a single parameter varies while others hold at base values.
  * **Stability score**: fraction of the parameter grid that produces
    a Sharpe above a user-defined threshold (default 0.0).
  * **Parameter importance ranking**: which parameters have the largest
    impact on Sharpe, measured by range across their sweep.

Usage::

    from quant.backtest.sensitivity import (
        SensitivityAnalyzer,
        SensitivityConfig,
        ParameterSweep,
    )

    config = SensitivityConfig(
        base_config=my_backtest_config,
        sweeps=[
            ParameterSweep("rebalance_frequency", [5, 10, 21, 42, 63]),
            ParameterSweep("commission_bps", [5.0, 10.0, 20.0]),
        ],
    )
    analyzer = SensitivityAnalyzer()
    result = analyzer.run(daily_returns, config)
    print(result.summary())
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from quant.backtest.multi_strategy import (
    MultiStrategyBacktestEngine,
    MultiStrategyConfig,
)

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ParameterSweep:
    """Definition of one parameter to vary.

    Attributes:
        parameter:  Dot-separated path to the config field (e.g.
                    ``"rebalance_frequency"``, ``"sleeves.0.capital_weight"``).
        values:     List of values to sweep.
        label:      Display label (defaults to ``parameter``).
    """

    parameter: str
    values: list = field(default_factory=list)
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.parameter


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParameterPoint:
    """Result of a single backtest run within a sensitivity sweep."""

    parameter: str
    value: object
    sharpe: float
    cagr: float
    max_drawdown: float
    volatility: float
    total_return: float
    n_rebalances: int


@dataclass(frozen=True, slots=True)
class ParameterSensitivity:
    """Sensitivity of results to one parameter."""

    parameter: str
    label: str
    points: list[ParameterPoint]

    # Derived metrics
    sharpe_range: float
    cagr_range: float
    max_drawdown_range: float
    best_sharpe_value: object
    worst_sharpe_value: object


@dataclass
class SensitivityResult:
    """Complete sensitivity analysis results."""

    n_runs: int
    base_sharpe: float
    base_cagr: float
    base_max_drawdown: float

    sensitivities: list[ParameterSensitivity]
    parameter_importance: list[tuple[str, float]]  # (param, sharpe_range) sorted desc
    stability_score: float  # fraction of runs with Sharpe > threshold

    all_points: list[ParameterPoint] = field(repr=False)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Sensitivity Analysis ({self.n_runs} runs)",
            "=" * 60,
            "",
            f"Base: Sharpe={self.base_sharpe:.2f}  CAGR={self.base_cagr:+.2%}"
            f"  MaxDD={self.base_max_drawdown:.2%}",
            f"Stability score: {self.stability_score:.1%} of runs Sharpe > 0",
            "",
            "Parameter Importance (by Sharpe range):",
            "-" * 60,
        ]
        for param, impact in self.parameter_importance:
            lines.append(f"  {param:<30s}  {impact:.3f}")

        lines.extend(["", "Per-Parameter Sensitivity:", "-" * 60])
        for sens in self.sensitivities:
            lines.append(f"\n  {sens.label}:")
            lines.append(
                f"    {'Value':<15s} {'Sharpe':>8s} {'CAGR':>9s} "
                f"{'MaxDD':>8s} {'Vol':>8s}"
            )
            for pt in sens.points:
                lines.append(
                    f"    {str(pt.value):<15s} {pt.sharpe:>8.2f} "
                    f"{pt.cagr:>+8.2%} {pt.max_drawdown:>7.2%} "
                    f"{pt.volatility:>7.2%}"
                )
            lines.append(
                f"    Best Sharpe @ {sens.best_sharpe_value}  |  "
                f"Worst @ {sens.worst_sharpe_value}  |  "
                f"Range: {sens.sharpe_range:.3f}"
            )

        return "\n".join(lines)


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis.

    Attributes:
        base_config:        Base multi-strategy config to perturb.
        sweeps:             Parameters to sweep.
        sharpe_threshold:   Threshold for stability score computation.
    """

    base_config: MultiStrategyConfig = field(
        default_factory=MultiStrategyConfig
    )
    sweeps: list[ParameterSweep] = field(default_factory=list)
    sharpe_threshold: float = 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SensitivityAnalyzer:
    """Backtest parameter sensitivity analyzer."""

    def run(
        self,
        returns: pd.DataFrame,
        config: SensitivityConfig,
    ) -> SensitivityResult:
        """Run sensitivity analysis across parameter sweeps.

        For each :class:`ParameterSweep`, each value is set on a copy of
        ``base_config`` while all other parameters remain at their base.
        A backtest is executed for each variant and results are collected.

        Args:
            returns: DataFrame of daily returns (DatetimeIndex x symbols).
            config:  Sensitivity configuration.

        Returns:
            :class:`SensitivityResult` with per-parameter diagnostics.
        """
        engine = MultiStrategyBacktestEngine()

        # Run base case
        base_report = engine.run(returns, config.base_config)
        base_sharpe = base_report.sharpe_ratio
        base_cagr = base_report.cagr
        base_mdd = base_report.max_drawdown

        all_points: list[ParameterPoint] = []
        sensitivities: list[ParameterSensitivity] = []

        for sweep in config.sweeps:
            points: list[ParameterPoint] = []

            for val in sweep.values:
                variant = self._make_variant(config.base_config, sweep.parameter, val)
                try:
                    report = engine.run(returns, variant)
                    pt = ParameterPoint(
                        parameter=sweep.parameter,
                        value=val,
                        sharpe=report.sharpe_ratio,
                        cagr=report.cagr,
                        max_drawdown=report.max_drawdown,
                        volatility=report.volatility,
                        total_return=report.total_return,
                        n_rebalances=report.n_rebalances,
                    )
                except Exception:
                    logger.warning(
                        "Sensitivity run failed for {}={}", sweep.parameter, val
                    )
                    pt = ParameterPoint(
                        parameter=sweep.parameter,
                        value=val,
                        sharpe=float("nan"),
                        cagr=float("nan"),
                        max_drawdown=float("nan"),
                        volatility=float("nan"),
                        total_return=float("nan"),
                        n_rebalances=0,
                    )

                points.append(pt)
                all_points.append(pt)

            # Compute per-parameter metrics
            valid = [p for p in points if np.isfinite(p.sharpe)]
            if valid:
                sharpes = [p.sharpe for p in valid]
                cagrs = [p.cagr for p in valid]
                mdds = [p.max_drawdown for p in valid]

                best_pt = max(valid, key=lambda p: p.sharpe)
                worst_pt = min(valid, key=lambda p: p.sharpe)

                sens = ParameterSensitivity(
                    parameter=sweep.parameter,
                    label=sweep.label,
                    points=points,
                    sharpe_range=max(sharpes) - min(sharpes),
                    cagr_range=max(cagrs) - min(cagrs),
                    max_drawdown_range=max(mdds) - min(mdds),
                    best_sharpe_value=best_pt.value,
                    worst_sharpe_value=worst_pt.value,
                )
            else:
                sens = ParameterSensitivity(
                    parameter=sweep.parameter,
                    label=sweep.label,
                    points=points,
                    sharpe_range=0.0,
                    cagr_range=0.0,
                    max_drawdown_range=0.0,
                    best_sharpe_value=None,
                    worst_sharpe_value=None,
                )
            sensitivities.append(sens)

        # Parameter importance ranking (by Sharpe range, descending)
        importance = sorted(
            [(s.label, s.sharpe_range) for s in sensitivities],
            key=lambda x: x[1],
            reverse=True,
        )

        # Stability score: fraction of valid runs with Sharpe > threshold
        valid_all = [p for p in all_points if np.isfinite(p.sharpe)]
        if valid_all:
            above = sum(1 for p in valid_all if p.sharpe > config.sharpe_threshold)
            stability = above / len(valid_all)
        else:
            stability = 0.0

        result = SensitivityResult(
            n_runs=len(all_points),
            base_sharpe=base_sharpe,
            base_cagr=base_cagr,
            base_max_drawdown=base_mdd,
            sensitivities=sensitivities,
            parameter_importance=importance,
            stability_score=stability,
            all_points=all_points,
        )

        logger.info(
            "Sensitivity analysis: {} runs | base Sharpe={:.2f} | "
            "stability={:.1%} | most impactful={}",
            result.n_runs,
            base_sharpe,
            stability,
            importance[0][0] if importance else "N/A",
        )

        return result

    @staticmethod
    def _make_variant(
        base: MultiStrategyConfig,
        parameter: str,
        value: object,
    ) -> MultiStrategyConfig:
        """Create a config variant with one parameter changed.

        Supports dot-separated paths for nested fields, e.g.
        ``"sleeves.0.capital_weight"`` sets ``config.sleeves[0].capital_weight``.
        """
        variant = copy.deepcopy(base)
        parts = parameter.split(".")
        obj: object = variant

        for part in parts[:-1]:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)  # type: ignore[index]

        final = parts[-1]
        if final.isdigit():
            obj[int(final)] = value  # type: ignore[index]
        else:
            setattr(obj, final, value)

        return variant
