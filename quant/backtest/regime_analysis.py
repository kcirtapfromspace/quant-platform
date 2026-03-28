"""Regime-conditioned performance analysis for multi-strategy backtests.

Takes a completed :class:`MultiStrategyBacktestReport` with regime history
and decomposes performance by detected market regime.  This helps PMs
understand *when* strategies make or lose money, and calibrate regime-aware
capital allocation parameters.

Key outputs:

  * **Per-regime Sharpe / vol / max-drawdown / total-return** for the
    portfolio and each sleeve.
  * **Regime duration statistics** — how long each regime persisted.
  * **Transition matrix** — probabilities of moving between regimes.

Usage::

    from quant.backtest.regime_analysis import RegimeAnalyzer

    analyzer = RegimeAnalyzer()
    result = analyzer.analyze(backtest_report)
    print(result.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.backtest import metrics as m
from quant.backtest.multi_strategy import MultiStrategyBacktestReport

# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegimePerformance:
    """Performance metrics for a single regime period."""

    regime: str
    n_days: int
    total_return: float
    annualized_return: float
    volatility: float
    sharpe: float
    max_drawdown: float
    pct_of_total_days: float


@dataclass(frozen=True, slots=True)
class SleeveRegimePerformance:
    """Per-sleeve performance within a specific regime."""

    sleeve: str
    regime: str
    total_return: float
    volatility: float
    sharpe: float


@dataclass
class RegimeTransitionMatrix:
    """Regime transition probability matrix.

    ``matrix[from_regime][to_regime]`` gives the empirical probability
    of transitioning from one regime to another.
    """

    regimes: list[str]
    matrix: dict[str, dict[str, float]]

    def get(self, from_regime: str, to_regime: str) -> float:
        return self.matrix.get(from_regime, {}).get(to_regime, 0.0)


@dataclass
class RegimeAnalysisResult:
    """Complete regime-conditioned analysis results."""

    n_regimes: int
    regime_labels: list[str]

    # Portfolio-level per-regime metrics
    portfolio_by_regime: list[RegimePerformance]

    # Per-sleeve per-regime metrics
    sleeve_by_regime: list[SleeveRegimePerformance]

    # Regime duration statistics
    avg_regime_duration: dict[str, float]
    max_regime_duration: dict[str, int]

    # Transition matrix
    transition_matrix: RegimeTransitionMatrix

    # Best and worst regimes
    best_regime: str
    worst_regime: str

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Regime-Conditioned Analysis ({self.n_regimes} regimes)",
            "=" * 60,
            "",
            f"{'Regime':<18s} {'Days':>6s} {'Return':>9s} {'Sharpe':>8s} "
            f"{'Vol':>8s} {'MaxDD':>8s} {'%Days':>7s}",
            "-" * 60,
        ]

        for rp in sorted(
            self.portfolio_by_regime, key=lambda x: x.sharpe, reverse=True
        ):
            lines.append(
                f"{rp.regime:<18s} {rp.n_days:>6d} "
                f"{rp.total_return:>+8.2%} {rp.sharpe:>8.2f} "
                f"{rp.volatility:>7.2%} {rp.max_drawdown:>7.2%} "
                f"{rp.pct_of_total_days:>6.1%}"
            )

        lines.extend([
            "",
            f"Best regime  : {self.best_regime}",
            f"Worst regime : {self.worst_regime}",
        ])

        # Regime durations
        if self.avg_regime_duration:
            lines.append("")
            lines.append("Avg regime duration (days):")
            for regime, dur in sorted(self.avg_regime_duration.items()):
                lines.append(f"  {regime:<18s}: {dur:.1f}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RegimeAnalyzer:
    """Regime-conditioned performance analyzer."""

    def analyze(
        self,
        report: MultiStrategyBacktestReport,
    ) -> RegimeAnalysisResult:
        """Analyze backtest performance conditioned on detected regime.

        Args:
            report: Completed multi-strategy backtest report with
                    ``regime_history`` populated.

        Returns:
            :class:`RegimeAnalysisResult` with per-regime metrics.

        Raises:
            ValueError: If regime_history is empty.
        """
        if report.regime_history.empty:
            raise ValueError(
                "regime_history is empty — run backtest with regime_config"
            )

        # Build daily regime labels by forward-filling rebalance-day labels
        daily_regime = self._build_daily_regime(
            report.regime_history, report.returns_series.index
        )

        returns = report.returns_series
        regime_labels = sorted(daily_regime.dropna().unique().tolist())

        # Portfolio-level per-regime analysis
        portfolio_by_regime: list[RegimePerformance] = []
        total_days = len(returns)

        for regime in regime_labels:
            mask = daily_regime == regime
            regime_returns = returns[mask]
            perf = self._compute_performance(regime, regime_returns, total_days)
            portfolio_by_regime.append(perf)

        # Per-sleeve per-regime analysis
        sleeve_by_regime: list[SleeveRegimePerformance] = []
        if not report.sleeve_returns.empty:
            for sleeve_name in report.sleeve_returns.columns:
                sleeve_rets = report.sleeve_returns[sleeve_name]
                for regime in regime_labels:
                    mask = daily_regime == regime
                    sr = sleeve_rets[mask]
                    sleeve_by_regime.append(
                        self._compute_sleeve_performance(
                            sleeve_name, regime, sr
                        )
                    )

        # Regime duration statistics
        avg_dur, max_dur = self._compute_durations(daily_regime, regime_labels)

        # Transition matrix
        tm = self._compute_transitions(daily_regime, regime_labels)

        # Best/worst regime by Sharpe
        best = max(portfolio_by_regime, key=lambda x: x.sharpe)
        worst = min(portfolio_by_regime, key=lambda x: x.sharpe)

        return RegimeAnalysisResult(
            n_regimes=len(regime_labels),
            regime_labels=regime_labels,
            portfolio_by_regime=portfolio_by_regime,
            sleeve_by_regime=sleeve_by_regime,
            avg_regime_duration=avg_dur,
            max_regime_duration=max_dur,
            transition_matrix=tm,
            best_regime=best.regime,
            worst_regime=worst.regime,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_daily_regime(
        regime_history: pd.Series, full_index: pd.DatetimeIndex
    ) -> pd.Series:
        """Forward-fill regime labels from rebalance points to daily."""
        daily = pd.Series(index=full_index, dtype=object)
        daily.loc[regime_history.index] = regime_history.values
        daily = daily.ffill()
        return daily

    @staticmethod
    def _compute_performance(
        regime: str, returns: pd.Series, total_days: int
    ) -> RegimePerformance:
        n = len(returns)
        if n < 2:
            return RegimePerformance(
                regime=regime,
                n_days=n,
                total_return=float(returns.sum()) if n > 0 else 0.0,
                annualized_return=0.0,
                volatility=0.0,
                sharpe=0.0,
                max_drawdown=0.0,
                pct_of_total_days=n / total_days if total_days > 0 else 0.0,
            )

        total_ret = float((1 + returns).prod() - 1)
        n_years = n / 252
        ann_ret = (
            float((1 + total_ret) ** (1 / n_years) - 1)
            if n_years > 0 and (1 + total_ret) > 0
            else 0.0
        )
        vol = float(returns.std() * math.sqrt(252))
        sharpe = m.sharpe_ratio(returns)
        equity = (1 + returns).cumprod()
        mdd = m.max_drawdown(equity)

        return RegimePerformance(
            regime=regime,
            n_days=n,
            total_return=total_ret,
            annualized_return=ann_ret,
            volatility=vol,
            sharpe=sharpe,
            max_drawdown=mdd,
            pct_of_total_days=n / total_days if total_days > 0 else 0.0,
        )

    @staticmethod
    def _compute_sleeve_performance(
        sleeve: str, regime: str, returns: pd.Series
    ) -> SleeveRegimePerformance:
        n = len(returns)
        if n < 2:
            return SleeveRegimePerformance(
                sleeve=sleeve,
                regime=regime,
                total_return=float(returns.sum()) if n > 0 else 0.0,
                volatility=0.0,
                sharpe=0.0,
            )

        total_ret = float((1 + returns).prod() - 1)
        vol = float(returns.std() * math.sqrt(252))
        sharpe = m.sharpe_ratio(returns)

        return SleeveRegimePerformance(
            sleeve=sleeve,
            regime=regime,
            total_return=total_ret,
            volatility=vol,
            sharpe=sharpe,
        )

    @staticmethod
    def _compute_durations(
        daily_regime: pd.Series, regime_labels: list[str]
    ) -> tuple[dict[str, float], dict[str, int]]:
        """Compute average and max duration per regime."""
        avg_dur: dict[str, float] = {}
        max_dur: dict[str, int] = {}

        values = daily_regime.dropna().values
        if len(values) == 0:
            return avg_dur, max_dur

        # Run-length encoding
        runs: list[tuple[str, int]] = []
        current = values[0]
        length = 1
        for i in range(1, len(values)):
            if values[i] == current:
                length += 1
            else:
                runs.append((current, length))
                current = values[i]
                length = 1
        runs.append((current, length))

        for regime in regime_labels:
            durations = [d for r, d in runs if r == regime]
            if durations:
                avg_dur[regime] = float(np.mean(durations))
                max_dur[regime] = max(durations)

        return avg_dur, max_dur

    @staticmethod
    def _compute_transitions(
        daily_regime: pd.Series, regime_labels: list[str]
    ) -> RegimeTransitionMatrix:
        """Compute regime transition probability matrix."""
        counts: dict[str, dict[str, int]] = {
            r: dict.fromkeys(regime_labels, 0) for r in regime_labels
        }

        values = daily_regime.dropna().values
        for i in range(1, len(values)):
            prev, curr = values[i - 1], values[i]
            if prev in counts and curr in counts[prev]:
                counts[prev][curr] += 1

        # Normalize to probabilities
        matrix: dict[str, dict[str, float]] = {}
        for from_r in regime_labels:
            total = sum(counts[from_r].values())
            if total > 0:
                matrix[from_r] = {
                    to_r: c / total for to_r, c in counts[from_r].items()
                }
            else:
                matrix[from_r] = dict.fromkeys(regime_labels, 0.0)

        return RegimeTransitionMatrix(regimes=regime_labels, matrix=matrix)
