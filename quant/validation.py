"""Strategy validation — backtest vs live performance comparison.

Provides a structured framework for comparing backtest expectations
against live trading performance, detecting performance drift, and
generating deployment recommendations (continue / reduce / halt).

Usage::

    from quant.validation import (
        StrategyValidator,
        ValidationConfig,
        LiveMetrics,
    )

    validator = StrategyValidator(ValidationConfig(
        max_sharpe_decay=0.30,
        max_drawdown_growth=0.50,
    ))

    result = validator.validate(
        backtest=backtest_report,
        live=LiveMetrics.from_equity_curve(live_equity_series),
    )

    print(result.recommendation)  # CONTINUE / REDUCE_CAPITAL / HALT
    print(result.summary())
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from quant.backtest.report import BacktestReport


class Recommendation(enum.Enum):
    """Deployment recommendation based on validation results."""

    CONTINUE = "continue"
    REDUCE_CAPITAL = "reduce_capital"
    HALT = "halt"


@dataclass
class ValidationConfig:
    """Thresholds for backtest-vs-live comparison.

    All decay/growth values are fractional (0.30 = 30%).

    Attributes:
        max_sharpe_decay:       Maximum acceptable Sharpe ratio decline.
                                 e.g. 0.30 = live Sharpe can be up to 30% lower.
        max_drawdown_growth:    Maximum acceptable drawdown increase.
                                 e.g. 0.50 = live MaxDD can be 50% worse.
        max_cagr_decay:         Maximum acceptable CAGR decline.
        min_win_rate_ratio:     Minimum live/backtest win rate ratio.
                                 e.g. 0.70 = live win rate must be >= 70% of backtest.
        min_profit_factor_ratio: Minimum live/backtest profit factor ratio.
        min_live_trades:        Minimum trades before validation is meaningful.
        signal_drift_threshold: Max KS statistic for signal distribution
                                 comparison (0–1). Lower = stricter.
    """

    max_sharpe_decay: float = 0.30
    max_drawdown_growth: float = 0.50
    max_cagr_decay: float = 0.40
    min_win_rate_ratio: float = 0.70
    min_profit_factor_ratio: float = 0.60
    min_live_trades: int = 5
    signal_drift_threshold: float = 0.20


@dataclass(frozen=True, slots=True)
class ValidationBreach:
    """A single threshold breach.

    Attributes:
        metric:     Name of the metric that breached.
        backtest:   Backtest value.
        live:       Live value.
        threshold:  The threshold that was exceeded.
        severity:   "warning" or "critical".
        detail:     Human-readable explanation.
    """

    metric: str
    backtest: float
    live: float
    threshold: float
    severity: str
    detail: str


@dataclass
class LiveMetrics:
    """Live trading performance metrics for comparison with backtest.

    Construct manually or via :meth:`from_equity_curve`.

    Attributes:
        sharpe_ratio:    Annualised Sharpe ratio.
        max_drawdown:    Maximum drawdown as positive fraction.
        cagr:            Compound annual growth rate.
        win_rate:        Fraction of winning trades.
        profit_factor:   Gross profit / gross loss.
        total_return:    Total return as decimal (0.10 = +10%).
        n_trades:        Number of completed round-trip trades.
        start_date:      First date of the live period.
        end_date:        Last date of the live period.
        equity_curve:    Optional time-series of portfolio values.
    """

    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    cagr: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_return: float = 0.0
    n_trades: int = 0
    start_date: date | None = None
    end_date: date | None = None
    equity_curve: pd.Series | None = field(default=None, repr=False)

    @classmethod
    def from_equity_curve(
        cls,
        equity: pd.Series,
        trade_returns: pd.Series | None = None,
    ) -> LiveMetrics:
        """Construct LiveMetrics from an equity curve time-series.

        Args:
            equity:         pd.Series indexed by date with portfolio values.
            trade_returns:  Optional pd.Series of per-trade returns (for
                            win_rate and profit_factor).

        Returns:
            Populated LiveMetrics instance.
        """
        if equity.empty:
            return cls()

        returns = equity.pct_change().dropna()
        n_bars = len(returns)

        # Sharpe ratio (annualised)
        if n_bars > 1 and returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        # Max drawdown
        cummax = equity.cummax()
        drawdowns = (cummax - equity) / cummax
        max_dd = float(drawdowns.max()) if not drawdowns.empty else 0.0

        # CAGR
        if n_bars > 0 and equity.iloc[0] > 0:
            total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
            years = n_bars / 252
            if years > 0 and total_ret > -1.0:
                cagr = float((1.0 + total_ret) ** (1.0 / years) - 1.0)
            else:
                cagr = 0.0
        else:
            total_ret = 0.0
            cagr = 0.0

        # Win rate and profit factor from trade returns
        if trade_returns is not None and len(trade_returns) > 0:
            wins = trade_returns[trade_returns > 0]
            losses = trade_returns[trade_returns <= 0]
            win_rate = float(len(wins) / len(trade_returns))
            gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
            gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            n_trades = len(trade_returns)
        else:
            win_rate = 0.0
            profit_factor = 0.0
            n_trades = 0

        start = equity.index[0]
        end = equity.index[-1]
        start_date = start.date() if hasattr(start, "date") else start
        end_date = end.date() if hasattr(end, "date") else end

        return cls(
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            cagr=cagr,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_return=float(total_ret),
            n_trades=n_trades,
            start_date=start_date,
            end_date=end_date,
            equity_curve=equity,
        )


@dataclass
class ValidationResult:
    """Aggregated result of backtest-vs-live validation.

    Attributes:
        passed:         True if all critical thresholds are within bounds.
        recommendation: Deployment recommendation.
        breaches:       List of threshold breaches detected.
        n_warnings:     Number of warning-level breaches.
        n_critical:     Number of critical-level breaches.
        insufficient_data: True if live trades < min_live_trades.
        metrics_comparison: Side-by-side metric comparison dict.
    """

    passed: bool
    recommendation: Recommendation
    breaches: list[ValidationBreach] = field(default_factory=list)
    n_warnings: int = 0
    n_critical: int = 0
    insufficient_data: bool = False
    metrics_comparison: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable validation summary."""
        lines = [
            f"Validation: {'PASSED' if self.passed else 'FAILED'}",
            f"Recommendation: {self.recommendation.value}",
            f"Breaches: {self.n_critical} critical, {self.n_warnings} warning",
        ]
        if self.insufficient_data:
            lines.append("  (insufficient data — results are preliminary)")
        lines.append("─" * 50)
        lines.append(f"{'Metric':<20} {'Backtest':>10} {'Live':>10} {'Delta':>10}")
        lines.append("─" * 50)
        for name, vals in self.metrics_comparison.items():
            bt = vals.get("backtest", 0.0)
            lv = vals.get("live", 0.0)
            delta = vals.get("delta_pct", 0.0)
            lines.append(f"{name:<20} {bt:>10.3f} {lv:>10.3f} {delta:>+9.1f}%")
        if self.breaches:
            lines.append("")
            lines.append("Breaches:")
            for b in self.breaches:
                lines.append(f"  [{b.severity.upper()}] {b.detail}")
        return "\n".join(lines)


class StrategyValidator:
    """Compares backtest expectations to live trading performance.

    Performs metric-by-metric comparison with configurable thresholds.
    Generates a deployment recommendation based on breach severity:
      - 0 critical breaches: CONTINUE
      - 1 critical breach: REDUCE_CAPITAL
      - 2+ critical breaches: HALT

    Args:
        config: Validation thresholds.
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self._config = config or ValidationConfig()

    def validate(
        self,
        backtest: BacktestReport,
        live: LiveMetrics,
    ) -> ValidationResult:
        """Run all validation checks.

        Args:
            backtest:  BacktestReport from the offline evaluation.
            live:      LiveMetrics from the live trading period.

        Returns:
            ValidationResult with pass/fail, breaches, and recommendation.
        """
        breaches: list[ValidationBreach] = []
        insufficient = live.n_trades < self._config.min_live_trades

        # ── Sharpe ratio ─────────────────────────────────────────────
        breaches.extend(self._check_sharpe(backtest, live))

        # ── Max drawdown ─────────────────────────────────────────────
        breaches.extend(self._check_drawdown(backtest, live))

        # ── CAGR ─────────────────────────────────────────────────────
        breaches.extend(self._check_cagr(backtest, live))

        # ── Win rate ─────────────────────────────────────────────────
        breaches.extend(self._check_win_rate(backtest, live))

        # ── Profit factor ────────────────────────────────────────────
        breaches.extend(self._check_profit_factor(backtest, live))

        # Classify
        n_critical = sum(1 for b in breaches if b.severity == "critical")
        n_warnings = sum(1 for b in breaches if b.severity == "warning")

        if n_critical >= 2:
            recommendation = Recommendation.HALT
        elif n_critical == 1:
            recommendation = Recommendation.REDUCE_CAPITAL
        else:
            recommendation = Recommendation.CONTINUE

        # If insufficient data, downgrade HALT to REDUCE_CAPITAL
        if insufficient and recommendation == Recommendation.HALT:
            recommendation = Recommendation.REDUCE_CAPITAL

        passed = n_critical == 0

        # Metrics comparison table
        comparison = self._build_comparison(backtest, live)

        result = ValidationResult(
            passed=passed,
            recommendation=recommendation,
            breaches=breaches,
            n_warnings=n_warnings,
            n_critical=n_critical,
            insufficient_data=insufficient,
            metrics_comparison=comparison,
        )

        self._log_result(result)
        return result

    def check_signal_drift(
        self,
        backtest_signals: pd.Series,
        live_signals: pd.Series,
    ) -> float:
        """Compare signal distributions using the Kolmogorov-Smirnov statistic.

        Args:
            backtest_signals: Signal scores from backtest.
            live_signals:     Signal scores from live trading.

        Returns:
            KS statistic (0–1). Higher = more drift.
            Returns 0.0 if either series is empty.
        """
        bt = backtest_signals.dropna()
        lv = live_signals.dropna()
        if len(bt) < 2 or len(lv) < 2:
            return 0.0

        # Two-sample KS test (manual implementation to avoid scipy dependency)
        combined = np.sort(np.concatenate([bt.values, lv.values]))
        bt_sorted = np.sort(bt.values)
        lv_sorted = np.sort(lv.values)

        bt_cdf = np.searchsorted(bt_sorted, combined, side="right") / len(bt_sorted)
        lv_cdf = np.searchsorted(lv_sorted, combined, side="right") / len(lv_sorted)

        return float(np.max(np.abs(bt_cdf - lv_cdf)))

    def check_equity_correlation(
        self,
        backtest_equity: pd.Series,
        live_equity: pd.Series,
    ) -> float:
        """Compute correlation between backtest and live equity curves.

        Aligns on date index, rebases both to start at 1.0, then computes
        the Pearson correlation of daily returns.

        Args:
            backtest_equity: Backtest equity curve (date-indexed).
            live_equity:     Live equity curve (date-indexed).

        Returns:
            Pearson correlation coefficient (-1 to 1).
            Returns 0.0 if alignment fails or data is insufficient.
        """
        if backtest_equity.empty or live_equity.empty:
            return 0.0

        bt_ret = backtest_equity.pct_change().dropna()
        lv_ret = live_equity.pct_change().dropna()

        # Align on common dates
        common = bt_ret.index.intersection(lv_ret.index)
        if len(common) < 5:
            return 0.0

        bt_aligned = bt_ret.loc[common]
        lv_aligned = lv_ret.loc[common]

        corr = np.corrcoef(bt_aligned.values, lv_aligned.values)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0

    # ── Private checks ───────────────────────────────────────────────

    def _check_sharpe(
        self, bt: BacktestReport, lv: LiveMetrics
    ) -> list[ValidationBreach]:
        if bt.sharpe_ratio <= 0:
            return []  # Can't meaningfully compare negative Sharpe

        decay = 1.0 - (lv.sharpe_ratio / bt.sharpe_ratio) if bt.sharpe_ratio != 0 else 0.0
        threshold = self._config.max_sharpe_decay

        if decay > threshold:
            severity = "critical" if decay > threshold * 1.5 else "warning"
            return [
                ValidationBreach(
                    metric="sharpe_ratio",
                    backtest=bt.sharpe_ratio,
                    live=lv.sharpe_ratio,
                    threshold=threshold,
                    severity=severity,
                    detail=(
                        f"Sharpe decay {decay:.1%} exceeds {threshold:.0%} threshold "
                        f"(backtest {bt.sharpe_ratio:.2f} → live {lv.sharpe_ratio:.2f})"
                    ),
                )
            ]
        return []

    def _check_drawdown(
        self, bt: BacktestReport, lv: LiveMetrics
    ) -> list[ValidationBreach]:
        if bt.max_drawdown <= 0:
            return []

        growth = (lv.max_drawdown / bt.max_drawdown - 1.0) if bt.max_drawdown > 0 else 0.0
        threshold = self._config.max_drawdown_growth

        if growth > threshold:
            severity = "critical" if growth > threshold * 1.5 else "warning"
            return [
                ValidationBreach(
                    metric="max_drawdown",
                    backtest=bt.max_drawdown,
                    live=lv.max_drawdown,
                    threshold=threshold,
                    severity=severity,
                    detail=(
                        f"Max drawdown grew {growth:.1%} beyond {threshold:.0%} threshold "
                        f"(backtest {bt.max_drawdown:.1%} → live {lv.max_drawdown:.1%})"
                    ),
                )
            ]
        return []

    def _check_cagr(
        self, bt: BacktestReport, lv: LiveMetrics
    ) -> list[ValidationBreach]:
        if bt.cagr <= 0:
            return []

        decay = 1.0 - (lv.cagr / bt.cagr) if bt.cagr != 0 else 0.0
        threshold = self._config.max_cagr_decay

        if decay > threshold:
            severity = "critical" if decay > threshold * 1.5 else "warning"
            return [
                ValidationBreach(
                    metric="cagr",
                    backtest=bt.cagr,
                    live=lv.cagr,
                    threshold=threshold,
                    severity=severity,
                    detail=(
                        f"CAGR decay {decay:.1%} exceeds {threshold:.0%} threshold "
                        f"(backtest {bt.cagr:.1%} → live {lv.cagr:.1%})"
                    ),
                )
            ]
        return []

    def _check_win_rate(
        self, bt: BacktestReport, lv: LiveMetrics
    ) -> list[ValidationBreach]:
        if bt.win_rate <= 0 or lv.n_trades == 0:
            return []

        ratio = lv.win_rate / bt.win_rate
        threshold = self._config.min_win_rate_ratio

        if ratio < threshold:
            return [
                ValidationBreach(
                    metric="win_rate",
                    backtest=bt.win_rate,
                    live=lv.win_rate,
                    threshold=threshold,
                    severity="warning",
                    detail=(
                        f"Win rate ratio {ratio:.1%} below {threshold:.0%} threshold "
                        f"(backtest {bt.win_rate:.1%} → live {lv.win_rate:.1%})"
                    ),
                )
            ]
        return []

    def _check_profit_factor(
        self, bt: BacktestReport, lv: LiveMetrics
    ) -> list[ValidationBreach]:
        if bt.profit_factor <= 0 or bt.profit_factor == float("inf") or lv.n_trades == 0:
            return []

        ratio = lv.profit_factor / bt.profit_factor if bt.profit_factor > 0 else 0.0
        threshold = self._config.min_profit_factor_ratio

        if ratio < threshold:
            return [
                ValidationBreach(
                    metric="profit_factor",
                    backtest=bt.profit_factor,
                    live=lv.profit_factor,
                    threshold=threshold,
                    severity="warning",
                    detail=(
                        f"Profit factor ratio {ratio:.1%} below {threshold:.0%} threshold "
                        f"(backtest {bt.profit_factor:.2f} → live {lv.profit_factor:.2f})"
                    ),
                )
            ]
        return []

    @staticmethod
    def _build_comparison(
        bt: BacktestReport, lv: LiveMetrics
    ) -> dict[str, dict[str, float]]:
        """Build a side-by-side metrics comparison dictionary."""

        def _row(bt_val: float, lv_val: float) -> dict[str, float]:
            delta = ((lv_val / bt_val - 1.0) * 100) if bt_val != 0 else 0.0
            return {"backtest": bt_val, "live": lv_val, "delta_pct": delta}

        return {
            "sharpe_ratio": _row(bt.sharpe_ratio, lv.sharpe_ratio),
            "max_drawdown": _row(bt.max_drawdown, lv.max_drawdown),
            "cagr": _row(bt.cagr, lv.cagr),
            "win_rate": _row(bt.win_rate, lv.win_rate),
            "profit_factor": _row(
                bt.profit_factor if bt.profit_factor != float("inf") else 0.0,
                lv.profit_factor if lv.profit_factor != float("inf") else 0.0,
            ),
            "total_return": _row(bt.total_return, lv.total_return),
            "n_trades": _row(float(bt.n_trades), float(lv.n_trades)),
        }

    @staticmethod
    def _log_result(result: ValidationResult) -> None:
        if result.passed:
            logger.info(
                "Validation: PASSED — {} (0 critical, {} warnings)",
                result.recommendation.value,
                result.n_warnings,
            )
        else:
            logger.warning(
                "Validation: FAILED — {} ({} critical, {} warnings)",
                result.recommendation.value,
                result.n_critical,
                result.n_warnings,
            )
