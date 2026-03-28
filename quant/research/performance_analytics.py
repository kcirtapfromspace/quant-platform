"""Comprehensive strategy performance analytics.

Computes a full suite of performance metrics from a daily return series,
including risk-adjusted returns, drawdown statistics, tail risk measures,
rolling metrics, and win/loss analysis.

Metrics computed:

  * **Return metrics** — total return, CAGR, daily mean, best/worst day.
  * **Risk metrics** — volatility, downside deviation, VaR, CVaR.
  * **Risk-adjusted** — Sharpe, Sortino, Calmar, information ratio.
  * **Drawdown** — max drawdown, avg drawdown, drawdown duration.
  * **Win/loss** — win rate, profit factor, avg win/loss ratio.
  * **Rolling** — rolling Sharpe and volatility series.

Usage::

    from quant.research.performance_analytics import (
        PerformanceAnalyzer,
        AnalyticsConfig,
    )

    analyzer = PerformanceAnalyzer()
    result = analyzer.analyze(daily_returns_series)
    print(result.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AnalyticsConfig:
    """Configuration for performance analytics.

    Attributes:
        risk_free_rate:     Annualised risk-free rate for Sharpe calculation.
        rolling_window:     Window for rolling metrics (trading days).
        var_confidence:     VaR/CVaR confidence level (e.g. 0.95 for 95% VaR).
        annualisation:      Trading days per year.
    """

    risk_free_rate: float = 0.0
    rolling_window: int = 63
    var_confidence: float = 0.95
    annualisation: int = TRADING_DAYS


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DrawdownInfo:
    """Drawdown statistics.

    Attributes:
        max_drawdown:       Deepest peak-to-trough drawdown (negative).
        avg_drawdown:       Average drawdown when in drawdown.
        max_duration_days:  Longest drawdown episode in trading days.
        n_drawdowns:        Number of distinct drawdown episodes.
    """

    max_drawdown: float
    avg_drawdown: float
    max_duration_days: int
    n_drawdowns: int


@dataclass
class PerformanceResult:
    """Complete performance analytics result.

    Attributes:
        n_days:             Number of trading days.
        total_return:       Cumulative return over period.
        cagr:               Compound annual growth rate.
        daily_mean:         Mean daily return.
        daily_std:          Daily standard deviation.
        annualised_vol:     Annualised volatility.
        downside_std:       Downside standard deviation (below 0).
        best_day:           Best single-day return.
        worst_day:          Worst single-day return.
        sharpe:             Annualised Sharpe ratio.
        sortino:            Annualised Sortino ratio.
        calmar:             Calmar ratio (CAGR / |max drawdown|).
        var:                Value at Risk at configured confidence.
        cvar:               Conditional VaR (expected shortfall).
        drawdown:           Drawdown statistics.
        win_rate:           Fraction of positive-return days.
        profit_factor:      Sum of gains / sum of losses.
        avg_win:            Average gain on winning days.
        avg_loss:           Average loss on losing days.
        win_loss_ratio:     avg_win / |avg_loss|.
        skewness:           Return distribution skewness.
        kurtosis:           Return distribution excess kurtosis.
        rolling_sharpe:     Rolling Sharpe series (pd.Series).
        rolling_vol:        Rolling annualised volatility series.
    """

    n_days: int = 0
    total_return: float = 0.0
    cagr: float = 0.0
    daily_mean: float = 0.0
    daily_std: float = 0.0
    annualised_vol: float = 0.0
    downside_std: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    var: float = 0.0
    cvar: float = 0.0
    drawdown: DrawdownInfo = field(
        default_factory=lambda: DrawdownInfo(0.0, 0.0, 0, 0),
    )
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_loss_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    rolling_sharpe: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float), repr=False,
    )
    rolling_vol: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float), repr=False,
    )

    def summary(self) -> str:
        """Return a human-readable performance summary."""
        lines = [
            f"Performance Analytics ({self.n_days} days)",
            "=" * 60,
            "",
            "Return metrics:",
            f"  Total return    : {self.total_return:+.2%}",
            f"  CAGR            : {self.cagr:+.2%}",
            f"  Best day        : {self.best_day:+.2%}",
            f"  Worst day       : {self.worst_day:+.2%}",
            "",
            "Risk metrics:",
            f"  Annualised vol  : {self.annualised_vol:.2%}",
            f"  Downside dev    : {self.downside_std:.4f}",
            f"  VaR (95%)       : {self.var:+.2%}",
            f"  CVaR (95%)      : {self.cvar:+.2%}",
            "",
            "Risk-adjusted:",
            f"  Sharpe          : {self.sharpe:+.2f}",
            f"  Sortino         : {self.sortino:+.2f}",
            f"  Calmar          : {self.calmar:+.2f}",
            "",
            "Drawdown:",
            f"  Max drawdown    : {self.drawdown.max_drawdown:.2%}",
            f"  Avg drawdown    : {self.drawdown.avg_drawdown:.2%}",
            f"  Max duration    : {self.drawdown.max_duration_days} days",
            f"  N episodes      : {self.drawdown.n_drawdowns}",
            "",
            "Win/loss:",
            f"  Win rate        : {self.win_rate:.1%}",
            f"  Profit factor   : {self.profit_factor:.2f}",
            f"  W/L ratio       : {self.win_loss_ratio:.2f}",
            "",
            "Distribution:",
            f"  Skewness        : {self.skewness:+.2f}",
            f"  Excess kurtosis : {self.kurtosis:+.2f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class PerformanceAnalyzer:
    """Comprehensive strategy performance analyzer.

    Args:
        config: Analytics configuration.
    """

    def __init__(self, config: AnalyticsConfig | None = None) -> None:
        self._config = config or AnalyticsConfig()

    @property
    def config(self) -> AnalyticsConfig:
        return self._config

    def analyze(self, returns: pd.Series) -> PerformanceResult:
        """Compute full performance analytics from a daily return series.

        Args:
            returns: Daily returns as a pd.Series (e.g. 0.01 = +1%).

        Returns:
            :class:`PerformanceResult` with comprehensive metrics.

        Raises:
            ValueError: If fewer than 2 observations.
        """
        r = returns.dropna()
        n = len(r)
        if n < 2:
            raise ValueError(f"Need at least 2 observations, got {n}")

        cfg = self._config
        ann = cfg.annualisation
        rf_daily = cfg.risk_free_rate / ann

        r_vals = r.values.astype(np.float64)

        # ── Return metrics ──────────────────────────────────────
        cum = np.cumprod(1.0 + r_vals)
        total_return = float(cum[-1] - 1.0)
        years = n / ann
        cagr = float((cum[-1]) ** (1.0 / years) - 1.0) if years > 0 and cum[-1] > 0 else 0.0
        daily_mean = float(r_vals.mean())
        daily_std = float(r_vals.std(ddof=1))
        ann_vol = daily_std * math.sqrt(ann)
        best = float(r_vals.max())
        worst = float(r_vals.min())

        # ── Downside deviation ──────────────────────────────────
        downside = r_vals[r_vals < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0

        # ── Risk-adjusted ratios ────────────────────────────────
        excess_mean = daily_mean - rf_daily
        sharpe = (excess_mean / daily_std * math.sqrt(ann)) if daily_std > 1e-15 else 0.0
        sortino = (excess_mean / downside_std * math.sqrt(ann)) if downside_std > 1e-15 else 0.0

        # ── VaR / CVaR ─────────────────────────────────────────
        sorted_r = np.sort(r_vals)
        var_idx = int(np.floor((1 - cfg.var_confidence) * n))
        var_idx = max(0, min(var_idx, n - 1))
        var_val = float(sorted_r[var_idx])
        cvar_val = float(sorted_r[: var_idx + 1].mean()) if var_idx >= 0 else var_val

        # ── Drawdown ───────────────────────────────────────────
        dd_info = self._compute_drawdowns(cum)

        # ── Calmar ─────────────────────────────────────────────
        calmar = 0.0
        if abs(dd_info.max_drawdown) > 1e-15:
            calmar = cagr / abs(dd_info.max_drawdown)

        # ── Win/loss ───────────────────────────────────────────
        wins = r_vals[r_vals > 0]
        losses = r_vals[r_vals < 0]
        win_rate = len(wins) / n if n > 0 else 0.0
        sum_wins = float(wins.sum()) if len(wins) > 0 else 0.0
        sum_losses = float(abs(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = sum_wins / sum_losses if sum_losses > 1e-15 else 0.0
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        wl_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-15 else 0.0

        # ── Distribution ───────────────────────────────────────
        skew = self._skewness(r_vals)
        kurt = self._kurtosis(r_vals)

        # ── Rolling metrics ────────────────────────────────────
        window = cfg.rolling_window
        rolling_sharpe = pd.Series(dtype=float, index=r.index, name="rolling_sharpe")
        rolling_vol = pd.Series(dtype=float, index=r.index, name="rolling_vol")

        if n >= window:
            r_series = pd.Series(r_vals, index=r.index)
            roll_mean = r_series.rolling(window).mean()
            roll_std = r_series.rolling(window).std(ddof=1)
            rolling_vol = roll_std * math.sqrt(ann)
            rolling_vol.name = "rolling_vol"

            excess = roll_mean - rf_daily
            rolling_sharpe = (excess / roll_std * math.sqrt(ann)).where(
                roll_std > 1e-15, 0.0,
            )
            rolling_sharpe.name = "rolling_sharpe"

        return PerformanceResult(
            n_days=n,
            total_return=total_return,
            cagr=cagr,
            daily_mean=daily_mean,
            daily_std=daily_std,
            annualised_vol=ann_vol,
            downside_std=downside_std,
            best_day=best,
            worst_day=worst,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            var=var_val,
            cvar=cvar_val,
            drawdown=dd_info,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=wl_ratio,
            skewness=skew,
            kurtosis=kurt,
            rolling_sharpe=rolling_sharpe,
            rolling_vol=rolling_vol,
        )

    # ── Drawdown computation ───────────────────────────────────

    @staticmethod
    def _compute_drawdowns(cumulative: np.ndarray) -> DrawdownInfo:
        """Compute drawdown statistics from a cumulative wealth curve."""
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / running_max - 1.0

        max_dd = float(drawdowns.min())
        # Average drawdown when in drawdown
        in_dd = drawdowns[drawdowns < -1e-10]
        avg_dd = float(in_dd.mean()) if len(in_dd) > 0 else 0.0

        # Drawdown episodes: count transitions into/out of drawdown
        is_dd = drawdowns < -1e-10
        n_episodes = 0
        max_duration = 0
        current_duration = 0

        for i in range(len(is_dd)):
            if is_dd[i]:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                if current_duration > 0:
                    n_episodes += 1
                current_duration = 0
        if current_duration > 0:
            n_episodes += 1  # Final unclosed drawdown

        return DrawdownInfo(
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_duration_days=max_duration,
            n_drawdowns=n_episodes,
        )

    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        """Compute sample skewness."""
        n = len(x)
        if n < 3:
            return 0.0
        m = x.mean()
        s = x.std(ddof=1)
        if s < 1e-15:
            return 0.0
        return float((n / ((n - 1) * (n - 2))) * np.sum(((x - m) / s) ** 3))

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """Compute excess kurtosis."""
        n = len(x)
        if n < 4:
            return 0.0
        m = x.mean()
        s = x.std(ddof=1)
        if s < 1e-15:
            return 0.0
        k4 = float(np.mean(((x - m) / s) ** 4))
        # Excess kurtosis (subtract 3 for normal distribution baseline)
        return k4 - 3.0
