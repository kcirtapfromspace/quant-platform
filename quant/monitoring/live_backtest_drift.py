"""Live vs backtest performance drift monitor.

Detects when live strategy performance deviates from backtested
expectations, serving as an early warning system for:

  * **Execution leakage** — live costs exceeding backtest assumptions.
  * **Overfitting** — backtest alpha that fails to materialise live.
  * **Regime shift** — market conditions that differ from the backtest period.
  * **Data issues** — stale or incorrect live data feeds.

Key metrics:

  * **Cumulative return gap** — divergence between live and backtest equity.
  * **Rolling Sharpe gap** — difference in rolling Sharpe ratios.
  * **Tracking error** — annualised volatility of the return difference.
  * **Drift z-score** — how many standard deviations the gap is from zero.
  * **Alert level** — GREEN / YELLOW / RED based on drift severity.

Usage::

    from quant.monitoring.live_backtest_drift import (
        DriftMonitor,
        DriftConfig,
    )

    monitor = DriftMonitor(DriftConfig(
        yellow_threshold_z=1.5,
        red_threshold_z=2.5,
    ))
    result = monitor.analyze(live_returns, backtest_returns)
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class AlertLevel(Enum):
    """Performance drift alert level."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass
class DriftConfig:
    """Configuration for live vs backtest drift monitoring.

    Attributes:
        rolling_window:         Window for rolling statistics (trading days).
        yellow_threshold_z:     Z-score threshold for YELLOW alert.
        red_threshold_z:        Z-score threshold for RED alert.
        min_observations:       Minimum days of live data needed.
        sharpe_gap_warning:     Sharpe ratio gap that triggers warning.
    """

    rolling_window: int = 63
    yellow_threshold_z: float = 1.5
    red_threshold_z: float = 2.5
    min_observations: int = 20
    sharpe_gap_warning: float = 0.50


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DriftSnapshot:
    """Point-in-time drift metrics."""

    date: object  # datetime-like
    live_cum_return: float
    backtest_cum_return: float
    cum_return_gap: float
    rolling_sharpe_live: float
    rolling_sharpe_backtest: float
    sharpe_gap: float
    drift_z_score: float
    alert_level: AlertLevel


@dataclass
class DriftResult:
    """Complete live vs backtest drift analysis.

    Attributes:
        snapshots:              Time series of drift snapshots.
        current_alert:          Most recent alert level.
        current_z_score:        Most recent drift z-score.
        cum_return_gap:         Current cumulative return gap.
        sharpe_gap:             Current rolling Sharpe gap.
        tracking_error:         Annualised tracking error (live - backtest).
        max_z_score:            Worst z-score observed.
        max_z_date:             Date of worst z-score.
        days_in_yellow:         Number of days at YELLOW or worse.
        days_in_red:            Number of days at RED.
        n_days:                 Total observation days.
        live_sharpe:            Full-period live Sharpe.
        backtest_sharpe:        Full-period backtest Sharpe.
    """

    snapshots: list[DriftSnapshot] = field(repr=False, default_factory=list)
    current_alert: AlertLevel = AlertLevel.GREEN
    current_z_score: float = 0.0
    cum_return_gap: float = 0.0
    sharpe_gap: float = 0.0
    tracking_error: float = 0.0
    max_z_score: float = 0.0
    max_z_date: object = None
    days_in_yellow: int = 0
    days_in_red: int = 0
    n_days: int = 0
    live_sharpe: float = 0.0
    backtest_sharpe: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Live vs Backtest Drift Monitor ({self.n_days} days)",
            "=" * 60,
            "",
            f"Current alert           : {self.current_alert.value}",
            f"Current z-score         : {self.current_z_score:+.2f}",
            f"Cumulative return gap   : {self.cum_return_gap:+.4f}",
            f"Rolling Sharpe gap      : {self.sharpe_gap:+.2f}",
            f"Tracking error (ann)    : {self.tracking_error:.4f}",
            "",
            f"Live Sharpe             : {self.live_sharpe:.2f}",
            f"Backtest Sharpe         : {self.backtest_sharpe:.2f}",
            "",
            f"Max z-score             : {self.max_z_score:+.2f} ({self.max_z_date})",
            f"Days in YELLOW+         : {self.days_in_yellow}",
            f"Days in RED             : {self.days_in_red}",
        ]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class DriftMonitor:
    """Monitors divergence between live and backtested performance.

    Args:
        config: Drift monitoring configuration.
    """

    def __init__(self, config: DriftConfig | None = None) -> None:
        self._config = config or DriftConfig()

    @property
    def config(self) -> DriftConfig:
        return self._config

    def analyze(
        self,
        live_returns: pd.Series,
        backtest_returns: pd.Series,
    ) -> DriftResult:
        """Analyze drift between live and backtest returns.

        Args:
            live_returns:     Daily returns from live trading.
            backtest_returns: Daily returns from backtest over the same period.

        Returns:
            :class:`DriftResult` with drift metrics and alert levels.

        Raises:
            ValueError: If fewer than ``min_observations`` common dates.
        """
        cfg = self._config

        common = live_returns.index.intersection(backtest_returns.index)
        if len(common) < cfg.min_observations:
            raise ValueError(
                f"Need at least {cfg.min_observations} common dates, "
                f"got {len(common)}"
            )

        live = live_returns.reindex(common).fillna(0.0)
        bt = backtest_returns.reindex(common).fillna(0.0)

        # Return difference series
        diff = live - bt
        n_days = len(common)

        # Cumulative returns
        live_cum = (1 + live).cumprod() - 1
        bt_cum = (1 + bt).cumprod() - 1

        # Full-period Sharpe ratios
        live_sharpe = self._sharpe(live)
        bt_sharpe = self._sharpe(bt)

        # Rolling statistics
        snapshots: list[DriftSnapshot] = []
        window = min(cfg.rolling_window, n_days)

        # Expanding z-score: mean(diff) / se(diff)
        diff_mean = diff.expanding(min_periods=cfg.min_observations).mean()
        diff_std = diff.expanding(min_periods=cfg.min_observations).std(ddof=1)
        diff_count = diff.expanding(min_periods=cfg.min_observations).count()
        z_scores = diff_mean / (diff_std / np.sqrt(diff_count))
        z_scores = z_scores.fillna(0.0)

        # Rolling Sharpe for live and backtest
        roll_live_sharpe = self._rolling_sharpe(live, window, cfg.min_observations)
        roll_bt_sharpe = self._rolling_sharpe(bt, window, cfg.min_observations)

        for date in common:
            z = float(z_scores.loc[date])
            abs_z = abs(z)

            if abs_z >= cfg.red_threshold_z:
                alert = AlertLevel.RED
            elif abs_z >= cfg.yellow_threshold_z:
                alert = AlertLevel.YELLOW
            else:
                alert = AlertLevel.GREEN

            rl_live = float(roll_live_sharpe.get(date, 0.0))
            rl_bt = float(roll_bt_sharpe.get(date, 0.0))

            snapshots.append(DriftSnapshot(
                date=date,
                live_cum_return=float(live_cum.loc[date]),
                backtest_cum_return=float(bt_cum.loc[date]),
                cum_return_gap=float(live_cum.loc[date] - bt_cum.loc[date]),
                rolling_sharpe_live=rl_live,
                rolling_sharpe_backtest=rl_bt,
                sharpe_gap=rl_live - rl_bt,
                drift_z_score=z,
                alert_level=alert,
            ))

        # Tracking error
        te = float(diff.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(diff) > 1 else 0.0

        # Aggregate alerts
        days_yellow = sum(1 for s in snapshots if s.alert_level != AlertLevel.GREEN)
        days_red = sum(1 for s in snapshots if s.alert_level == AlertLevel.RED)

        # Max z-score
        abs_z_series = z_scores.abs()
        max_z_idx = abs_z_series.idxmax() if len(abs_z_series) > 0 else None
        max_z = float(z_scores.loc[max_z_idx]) if max_z_idx is not None else 0.0

        current = snapshots[-1] if snapshots else None

        return DriftResult(
            snapshots=snapshots,
            current_alert=current.alert_level if current else AlertLevel.GREEN,
            current_z_score=current.drift_z_score if current else 0.0,
            cum_return_gap=current.cum_return_gap if current else 0.0,
            sharpe_gap=current.sharpe_gap if current else 0.0,
            tracking_error=te,
            max_z_score=max_z,
            max_z_date=max_z_idx,
            days_in_yellow=days_yellow,
            days_in_red=days_red,
            n_days=n_days,
            live_sharpe=live_sharpe,
            backtest_sharpe=bt_sharpe,
        )

    @staticmethod
    def _sharpe(returns: pd.Series) -> float:
        """Annualised Sharpe ratio (assuming rf=0)."""
        if len(returns) < 2:
            return 0.0
        std = float(returns.std(ddof=1))
        if std < 1e-10:
            return 0.0
        return float(returns.mean() / std * np.sqrt(TRADING_DAYS))

    @staticmethod
    def _rolling_sharpe(
        returns: pd.Series, window: int, min_periods: int,
    ) -> pd.Series:
        """Rolling annualised Sharpe ratio."""
        roll_mean = returns.rolling(window, min_periods=min_periods).mean()
        roll_std = returns.rolling(window, min_periods=min_periods).std(ddof=1)
        roll_std = roll_std.replace(0.0, np.nan)
        return (roll_mean / roll_std * np.sqrt(TRADING_DAYS)).fillna(0.0)
