"""Drawdown recovery analysis for backtest evaluation.

Goes beyond max drawdown to give PMs a full picture of drawdown behaviour:

  * **Drawdown episodes**: every peak-to-trough-to-recovery cycle.
  * **Recovery statistics**: average and max recovery time, conditional
    recovery probability.
  * **Underwater duration**: time spent below high-water mark.
  * **Tail drawdown risk**: percentile-based worst-case drawdown depth.

Usage::

    from quant.backtest.drawdown_analysis import DrawdownAnalyzer

    analyzer = DrawdownAnalyzer()
    result = analyzer.analyze(equity_curve)
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DrawdownEpisode:
    """One peak-to-trough-to-recovery drawdown event.

    Attributes:
        peak_date:      Date of the high-water mark before the drawdown.
        trough_date:    Date of the maximum drawdown within this episode.
        recovery_date:  Date equity first recovers to the peak (or None
                        if still underwater at series end).
        depth:          Maximum drawdown depth as a positive fraction.
        duration_days:  Trading days from peak to recovery (or to end if
                        still underwater).
        drawdown_days:  Trading days from peak to trough.
        recovery_days:  Trading days from trough to recovery (or None).
    """

    peak_date: object
    trough_date: object
    recovery_date: object | None
    depth: float
    duration_days: int
    drawdown_days: int
    recovery_days: int | None


@dataclass
class DrawdownAnalysisResult:
    """Complete drawdown analysis results."""

    # Episode list
    episodes: list[DrawdownEpisode]
    n_episodes: int

    # Depth statistics
    max_drawdown: float
    avg_drawdown: float
    median_drawdown: float
    p95_drawdown: float  # 95th percentile drawdown depth

    # Duration statistics (in trading days)
    max_duration: int
    avg_duration: float
    median_duration: float
    max_recovery_time: int | None  # None if never recovered
    avg_recovery_time: float | None

    # Underwater statistics
    total_underwater_days: int
    pct_time_underwater: float  # fraction of total days
    longest_underwater: int

    # Recovery statistics
    n_recovered: int
    recovery_rate: float  # fraction of episodes that recovered

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Drawdown Analysis ({self.n_episodes} episodes)",
            "=" * 60,
            "",
            f"Max drawdown          : {self.max_drawdown:.2%}",
            f"Avg drawdown          : {self.avg_drawdown:.2%}",
            f"Median drawdown       : {self.median_drawdown:.2%}",
            f"95th pctile drawdown  : {self.p95_drawdown:.2%}",
            "",
            f"Max duration (days)   : {self.max_duration}",
            f"Avg duration (days)   : {self.avg_duration:.1f}",
            f"Max recovery (days)   : {self.max_recovery_time or 'N/A'}",
            f"Avg recovery (days)   : {self.avg_recovery_time:.1f}"
            if self.avg_recovery_time is not None
            else "Avg recovery (days)   : N/A",
            "",
            f"Time underwater       : {self.pct_time_underwater:.1%}",
            f"Longest underwater    : {self.longest_underwater} days",
            f"Recovery rate         : {self.recovery_rate:.1%}"
            f" ({self.n_recovered}/{self.n_episodes})",
        ]

        if self.episodes:
            lines.extend(["", "Top 5 Deepest Drawdowns:", "-" * 60])
            lines.append(
                f"  {'Depth':>7s}  {'Peak':>12s}  {'Trough':>12s}  "
                f"{'Recovery':>12s}  {'Duration':>8s}"
            )
            for ep in sorted(self.episodes, key=lambda e: e.depth, reverse=True)[:5]:
                rec = str(ep.recovery_date)[:10] if ep.recovery_date else "ongoing"
                lines.append(
                    f"  {ep.depth:>6.2%}  {str(ep.peak_date)[:10]:>12s}  "
                    f"{str(ep.trough_date)[:10]:>12s}  {rec:>12s}  "
                    f"{ep.duration_days:>7d}d"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DrawdownConfig:
    """Configuration for drawdown analysis.

    Attributes:
        min_depth:  Minimum drawdown depth to count as an episode
                    (filters out noise).  Default 0.5% (0.005).
    """

    min_depth: float = 0.005


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class DrawdownAnalyzer:
    """Drawdown recovery analyzer."""

    def __init__(self, config: DrawdownConfig | None = None) -> None:
        self._config = config or DrawdownConfig()

    def analyze(self, equity_curve: pd.Series) -> DrawdownAnalysisResult:
        """Analyze drawdown behaviour from an equity curve.

        Args:
            equity_curve: Portfolio value series (DatetimeIndex).

        Returns:
            :class:`DrawdownAnalysisResult` with episodes and statistics.

        Raises:
            ValueError: If equity curve has fewer than 2 data points.
        """
        if len(equity_curve) < 2:
            raise ValueError("equity_curve must have at least 2 data points")

        episodes = self._find_episodes(equity_curve)
        n_total = len(equity_curve)

        if not episodes:
            return DrawdownAnalysisResult(
                episodes=[],
                n_episodes=0,
                max_drawdown=0.0,
                avg_drawdown=0.0,
                median_drawdown=0.0,
                p95_drawdown=0.0,
                max_duration=0,
                avg_duration=0.0,
                median_duration=0.0,
                max_recovery_time=None,
                avg_recovery_time=None,
                total_underwater_days=0,
                pct_time_underwater=0.0,
                longest_underwater=0,
                n_recovered=0,
                recovery_rate=0.0,
            )

        depths = [ep.depth for ep in episodes]
        durations = [ep.duration_days for ep in episodes]
        recovered = [ep for ep in episodes if ep.recovery_date is not None]
        recovery_times = [ep.recovery_days for ep in recovered if ep.recovery_days is not None]

        # Underwater calculation from drawdown series
        rolling_max = equity_curve.cummax()
        dd = (equity_curve - rolling_max) / rolling_max
        underwater_mask = dd < -1e-10
        total_uw = int(underwater_mask.sum())

        # Longest underwater streak
        longest_uw = self._longest_streak(underwater_mask)

        return DrawdownAnalysisResult(
            episodes=episodes,
            n_episodes=len(episodes),
            max_drawdown=max(depths),
            avg_drawdown=float(np.mean(depths)),
            median_drawdown=float(np.median(depths)),
            p95_drawdown=float(np.percentile(depths, 95)) if len(depths) >= 2 else max(depths),
            max_duration=max(durations),
            avg_duration=float(np.mean(durations)),
            median_duration=float(np.median(durations)),
            max_recovery_time=max(recovery_times) if recovery_times else None,
            avg_recovery_time=float(np.mean(recovery_times)) if recovery_times else None,
            total_underwater_days=total_uw,
            pct_time_underwater=total_uw / n_total if n_total > 0 else 0.0,
            longest_underwater=longest_uw,
            n_recovered=len(recovered),
            recovery_rate=len(recovered) / len(episodes),
        )

    def _find_episodes(
        self, equity_curve: pd.Series
    ) -> list[DrawdownEpisode]:
        """Identify drawdown episodes from the equity curve."""
        values = equity_curve.values.astype(float)
        index = equity_curve.index
        n = len(values)
        min_depth = self._config.min_depth

        episodes: list[DrawdownEpisode] = []
        peak_val = values[0]
        peak_idx = 0
        trough_val = values[0]
        trough_idx = 0
        in_drawdown = False

        for i in range(1, n):
            v = values[i]

            if v >= peak_val and not in_drawdown:
                # New high-water mark — no active drawdown
                peak_val = v
                peak_idx = i
                trough_val = v
                trough_idx = i
                continue

            if v < trough_val:
                # Deeper trough
                trough_val = v
                trough_idx = i
                in_drawdown = True

            dd_depth = (peak_val - trough_val) / peak_val if peak_val > 0 else 0.0

            if v >= peak_val and in_drawdown:
                # Recovery — close the episode
                if dd_depth >= min_depth:
                    drawdown_days = trough_idx - peak_idx
                    recovery_days = i - trough_idx
                    episodes.append(
                        DrawdownEpisode(
                            peak_date=index[peak_idx],
                            trough_date=index[trough_idx],
                            recovery_date=index[i],
                            depth=dd_depth,
                            duration_days=i - peak_idx,
                            drawdown_days=drawdown_days,
                            recovery_days=recovery_days,
                        )
                    )
                # Reset
                peak_val = v
                peak_idx = i
                trough_val = v
                trough_idx = i
                in_drawdown = False

        # If still in drawdown at end of series
        if in_drawdown:
            dd_depth = (peak_val - trough_val) / peak_val if peak_val > 0 else 0.0
            if dd_depth >= min_depth:
                episodes.append(
                    DrawdownEpisode(
                        peak_date=index[peak_idx],
                        trough_date=index[trough_idx],
                        recovery_date=None,
                        depth=dd_depth,
                        duration_days=n - 1 - peak_idx,
                        drawdown_days=trough_idx - peak_idx,
                        recovery_days=None,
                    )
                )

        return episodes

    @staticmethod
    def _longest_streak(mask: pd.Series) -> int:
        """Find the longest consecutive True streak."""
        if mask.empty or not mask.any():
            return 0
        vals = mask.values
        longest = 0
        current = 0
        for v in vals:
            if v:
                current += 1
                if current > longest:
                    longest = current
            else:
                current = 0
        return longest
