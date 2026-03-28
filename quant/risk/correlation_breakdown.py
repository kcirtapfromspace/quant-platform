"""Correlation breakdown detection for portfolio risk monitoring.

Detects structural spikes in cross-asset or cross-strategy correlations
that signal diversification collapse.  In crises, correlations converge
toward 1 — destroying the diversification benefit the CIO relies on.

Key diagnostics:

  * **Rolling average correlation**: time series of mean pairwise corr.
  * **Breakdown events**: periods where avg corr exceeds a threshold.
  * **Absorption ratio**: fraction of total variance explained by the
    top principal components (high = concentrated risk).
  * **Diversification ratio**: ratio of weighted-average asset vol to
    portfolio vol (higher = more diversification benefit).

Usage::

    from quant.risk.correlation_breakdown import (
        CorrelationBreakdownMonitor,
        BreakdownConfig,
    )

    monitor = CorrelationBreakdownMonitor()
    result = monitor.analyze(returns_df, window=63)
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BreakdownConfig:
    """Configuration for correlation breakdown detection.

    Attributes:
        window:                 Rolling window for correlation computation.
        min_periods:            Minimum periods for a valid rolling window.
        breakdown_threshold:    Avg correlation above this triggers a breakdown event.
        absorption_n_components: Number of top PCs for absorption ratio.
    """

    window: int = 63
    min_periods: int = 30
    breakdown_threshold: float = 0.70
    absorption_n_components: int = 1


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BreakdownEvent:
    """One period of elevated correlation."""

    start_date: object
    end_date: object
    duration_days: int
    peak_correlation: float
    avg_correlation: float


@dataclass
class BreakdownResult:
    """Complete correlation breakdown analysis."""

    n_assets: int
    n_days: int

    # Time series
    rolling_avg_corr: pd.Series = field(repr=False)
    rolling_absorption: pd.Series = field(repr=False)
    diversification_ratio: pd.Series = field(repr=False)

    # Events
    breakdown_events: list[BreakdownEvent]
    n_breakdowns: int
    total_breakdown_days: int
    pct_time_in_breakdown: float

    # Current state
    current_avg_corr: float
    current_absorption: float
    current_diversification_ratio: float
    is_in_breakdown: bool

    def summary(self) -> str:
        """Return a human-readable summary."""
        status = "BREAKDOWN" if self.is_in_breakdown else "Normal"
        lines = [
            f"Correlation Breakdown Monitor ({self.n_assets} assets, {self.n_days} days)",
            "=" * 65,
            "",
            f"Current status          : {status}",
            f"Current avg correlation  : {self.current_avg_corr:.3f}",
            f"Current absorption ratio : {self.current_absorption:.3f}",
            f"Current diversif. ratio  : {self.current_diversification_ratio:.2f}",
            "",
            f"Breakdown events        : {self.n_breakdowns}",
            f"Total breakdown days    : {self.total_breakdown_days}",
            f"Time in breakdown       : {self.pct_time_in_breakdown:.1%}",
        ]

        if self.breakdown_events:
            lines.extend(["", "Breakdown Events:", "-" * 65])
            lines.append(
                f"  {'Start':>12s}  {'End':>12s}  {'Days':>5s}  "
                f"{'Peak Corr':>10s}  {'Avg Corr':>10s}"
            )
            for ev in self.breakdown_events:
                lines.append(
                    f"  {str(ev.start_date)[:10]:>12s}  "
                    f"{str(ev.end_date)[:10]:>12s}  "
                    f"{ev.duration_days:>5d}  "
                    f"{ev.peak_correlation:>10.3f}  "
                    f"{ev.avg_correlation:>10.3f}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class CorrelationBreakdownMonitor:
    """Monitors correlation dynamics for breakdown events."""

    def __init__(self, config: BreakdownConfig | None = None) -> None:
        self._config = config or BreakdownConfig()

    def analyze(self, returns: pd.DataFrame) -> BreakdownResult:
        """Analyze correlation dynamics from asset returns.

        Args:
            returns: Daily returns (DatetimeIndex x assets/strategies).

        Returns:
            :class:`BreakdownResult` with rolling metrics and breakdown events.

        Raises:
            ValueError: If fewer than 2 assets or insufficient data.
        """
        cfg = self._config

        if returns.shape[1] < 2:
            raise ValueError("Need at least 2 assets/strategies")
        if len(returns) < cfg.min_periods:
            raise ValueError(
                f"Need at least {cfg.min_periods} days, got {len(returns)}"
            )

        n_assets = returns.shape[1]
        n_days = len(returns)
        filled = returns.fillna(0.0)

        # Compute rolling average pairwise correlation
        avg_corrs = self._rolling_avg_corr(filled, cfg.window, cfg.min_periods)

        # Compute rolling absorption ratio
        absorptions = self._rolling_absorption(
            filled, cfg.window, cfg.min_periods, cfg.absorption_n_components
        )

        # Compute rolling diversification ratio
        div_ratios = self._rolling_diversification_ratio(
            filled, cfg.window, cfg.min_periods
        )

        # Detect breakdown events
        events = self._detect_breakdowns(avg_corrs, cfg.breakdown_threshold)
        total_bd_days = sum(ev.duration_days for ev in events)

        # Current state
        last_corr = float(avg_corrs.iloc[-1]) if not avg_corrs.empty else 0.0
        last_abs = float(absorptions.iloc[-1]) if not absorptions.empty else 0.0
        last_div = float(div_ratios.iloc[-1]) if not div_ratios.empty else 1.0
        in_breakdown = last_corr >= cfg.breakdown_threshold

        valid_days = int(avg_corrs.notna().sum())

        return BreakdownResult(
            n_assets=n_assets,
            n_days=n_days,
            rolling_avg_corr=avg_corrs,
            rolling_absorption=absorptions,
            diversification_ratio=div_ratios,
            breakdown_events=events,
            n_breakdowns=len(events),
            total_breakdown_days=total_bd_days,
            pct_time_in_breakdown=total_bd_days / valid_days if valid_days > 0 else 0.0,
            current_avg_corr=last_corr,
            current_absorption=last_abs,
            current_diversification_ratio=last_div,
            is_in_breakdown=in_breakdown,
        )

    @staticmethod
    def _rolling_avg_corr(
        returns: pd.DataFrame, window: int, min_periods: int,
    ) -> pd.Series:
        """Compute rolling average pairwise correlation."""
        n = len(returns)
        n_assets = returns.shape[1]
        values = returns.values.astype(float)
        result = np.full(n, np.nan)

        for i in range(min_periods - 1, n):
            start = max(0, i - window + 1)
            chunk = values[start : i + 1]
            if chunk.shape[0] < min_periods:
                continue
            corr = np.corrcoef(chunk.T)
            # Average of upper triangle (excluding diagonal)
            mask = np.triu(np.ones((n_assets, n_assets), dtype=bool), k=1)
            upper = corr[mask]
            if len(upper) > 0:
                result[i] = float(np.nanmean(upper))

        return pd.Series(result, index=returns.index, name="avg_corr")

    @staticmethod
    def _rolling_absorption(
        returns: pd.DataFrame, window: int, min_periods: int,
        n_components: int,
    ) -> pd.Series:
        """Compute rolling absorption ratio (top-N PC variance / total)."""
        n = len(returns)
        values = returns.values.astype(float)
        result = np.full(n, np.nan)

        for i in range(min_periods - 1, n):
            start = max(0, i - window + 1)
            chunk = values[start : i + 1]
            if chunk.shape[0] < min_periods:
                continue
            try:
                cov = np.cov(chunk.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
                total_var = eigenvalues.sum()
                if total_var > 1e-12:
                    top_n = eigenvalues[:n_components].sum()
                    result[i] = float(top_n / total_var)
            except np.linalg.LinAlgError:
                continue

        return pd.Series(result, index=returns.index, name="absorption_ratio")

    @staticmethod
    def _rolling_diversification_ratio(
        returns: pd.DataFrame, window: int, min_periods: int,
    ) -> pd.Series:
        """Compute rolling diversification ratio.

        DR = sum(w_i * sigma_i) / sigma_portfolio
        where equal weights are assumed. DR > 1 means diversification is working.
        """
        n = len(returns)
        n_assets = returns.shape[1]
        values = returns.values.astype(float)
        w = np.ones(n_assets) / n_assets
        result = np.full(n, np.nan)

        for i in range(min_periods - 1, n):
            start = max(0, i - window + 1)
            chunk = values[start : i + 1]
            if chunk.shape[0] < min_periods:
                continue
            try:
                cov = np.cov(chunk.T)
                asset_vols = np.sqrt(np.diag(cov))
                weighted_avg_vol = float(w @ asset_vols)
                port_var = float(w @ cov @ w)
                port_vol = np.sqrt(port_var) if port_var > 0 else 0.0
                result[i] = weighted_avg_vol / port_vol if port_vol > 1e-12 else 1.0
            except np.linalg.LinAlgError:
                continue

        return pd.Series(result, index=returns.index, name="diversification_ratio")

    @staticmethod
    def _detect_breakdowns(
        avg_corr: pd.Series, threshold: float,
    ) -> list[BreakdownEvent]:
        """Detect periods where avg correlation exceeds threshold."""
        events: list[BreakdownEvent] = []
        values = avg_corr.values
        index = avg_corr.index
        n = len(values)

        in_event = False
        start_idx = 0
        peak_corr = 0.0
        corr_sum = 0.0
        count = 0

        for i in range(n):
            v = values[i]
            if np.isnan(v):
                if in_event:
                    events.append(BreakdownEvent(
                        start_date=index[start_idx],
                        end_date=index[i - 1],
                        duration_days=count,
                        peak_correlation=peak_corr,
                        avg_correlation=corr_sum / count if count > 0 else 0.0,
                    ))
                    in_event = False
                continue

            if v >= threshold:
                if not in_event:
                    in_event = True
                    start_idx = i
                    peak_corr = v
                    corr_sum = v
                    count = 1
                else:
                    peak_corr = max(peak_corr, v)
                    corr_sum += v
                    count += 1
            elif in_event:
                events.append(BreakdownEvent(
                    start_date=index[start_idx],
                    end_date=index[i - 1],
                    duration_days=count,
                    peak_correlation=peak_corr,
                    avg_correlation=corr_sum / count if count > 0 else 0.0,
                ))
                in_event = False

        # Close trailing event
        if in_event:
            events.append(BreakdownEvent(
                start_date=index[start_idx],
                end_date=index[n - 1],
                duration_days=count,
                peak_correlation=peak_corr,
                avg_correlation=corr_sum / count if count > 0 else 0.0,
            ))

        return events
